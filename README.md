# polars-hf

Read and write [Hugging Face Hub buckets](https://huggingface.co/docs/hub/storage-buckets) with
[Polars](https://pola.rs), as a pure-Python **IO plugin**. No fork of Polars, no compiled
extensions — just install and scan.

> **Status:** alpha, pre-release (not on PyPI yet — install from git, see below).
> Reads (`scan_bucket`) and writes (`sink_bucket`, including partitioned) are implemented.

## Why

Stock Polars already reads `hf://datasets/...` and `hf://spaces/...` natively. It does **not** yet
read `hf://buckets/...`. `polars-hf` fills that gap from the outside.

The files are read by **native** `pl.scan_parquet` scans: bucket files are XET-backed, so
`polars-hf` follows the authenticated Hub `resolve` redirect to a presigned CDN URL
(`us.aws.cdn.hf.co/xet-bridge-*`) and hands that to Polars. Polars' own Rust object store then does
async, concurrent, **range-read** scans — so **projection, predicate, and row-limit pushdown**,
streaming, and multi-file concurrency work natively and only the column chunks actually needed
are transferred. (This is the same read
mechanism upstream's `hf://` reader uses; we just resolve the signed URL in Python because stock
Polars can't attach a bearer token to a generic `https://` URL.)

A presigned URL holds a signature and is valid for about one hour. `scan_bucket` therefore does not
put the URLs in the query plan. By default the plan holds URLs of a small server in your own
process (`http://127.0.0.1:PORT/...`) that redirects Polars to the presigned URLs, so the scan node
stays the native one. See [Read modes](#read-modes).

> [!NOTE]
> **This may be a stopgap.** Native `hf://buckets/...` support is proposed upstream in Polars —
> [pola-rs/polars#27611](https://github.com/pola-rs/polars/issues/27611) (reads) and
> [pola-rs/polars#26909](https://github.com/pola-rs/polars/issues/26909) (streaming sink). If those
> land, `polars-hf` becomes redundant; until then, it fills the gap from the outside.

## Install

Not on PyPI yet — install from git:

```bash
uv add "polars-hf @ git+https://github.com/davanstrien/polars-hf"
# or: pip install "git+https://github.com/davanstrien/polars-hf"
```

Requires `polars>=1.40,<3` and `huggingface_hub>=1.12,<3`. Writes that use no local disk need
`huggingface_hub>=1.19` (see [Writing](#writing)).

### On Hugging Face Jobs

Use a [PEP 723](https://peps.python.org/pep-0723/) inline-dependency script so the Job pulls the
plugin straight from git — no build, no PyPI:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["polars-hf @ git+https://github.com/davanstrien/polars-hf@main"]
# ///
import polars as pl
import polars_hf as plhf

plhf.scan_bucket("hf://buckets/me/data/*.parquet").filter(pl.col("score") > 0.5).collect()
```

```bash
hf jobs uv run --secrets HF_TOKEN --flavor cpu-upgrade my_script.py
```

See [`examples/run_on_hf_jobs.py`](examples/run_on_hf_jobs.py) for a runnable example.

## Usage

```python
import polars as pl
import polars_hf as plhf

# A single file, a glob, or a whole bucket/directory (every .parquet / .pq file below it):
lf = plhf.scan_bucket("hf://buckets/my-namespace/my-bucket/data/*.parquet")

df = (
    lf.filter(pl.col("label") == 1)   # predicate pushdown
      .select("text", "label")        # projection pushdown
      .head(100)                       # row-limit pushdown
      .collect()
)
```

`scan_bucket` returns a lazy `LazyFrame` and works with the streaming engine.

### Read modes

A presigned URL gives read access to one file for about 60 minutes to anyone who has it.
`scan_bucket(uri, resolve=...)` has three modes; they differ in where these URLs are.

| | `"redirect"` (default) | `"collect"` | `"now"` |
| --- | --- | --- | --- |
| LazyFrame | native `scan_parquet` node over `http://127.0.0.1:PORT/...` URLs | Polars IO-plugin node that holds the bucket paths | native `scan_parquet` node over the presigned URLs |
| In `explain()`, `serialize()`, error messages, `POLARS_VERBOSE` log | the local URL | the bucket paths (the verbose log of Polars prints the presigned URLs) | the presigned URLs, with their signature |
| Row count, `tail()`, slices | native: footers only | a row count reads one whole column of every file; `tail()` and offset slices scan all files | native: footers only |
| `scan_bucket()` sends | the listing | the listing | the listing and one `resolve` request per file |
| The plan is valid | while the process that made it runs, on its machine | in any process with a token (`cloudpickle` needed to serialize) | ~60 minutes, anywhere |
| Needs | a local server in the process; loopback not behind a proxy | nothing | `POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1` |
| `include_file_paths=` | `hf://` URIs | `hf://` URIs | presigned URLs |
| `hive_partitioning=` | works (the local URL has the bucket path) | finds no keys | finds no keys |

Without `resolve=`, the mode is the value of the environment variable `POLARS_HF_RESOLVE` if it is
set, else `"redirect"`.

#### `resolve="redirect"`

`scan_bucket` gives Polars one URL per file on a small HTTP server that runs in your process:

```
http://127.0.0.1:{port}/{random id}/valid-only-while-pid-{pid}-runs/{bucket path}
```

Polars scans these URLs natively. The server answers each request with `302 Location: <presigned
URL>`, and Polars follows the redirect and reads the bytes from the CDN itself, with its `Range`
header. No file data passes through Python. The server resolves the presigned URL of a file with
one `resolve` request (also when many requests for the file arrive together), keeps it, and
resolves it again when it is older than 30 minutes, so a LazyFrame does not expire. Every
`scan_bucket` call gets new URLs: a file that was replaced is read anew by the next call.

Measured on HF Jobs (`cpu-performance`, Polars 2.0.0rc2, 12 files, 28 GB, jobs
`6abf69f8fbc85ba682369709` and `6abf6c26fbc85ba6823699c8`; selective queries were noisy in both
arms):

| Query | `resolve="redirect"` | presigned URLs read directly |
| --- | --- | --- |
| row count, `select(pl.len())` | 0.5–0.6 s | 0.5–0.6 s |
| `head(5)` | 0.6–0.9 s | 0.5 s |
| `tail(5)` | 0.6–0.8 s | 0.6 s |
| full scan of four columns | 48.2–50.6 s | 46.8–48.1 s |
| one small column | 5.9–9.0 s | 8.6–22.1 s |
| filter and two columns | 4.1–13.2 s | 5.5–7.1 s |

**Security notes.**

- The server listens on `127.0.0.1` only, on a port that the operating system chooses, and starts
  with the first redirect-mode `scan_bucket` call of the process.
- The random id in a local URL is a capability: another local process (or user of the machine) that
  learns a local URL can get the presigned URL of that one file while your process runs. It gives
  nothing on another machine and nothing after your process has exited. Treat a plan of this mode
  like a local file path of data you can read, not like a secret.
- The server serves only the exact file paths of a `scan_bucket` call. It never resolves a path
  that a request names; a path outside the scan gets `404`.
- A request whose `Host` header is not `127.0.0.1:PORT` or `localhost:PORT` gets `403`, so a web
  page cannot reach the server through DNS rebinding.
- The token is used by the resolver in your process. It is not in a URL, in the plan or in an
  answer of the server.

**Limits of the mode.**

- **Proxy variables.** The HTTP client of Polars sends requests for `127.0.0.1` to a proxy when
  `HTTP_PROXY`, `http_proxy`, `ALL_PROXY` or `all_proxy` is set and `NO_PROXY` / `no_proxy` does
  not cover the loopback address. `scan_bucket` raises a `RuntimeError` in that case, before any
  request. Set `NO_PROXY=127.0.0.1,localhost` (with the entries you already have) or use
  `resolve="collect"`. The package does not change the environment.
- **Other processes.** A plan works only while the process that made it runs, on the same machine.
  Polars runs the plan itself, so the package gets no control when a plan is used elsewhere (a
  serialized plan in a new process, a worker on another machine, a child process after its parent
  has exited). Polars then cannot connect, retries for 5 to 15 s and raises an `OSError` that
  names the local URL; the `valid-only-while-pid-...-runs` part of the URL is there for that
  message. Use `resolve="collect"` for a plan that another process runs. After a `fork`, a new
  `scan_bucket` call in the child starts a server of the child.
- **Many LazyFrames.** The server keeps the file lists of the 1,024 `scan_bucket` calls that were
  used last. Polars copies plans freely, so the package cannot tell when a LazyFrame is no longer
  used; the bound is by count. A LazyFrame of an older call then fails with a `404` that says to
  call `scan_bucket` again.
- **A URL that the CDN refuses early.** The server does not see the answers of the CDN. If the CDN
  refuses a presigned URL that is less than 30 minutes old, queries on that LazyFrame fail until
  the URL is 30 minutes old; a new `scan_bucket` call resolves new URLs.
- **Rate limits.** The `resolve` requests of a query are sent while Polars waits for the answer of
  the local server, so a retry wait is made only if it ends within 20 s. A rate limit with a
  longer reset fails the query (see [Errors](#errors)). `resolve="collect"` waits up to 10 minutes
  per group of files.
- **Replaced files.** A LazyFrame reads a file through the URL it resolved first for up to 30
  minutes, also if the file is replaced in between.

#### `resolve="collect"`

The LazyFrame is a Polars [IO-plugin source](https://docs.pola.rs/user-guide/plugins/io_plugins/)
that holds the bucket paths and resolves the presigned URLs when the query runs. It needs no local
server, and its serialized plan runs in any process that has a token. A query runs like this:

1. **Schema.** When Polars first needs the schema (`collect()`, `collect_schema()`, `explain()`),
   the first file is resolved and its footer is read: one `resolve` request and one to three CDN
   requests (it depends on the Polars version), once per LazyFrame. The schema is the one a
   native scan of the first file has with your scan options.
2. **Groups.** The files are scanned in path order in groups of 64. The URLs of a group are
   resolved when the group before it is exhausted, so later groups get new URLs and a scan can run
   for longer than one hour. One group is one native multi-file scan that gets the projection, the
   predicate and the row limit of the query, and its DataFrames are passed on unchanged: the rows
   and their order are those of one native scan. A group must be scanned within the hour that its
   URLs are valid.
3. **Row limits.** A query with a row limit (`head(n)`) uses groups of 1, 4, 16, then 64 files and
   stops when it has its rows.
4. **URLs are used again for 5 minutes.** The LazyFrame keeps the URLs it resolved (in memory, at
   most one per file). A group uses the URL of a file again if it is less than 5 minutes old when
   the scan of the group starts, so a second query right after the first one sends no `resolve`
   request. An older URL is resolved again. If the CDN refuses a URL that was used again, the group
   is resolved and scanned once more, provided that it has not returned rows yet.

What the IO-plugin node costs. Polars pushes a projection, a predicate and `head(n)` into an IO
source, and nothing else. Measured on HF Jobs (`cpu-performance`, Polars 2.0.0rc2, 12 files,
28 GB, job `6abec96b404719ba3761a56b`):

| Query | `resolve="collect"` | native node |
| --- | --- | --- |
| row count, `select(pl.len())` | 30–33 s, 27.7 GB downloaded | 1.1 s, 0.07 GB downloaded |
| `tail(5)` | 19 GB peak memory | 2.6 GB peak memory |
| one small column | 3.3–4.8 s | 2.1–2.9 s |
| `head(5)` | 0.8 s | 0.7 s |
| full scan of four columns | 22 s | 24 s |

- **Row counts.** `lf.select(pl.len())` is not a metadata query in this mode: Polars asks an IO
  source for one column to count its rows. The package does not choose that column, and it can be
  the largest one (in the measurement above it was nearly all of the data). Use
  `plhf.count_rows(uri)`, or the default mode.
- `tail()` and slices with an offset scan all files.
- All files are one group (all URLs are resolved at the start of the query) with `row_index_name=`
  or `n_rows=`, and for a row limit *before* a predicate (`lf.head(n).filter(...)`).
- Every group is scanned with the schema of the first file of the scan (passed to the native scan
  as `schema=`), unless you give `schema=` yourself. `missing_columns=`, `extra_columns=` and
  `cast_options=` then apply to every file as in one native scan; the tests compare both on files
  with different columns and types. One difference remains for such files without these options:
  a row count, or a slice that lies outside of the rows, can be answered by the native node from
  the footers and raises a schema error in this mode, because this mode reads a column of every
  file.
- A consumer that stops reading a `collect_batches()` iterator does not stop the query: Polars
  keeps running it, so the following groups are still resolved and scanned.
- Polars prints the presigned URLs it scans to stderr when `POLARS_VERBOSE=1` is set. The package
  cannot prevent that in this mode: treat a verbose log as a secret for an hour.
- `LazyFrame.serialize()` needs the `cloudpickle` package (Polars pickles an IO source with it).
  The serialized plan holds the endpoint, the bucket paths and the scan options except
  `storage_options=` and `credential_provider=`, which can hold credentials. It does not hold the
  token or a presigned URL. A deserialized query therefore uses the token of its own environment,
  also if `token=` was given to `scan_bucket`, and runs without these two options.

#### `resolve="now"`

`scan_bucket` resolves every file and returns the native `scan_parquet` node over the presigned
URLs. The URLs, with their signature, are then in `explain()`, in `serialize()`, in the messages
of read errors, in the verbose log of Polars and in an `include_file_paths=` column. Each one
gives read access to one file for about 60 minutes to anyone who sees it. The mode is therefore
refused with a `ValueError` unless the environment variable
`POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1` is set. Nothing falls back to this mode. Collect within
the hour, and treat the plan and its logs as secrets.

#### Row counts

```python
plhf.count_rows("hf://buckets/my-namespace/my-bucket/data/*.parquet")
```

`count_rows` returns the number of rows as an `int`, from the parquet footers, in every mode and
Polars version. It has the path rules and the errors of `scan_bucket`, sends one `resolve` request
per file and keeps nothing. It does not compare the schemas of the files. In the default mode
`lf.select(pl.len())` reads the footers too.

### Writing

```python
# Single file (parquet/csv/ipc/ndjson; format inferred from the extension):
plhf.sink_bucket(lf, "hf://buckets/ns/name/out.parquet")

# Partitioned: pass a base prefix + partition options (native pl.PartitionBy):
plhf.sink_bucket(lf, "hf://buckets/ns/name/by_year", partition_by="year")
plhf.sink_bucket(lf, "hf://buckets/ns/name/shards", max_rows_per_file=1_000_000)

# A destination that exists is an error by default. To write there again, say how:
plhf.sink_bucket(lf, "hf://buckets/ns/name/out.parquet", mode="overwrite")   # replace the file
plhf.sink_bucket(more, "hf://buckets/ns/name/by_year", partition_by="year", mode="append")
plhf.sink_bucket(lf, "hf://buckets/ns/name/by_year", partition_by="year", mode="overwrite")
```

`sink_bucket` accepts a `LazyFrame` (streaming) or a `DataFrame` and runs the query before it
returns (`lazy=True` is rejected). Partitioned writes split by key, by size, or both.
`max_rows_per_file` is a limit. `max_bytes_per_file` is a target, not a limit: it is Polars'
`approximate_bytes_per_file`, an estimate made while the rows are written, so a file can be larger.

**Object names.** A partitioned write uses the names Polars writes to a local directory:
`key=value/` directories and files `00000000.parquet`, `00000001.parquet`, ... (the index is
hexadecimal; the extension is `.parquet`, `.csv`, `.ipc` or `.jsonl`). Key values are
percent-encoded like Polars does (`/`, `=`, `%`, `:`, space, control characters and non-ASCII
bytes), and a null key is `__HIVE_DEFAULT_PARTITION__`. With `mode="append"` every file name also
carries a token that is unique to the call: `00000000-1f0c9a52b7e3.parquet` (12 random hex
characters, the same for all files of one call). A partitioned write whose query returns no rows
writes one file with the schema and no rows, `{prefix}/00000000.{extension}` (with the token in
`mode="append"`), so the prefix can be scanned afterwards.

**`mode`** sets what happens if the destination exists. A single-file destination exists if there
is an object at that path. A partitioned destination exists if there is a file anywhere below the
base prefix.

| `mode` | Single file | Partitioned (base prefix) |
| --- | --- | --- |
| `"error"` (default) | `FileExistsError` if the object exists | `FileExistsError` if the destination exists |
| `"append"` | not possible: `ValueError` | new files are added under names that carry a random 48-bit run id per call; nothing is deleted |
| `"overwrite"` | the object is replaced | the new files are registered (a file with the same name is replaced), then the files that were below the prefix before the write, and that this call did not write, are deleted |

**A file and a directory of one name are refused, in every mode.** A single-file write to `out`
raises `FileExistsError` if there are objects below `out/`, and a partitioned write to `out/...`
raises `FileExistsError` if `out` is a file. The bucket could store both; `sink_bucket` does not
create such a pair. Only the destination itself is checked: an *ancestor* that is a file (a write to
`a/b/c.parquet` while `a` is a file) is not detected.

The checks run before anything is uploaded. An existence check costs one listing page of the
destination directory plus one exact check of the path, whatever the destination and its siblings
hold. The checks do not exclude a concurrent writer.

| Write | Requests before the upload |
| --- | --- |
| single file, `"error"` | 2: one `HEAD` of the path, one listing page of `path/` |
| single file, `"overwrite"` | 1: one listing page of `path/` |
| partitioned, `"error"` | 2: one `HEAD` of the prefix, one listing page of `prefix/` (1 at the bucket root) |
| partitioned, `"append"` | 1: one `HEAD` of the prefix (0 at the bucket root); the prefix is not listed |
| partitioned, `"overwrite"` | one `HEAD` of the prefix, then every listing page of `prefix/` |

These requests use the listing and the retry limits of the read path: bounded retries, a deadline
of 10 minutes per call, `PermissionError` for a token without access and `FileNotFoundError` for a
bucket that does not exist.

`"append"` gives the files of each call names with a random 48-bit run id. Two appends with the
same partition keys therefore keep the rows of both, and running the same job twice adds the rows
twice. A name collision with an existing object is negligible but not excluded by a check:
`sink_bucket` does not list the prefix to look for a name that exists already.

`"overwrite"` needs a directory below the bucket root: `hf://buckets/ns/name` and
`hf://buckets/ns/name/` are refused with a `ValueError`. It lists the prefix before the write and
deletes only files from that listing, so a file that another writer adds during the write is kept.
A query that returns no rows still deletes the previous files and leaves one file with the schema
and no rows.
A file that another writer *replaces* during the write is still deleted.

**Paths.** The Hub refuses a path with a backslash, an empty segment (`a//b`) or a `.` / `..`
segment. `sink_bucket` also refuses a control character in a path and a path segment of more than
255 bytes, with both backends, because the `"staged"` backend cannot create such a name on a local
file system. It raises a `ValueError` for such a destination before it sends a request. A partition
column name or value can produce such a path too (a backslash is not percent-encoded, and an
encoded value counts with its `key=` prefix towards the 255 bytes); the error names the `key=value`
segment, and no file of that write is registered.

**`backend`** sets how the output reaches the bucket:

| `backend` | How | Local disk | Needs |
| --- | --- | --- | --- |
| `"stream"` | each output file is streamed into Xet storage with `hf_xet` while Polars writes it, then all files are registered in the bucket | none for the output | `huggingface_hub>=1.19` and the `hf_xet` that it requires (installed with it on x86_64 and arm64) |
| `"staged"` | Polars writes the output to a temporary directory; the public `HfApi.batch_bucket_files` uploads and registers it | the size of the complete output | any supported `huggingface_hub` |

The default (`backend=None`) is the environment variable `POLARS_HF_SINK_BACKEND` if set, else
`"stream"` when the installed packages support it, else `"staged"`. An explicit `"stream"` that cannot run
raises an error; it does not fall back.

With both backends, the deletes of `mode="overwrite"` are sent to the bucket batch endpoint
(`/api/buckets/{id}/batch`) directly, not through `HfApi.batch_bucket_files`: `huggingface_hub` 1.x
does not report a rejected delete, and the direct request lets `sink_bucket` check the answer.

Notes on the `"stream"` backend:

- It uses private helpers of `huggingface_hub.utils._xet` and builds the
  `/api/buckets/{id}/batch` request itself, because the public API has no streaming upload. If a
  later `huggingface_hub` or `hf_xet` changes these, `sink_bucket` raises a `RuntimeError` that
  names the installed versions and points to `backend="staged"`; it does not fall back silently.
- A `KeyboardInterrupt` during a write aborts the process-wide Xet session, as `huggingface_hub`
  does in its own uploads. Other Xet uploads and downloads that run in the same process are
  cancelled.
- All upload streams stay open until the query ends, because Polars does not signal when one file is
  complete.

Notes on the `"staged"` backend:

- It stages in the directory named by `POLARS_HF_STAGING_DIR`, else in the system temporary
  directory, and removes the files when the write ends.
- With `huggingface_hub` 1.x, `HfApi.batch_bucket_files` does not report files that the bucket
  rejected. The backend then lists the destination once after the upload (one extra listing of the
  file's string prefix, or of the base prefix, per write) and raises if a file is missing or has
  another size. `huggingface_hub` 2.x reports rejected files itself, and no listing is made.

**Memory.** In laptop runs with outputs up to 4.5 GB, the `"stream"` backend used 0.5–1.8 GB more
memory than Polars alone. A bound on its memory use is not proven. See
[Measurements](#measurements).

**What a failure leaves behind.** With both backends, no file is registered in the bucket before the
Polars sink has finished without an error. If the query fails or is interrupted, the destination is
unchanged: an existing object keeps its content, and no partial or empty file appears. An upload
that fails before the first registration request also leaves the destination unchanged. A failure
of a registration request does not: see the list below.

A failed registration or delete request raises `polars_hf.BucketRegistrationError` (a
`RuntimeError`) with both backends, and so does a failed upload of the `"staged"` backend (an HTTP
error, a timeout or a connection error; the original error is the `__cause__`). Its `failures`
attribute lists the operations the bucket rejected. After a timeout or a connection error the
request may or may not have been applied: the message says so, and a listing of the destination
shows the state. An error of the `hf_xet` upload itself is raised as `hf_xet` reports it.

The write is **not transactional**, because the bucket API has no transactions:

- Files are registered in requests of at most 1,000 operations. If a write of more than 1,000 files
  fails between two requests, the files of the earlier requests stay in the bucket.
- If the bucket rejects single files of a request (a 200 answer with `failed` entries), it still
  applies the other files of that request, also in a write of fewer than 1,000 files. `sink_bucket`
  then raises an error that lists the rejected paths.
- `mode="overwrite"` deletes the stale files only after all new files are registered, in separate
  requests. If the process stops or a request fails between the registration and the end of the
  deletion, the prefix holds the new files and the remaining stale files, and (for a failed
  request) the error is raised.
- A reader can see a part of the new files while the requests are in progress.

#### Measurements

Peak RSS of one process, one run each, on one macOS laptop, with uncompressed incompressible data.
"Polars alone" is the same query written to a file object that discards the bytes.

| Output | Polars alone | `"staged"` | `"stream"` |
| --- | --- | --- | --- |
| 1 file, 2.2 GB | 1.05 GB | 1.58 GB | 1.78 GB |
| 1 file, 4.5 GB | 1.56 GB | not measured | 2.99 GB |
| 16 files, 2.2 GB | 1.80 GB | 1.51 GB | 3.56 GB |
| 2,000 files, 2.2 GB | 3.35 GB | 3.34 GB (13 s) | 4.90 GB (62 s) |

These numbers describe these runs only; they are not a bound.

### Authentication

By default the token is resolved by `huggingface_hub` (the `HF_TOKEN` environment variable or your
cached `hf auth login`). You can also pass one explicitly:

```python
plhf.scan_bucket("hf://buckets/ns/name/data.parquet", token="hf_...")
```

## Supported URIs

```
hf://buckets/{namespace}/{name}/{path}
```

`scan_bucket` reads `{path}` as:

1. **A glob**, if it has a glob character (`*`, `?` or `[`), for example `data/*.parquet` or
   `data/**/part-*.parquet`: every *file* that matches. `*`, `?` and `[...]` match inside one path
   segment; `**` must be a whole segment and matches any number of directories. Braces (`{a,b}`)
   are not expanded. A glob never passes a sub-directory to the scan, and it does not filter by
   extension: `data/*` also selects `data/notes.txt`, and Polars then fails at `collect()` because
   that file is not parquet. Use `data/*.parquet`, or `data` for the directory reading. A file
   whose name is exactly the pattern wins over the matches (`data[1].parquet` reads that file,
   not `data1.parquet`). A *directory* name with glob characters is not read literally: escape
   them with a character class — `[[]` for `[`, `[*]` for `*`, `[?]` for `?`. The files of the
   directory `run[1]/` are `run[[]1]/*.parquet`.
2. Else **a single file**, if a file with exactly this name exists — whatever its extension.
3. Else **a directory, or the whole bucket** when `{path}` is empty: every `.parquet` / `.pq` file
   below it, at any depth; the extension is matched case-insensitively. A trailing `/` forces
   this reading. A directory named `out.parquet/` is scanned as a directory. Only the files of
   that directory are listed: a sibling such as `train_full/` costs nothing when you scan `train`.

A glob selects files, so it cannot end with `/`: `data/*/` raises `ValueError`. Use
`data/*/*.parquet` or `data/**/*.parquet`.

Rules for the URI itself:

- Buckets have **no** revision concept, so `@revision` after the bucket name is rejected (matching
  the Hub). Below the bucket, `@` is a normal character: `.../exports/user@example.com.parquet`.
- An empty path segment (`a//b`), a `..` segment and whitespace at the end of the URI raise
  `ValueError`. These rules are part of the URI parser, so they apply to `sink_bucket` too.
- `hf://datasets/...` and `hf://spaces/...` are read natively by Polars — use
  `pl.scan_parquet(...)` for those.

### Hub requests

`scan_bucket` reads no file data. The call makes these requests to the Hub:

| `{path}` | Requests of `scan_bucket()` |
| --- | --- |
| one file, any extension | 1 `resolve` (HEAD) |
| a directory of N parquet files (`data`) | 1 `resolve` (answered "not found") + 1 listing per page of results |
| the same with a trailing slash (`data/`), or the whole bucket | 1 listing per page |
| a glob that selects N files | 1 listing per page |
| a path that does not exist | 1 `resolve` + 1 listing |

A glob whose only glob segment is the last one (`data/*.parquet`) lists that directory only. Other
globs list the subtree below the text before their first glob character. An invalid glob raises
before any request.

With `resolve="now"` the call then sends one `resolve` request for each of the N files. In the
other modes these requests are sent when a query runs.

`resolve="redirect"` (default): one `resolve` request for a file when Polars first asks the local
server for it, and one more when its URL is older than 30 minutes.

| Query on N files | `resolve` requests |
| --- | --- |
| a full scan, a projection, a filter, a row count | N (N - 1 for a single-file URI: the call resolved it) |
| `head(n)`, `tail(n)` | the files that Polars opens: 1 for `head(5)` on Polars 1.40; most of the files on Polars 1.44 and 2.0, which read footers ahead |
| a second query on the same LazyFrame within 30 minutes | 0 |

`resolve="collect"`:

| Query on N files | `resolve` requests |
| --- | --- |
| first use of the schema (`collect()`, `collect_schema()`, `explain()`) | 1, for the first file; 0 for a single-file URI |
| a full scan, a projection, a filter | N - 1: the URL of the first file is known from the schema |
| `head(n)` | the files of the groups it reads: 1, then 4, 16, 64 |
| a second query on the same LazyFrame | 0 for the files whose URL is less than 5 minutes old; 1 for each other file |

`count_rows(uri)` sends N, in the call.

`resolve` requests count in the Hub's "resolvers" rate limit. In the redirect mode Polars decides
when it opens a file, and the local server sends at most 32 `resolve` requests at once. In the
collect mode a query sends at most 64 at once, and the next 64 only when the group before is read
(except in its one-group cases, see [Read modes](#read-modes)).

**Retries.** A `408`, `429` or `5xx` answer to a `resolve` request or to a listing page is retried
up to 5 times; only the failed request is sent again. `scan_bucket` sends the listing requests
itself (it does not call `HfApi.list_bucket_tree`), so the same limits apply to every page and on
every supported `huggingface_hub` version. The wait before a retry is the
one the Hub asks for (rate-limit reset, `Retry-After`), else 1 s doubling up to 8 s. When the Hub
asks for more than 5 s, one warning per `scan_bucket` call (and per group of a query) announces
the wait, so a paused scan is not silent. If your warning filter turns warnings into errors, the message is logged (logger
`polars_hf.read`) instead, and the scan continues.

A wait is made only while it ends within 10 minutes. The 10 minutes start with the `scan_bucket`
call for the requests of the call. For the requests of a query they start again with the schema
read and with every group of files, so the limit is per group, not per query. A rate-limit
reset in 300 s is waited for; a wait that would pass the 10 minutes raises `HfHubHTTPError` at
once. The message says how long the Hub asked to wait and how much time was left; for a rate
limit it also names the bucket, the quota and how many files were already resolved. That limit
bounds the waits, not the requests: a request already in flight can still run to its 30 s timeout
after it. A `Retry-After` that is an HTTP date (or not a finite number) is ignored and the backoff
is used. Timeouts and connection errors are not retried.

All requests use the shared HTTP session of `huggingface_hub`, so `HF_HUB_OFFLINE=1` and a custom
client factory (`huggingface_hub.set_client_factory`) apply.

### Errors

| Situation | Exception |
| --- | --- |
| The bucket does not exist, or nothing matches the URI | `FileNotFoundError` (the URI is in the message) |
| The Hub answers `401` / `403` | `PermissionError` naming the bucket; the `HfHubHTTPError` is its `__cause__` |
| A matched file is empty (0 bytes) | `ValueError` naming the file |
| A glob ends with `/`, or uses `**` inside a segment (`data/**.parquet`) | `ValueError` |
| The Hub serves a file itself instead of redirecting to a presigned URL (a file that is not Xet-backed) | `RuntimeError`; Polars cannot read a URL that needs the token |
| Any other HTTP error, or a `408` / `429` / `5xx` that the retries did not clear | `huggingface_hub.errors.HfHubHTTPError` |
| A listing answer is not what the Hub API documents (not a JSON list, a next link that leaves the Hub or repeats a page) | `RuntimeError` |
| A listing with many pages does not finish within 10 minutes | `TimeoutError` |
| A timeout or a connection error | the exception of the HTTP library (`httpx` / `httpx2`), not retried |

A private bucket that the token cannot see is reported by the Hub as "not found", so it raises
`FileNotFoundError`, not `PermissionError`.

**Which call raises.** `scan_bucket()` raises the errors of the URI, of the listing and of a
single-file URI: an invalid URI or glob, an unknown `resolve` mode, a missing bucket or path,
nothing matched, an empty file found in the listing, no access. With `resolve="now"` it raises all
errors of the table. In the other modes the `resolve` requests of the listed files are sent by the
query, so `collect()` reports their failures.

**Errors of a query, `resolve="redirect"`.** Polars reads through the local server, so every
failure is an `OSError` of Polars' HTTP client (`object-store error: ...`) that names the local
URL of the file, on Polars 1.40, 1.44 and 2.0.0rc2 alike. The status in the message tells the
cause. Polars 2 also prints the text that the local server sends (`polars-hf: ...`); Polars 1.x
asks with `HEAD` first and prints no text. In every version the reason is logged with the logger
`polars_hf._redirect` at level `WARNING`.

| Situation | What `collect()` raises |
| --- | --- |
| A file was deleted after the listing | `OSError`, `404 Not Found`, at once; the log names the `hf://` URI |
| The token lost access (`401` / `403` from the Hub) | `OSError`, `403 Forbidden`, at once |
| The Hub rate limit, with a reset later than 20 s | `OSError`, `424 Failed Dependency`, at once; the log has the quota (Polars would retry a `429`) |
| An empty file, or a file that the Hub does not redirect | `OSError`, `424 Failed Dependency`, at once |
| The Hub answers `5xx` / `408`, or cannot be reached, after the package's retries | `OSError`, `503 Service Unavailable`, after the 10 retries of Polars (10 to 30 s) |
| The CDN refuses the presigned URL (`403`, `416`) | `OSError` with that status, at once |
| The CDN fails (`5xx`, dropped connection, truncated body) | `OSError`, after the retries of Polars (5 to 20 s) |
| The plan is used after its process has exited | `OSError`, `error sending request`, after 5 to 15 s |
| The LazyFrame is older than the 1,024 scans used last | `OSError`, `404 Not Found`, at once |

None of these messages holds a presigned URL: Polars names the URL it was given, not the redirect
target. The tests check this for each case above, and for the verbose log of Polars.

**Errors of a query, `resolve="collect"`.** `collect()` raises the errors of the table at the top:
a file that was deleted after the listing (`FileNotFoundError`), a `401` / `403`
(`PermissionError`), a rate limit or server error that the retries did not clear
(`HfHubHTTPError`), a file that the Hub does not redirect (`RuntimeError`). The messages are the
same. How the exception arrives depends on Polars:

- Polars 2 (tested with 2.0.0rc2) raises the exception itself.
- Polars 1.x wraps every exception of an IO source: `collect()` raises
  `polars.exceptions.ComputeError`, and its message holds the type name and the message of the
  original exception (`... FileNotFoundError: no such file: 'hf://buckets/...'`).
- All Polars versions wrap an error of the schema read (the `resolve` request of the first file)
  in a `ComputeError` that starts with `schema callable failed`.

A read error of Polars itself (a presigned URL that the CDN refuses, a file that is not parquet)
keeps its type. Its message names the `hf://` URI of the file; the presigned URL is removed from
the message, and the exception has no `__cause__` or `__context__` that holds it.

## Performance

Bucket reads fetch range requests from the XET CDN.

- **The default mode is the native scan.** The local redirect adds one request to `127.0.0.1` per
  request of Polars. On HF Jobs (12 files, 28 GB, Polars 2.0.0rc2) a full scan of four columns
  took 48.2–50.6 s through the redirect and 46.8–48.1 s on the presigned URLs directly, and a row
  count 0.5–0.6 s in both; [Read modes](#read-modes) has the table. The server must accept many
  connections at once: a prototype with a listen backlog of 5 took 165–264 s for the selective
  queries, so the server is an `asyncio` server with a backlog of 4,096.
- **The IO-plugin node (`resolve="collect"`).** The native scan runs inside an IO source, and whole
  DataFrames cross into Python. On HF Jobs (`cpu-performance`, Polars 2.0.0rc2, 54 files, 126.7 GB,
  four columns, CDN-warm, one run each) a prototype of this wrapper took 244 s and the native node
  301 s, with the same result: no overhead was measured for a full scan. Queries that Polars does
  not push into an IO source cost more: a row count reads a whole column, and `tail()` and slices
  with an offset scan all files.
- **Download, then scan.** In the same run, downloading the files with `hf_xet` in a rolling window
  and scanning them from local disk took 114 s (7.3 GB peak RSS, 15.1 GB of staging disk).
  `polars-hf` does not do this; [`benchmarks/read_paths.py`](benchmarks/read_paths.py) has that arm
  for comparison.
- **Cold vs warm CDN.** The *first* read of freshly written/copied data pays a cold-CDN penalty
  (the bytes aren't at the edge yet); subsequent reads are much faster. For repeated large-scale
  reads, consider [pre-warming](https://huggingface.co/docs/hub/storage-buckets#pre-warming-and-cdn)
  the bucket.
- **`POLARS_CONCURRENCY_BUDGET`.** `polars-hf` does not set Polars environment variables. Earlier
  versions set `POLARS_CONCURRENCY_BUDGET=64` at import, for every cloud scan of the process. On
  HF Jobs (14 GB) that setting made no difference on Polars 2.0.0rc2 (31.0 and 32.1 s with it,
  29.1 and 33.6 s without) and about 8% on Polars 1.44.2
  ([job ids](benchmarks/README.md#polars_concurrency_budget)). On Polars 1.x you can set
  it yourself, before the first cloud read of the process.

[`benchmarks/`](benchmarks/README.md) has the scripts, what each number means, and the measured
tables with their job ids.

## Scan options

Extra keyword arguments to `scan_bucket` are forwarded to `pl.scan_parquet` (in the collect mode:
to the native scan of every group) — e.g. heterogeneous schemas across globbed files:

```python
plhf.scan_bucket(uri, missing_columns="insert", extra_columns="ignore")
```

or `storage_options={"max_retries": 5}` for flaky connections (this replaces the deprecated
`retries=` option of Polars; it applies to the data requests Polars makes, not to the Hub requests
above).

`include_file_paths="file"` adds the `hf://buckets/...` URI of the file of every row, in the
redirect and the collect mode. (With `resolve="now"` the column holds the presigned URL, signature
included.) In the collect mode the URI is found from the presigned URL of the file: if two files
with the same content get the same URL and are in the same group of 64 files, their rows show the
URI of one of them.

`hive_partitioning=True` reads `key=value` directories of the bucket path in the redirect mode,
because the path of a local URL is the bucket path. In the other modes Polars sees the presigned
URLs, which do not have the bucket path, and finds no partition columns.

## Limitations

- Reads cover parquet; writes cover parquet/csv/ipc/ndjson. Delta/Iceberg are out of scope.
- The default read mode needs a local HTTP server in the process and a loopback address that is
  not behind a proxy, and its plans are valid only while the process runs. `resolve="collect"` has
  neither limit, but it is a Polars IO-plugin node: a row count reads one whole column
  (`plhf.count_rows` reads the footers), `tail()` and slices with an offset scan all files, hive
  partition columns are not inferred, and on Polars 1.x an error of a query is a `ComputeError`
  that quotes the original exception. See [Read modes](#read-modes) and [Errors](#errors).

## Development

```bash
uv sync
uv run ruff check .
uv run pytest                 # offline suite (the default)
uv run pytest -m staging      # live tests against the Hub CI staging instance
```

`addopts` in `pyproject.toml` deselects the staging tests by default. A command such as
`uv run pytest tests/test_read.py` therefore selects nothing: add `-m staging`.

**Offline tests** need no network and no token. `tests/fakehub.py` runs a local fake Hub: one HTTP
server for the bucket API and the `resolve` redirect, and a second one (another origin) that serves
the "presigned" URLs with range requests. It records every request and the bytes it serves, and can
be scripted to fail (`429`, `503`, `403`, ...), and to refuse the presigned URLs it has made so far
(`expire_signed_urls()`), like the real CDN after an hour, or to break a connection
(`action="reset"`, `action="truncate"`). File uploads do not go over HTTP; the fake has one
seam per sink backend. For the `"staged"` backend, a patched `HfApi._batch_bucket_files` stores the
files in memory, so the client-side chunking of the public method stays real. For the `"stream"`
backend, an in-memory object replaces the `hf_xet` upload commit, and the backend's own
registration request goes to the fake's `POST /api/buckets/{id}/batch` route. The fake also reports
the `"stream"` backend as available, so the offline suite runs both backends with every supported
`huggingface_hub`. The real `hf_xet` upload is covered by the staging tests only. Use the
`fake_hub` / `fake_bucket` fixtures. `tests/sinks.py` is the one place that names the write design
(backends and failure switches); write tests go through its helpers.

**Staging tests** (`-m staging`) do real round-trips against `https://hub-ci.huggingface.co`, the
instance `huggingface_hub` uses for its own tests. `tests/conftest.py` sets `HF_ENDPOINT` and
`HF_TOKEN` to the staging endpoint and its public CI token *before* `huggingface_hub` is imported,
so the test process never talks to `huggingface.co` and never reads your own token. Each test
creates a uniquely named bucket and deletes it afterwards. `tests/test_staging_sinks.py` runs the
same write scenarios for both sink backends; the `stream` cases are skipped when the installed
`huggingface_hub` is older than 1.19. `test_local_disk_use` measures the growth of the temporary
directory during a write (16 MB by default; `POLARS_HF_DISK_TEST_MB=300 uv run pytest -m staging -k
test_local_disk_use -s` prints a larger measurement).
`test_big_write_is_identical_with_both_backends` uploads about 300 MB with each backend and compares
the sizes and Xet hashes of all objects; it runs only with `POLARS_HF_BIG_STAGING=1`, which the weekly
CI run sets. Staging can answer
`409`/`502`/`503`/`504` or time out; a test that fails with one of these is rerun automatically
(`pytest-rerunfailures`). Other failures are not rerun.

**Property tests** (`tests/test_properties.py`) use [Hypothesis](https://hypothesis.readthedocs.io)
with a derandomized profile, so every run executes the same examples. For a randomized pass, select
the `random` profile (`--hypothesis-seed` has an effect only with this profile):

```bash
HYPOTHESIS_PROFILE=random uv run pytest tests/test_properties.py
HYPOTHESIS_PROFILE=random uv run pytest tests/test_properties.py --hypothesis-seed=1234
```

**Known bugs** go in `tests/test_known_bugs.py` as `@pytest.mark.xfail(strict=True)` tests (none is
open at the moment; the tests there are regression tests of fixed bugs). Such a test
asserts the behaviour we *want* and fails today for the reason in its `reason=`, so it is reported
as `xfailed`. Because the marker is strict, a test that starts to pass turns the suite red
(`XPASS(strict)`): the pull request that fixes a bug must remove the marker in the same change. To
see the current failure of such a test, run it with `--runxfail`.

CI runs the offline suite on Python 3.10–3.14 with the locked dependencies, and again with the
lowest supported direct dependencies, the latest releases, and the newest Polars 2 pre-release
(that last job is allowed to fail). The staging suite runs on pull requests and on pushes to `main`. A weekly
scheduled run repeats all jobs with the `random` Hypothesis profile.

## License

MIT
