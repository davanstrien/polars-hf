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
put the URLs in the query plan: it returns a LazyFrame over a Polars
[IO-plugin source](https://docs.pola.rs/user-guide/plugins/io_plugins/) that resolves the URLs when
the query runs and delegates to the native scan. See
[When the URLs are resolved](#when-the-urls-are-resolved).

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

### When the URLs are resolved

`scan_bucket(uri, resolve=...)` has two modes:

| | `resolve="collect"` (default) | `resolve="now"` |
| --- | --- | --- |
| LazyFrame | a Polars IO-plugin node that holds the bucket paths | the native `scan_parquet` node over the presigned URLs |
| `scan_bucket()` does | the listing (see [Hub requests](#hub-requests)) | the listing and one `resolve` request per file |
| Presigned URLs in `explain()`, `serialize()`, error messages | no | yes, with their signature |
| Validity | a URL is resolved shortly before it is used, each time the query runs | ~1 hour from the `scan_bucket` call; call it again for a new plan |
| `head(5)` on N files | resolves 1 file | resolves N files |
| `include_file_paths="col"` | the `hf://buckets/...` URI of the file | the presigned URL |
| `select(pl.len())` | reads one column | reads the footers only |
| `tail(n)`, `slice(offset, n)` | scan all files | read only the files they need |

With `resolve="collect"` a query runs like this:

1. **Schema.** When Polars first needs the schema (`collect()`, `collect_schema()`, `explain()`),
   the first file is resolved and its footer is read: one `resolve` request and one to three CDN
   requests (it depends on the Polars version), once per LazyFrame. The schema is the one a native scan of the first file has with
   your scan options.
2. **Groups.** The files are scanned in path order in groups of 64. The URLs of a group are
   resolved when the group before it is exhausted, so later groups get new URLs and a scan can run
   for longer than one hour. One group is one native multi-file scan that gets the projection, the
   predicate and the row limit of the query, and its DataFrames are passed on unchanged: the rows
   and their order are those of one native scan. A group must be scanned within the hour that its
   URLs are valid.
3. **Row limits.** A query with a row limit (`head(n)`) uses groups of 1, 4, 16, then 64 files and
   stops when it has its rows.

All files are one group (all URLs are resolved at the start of the query) with `row_index_name=` or
`n_rows=`, and for a row limit *before* a predicate (`lf.head(n).filter(...)`).

Every group is scanned with the schema of the first file of the scan (passed to the native scan as
`schema=`), unless you give `schema=` yourself. `missing_columns=`, `extra_columns=` and
`cast_options=` then apply to every file as in one native scan; the tests compare both modes on
files with different columns and types.

`LazyFrame.serialize()` of the default mode needs the `cloudpickle` package (Polars pickles an IO
source with it). The serialized plan holds the endpoint, the bucket paths and the scan options. It
does not hold the token or a presigned URL: a deserialized query uses the token of its own
environment, also if `token=` was given to `scan_bucket`.

Use `resolve="now"` when a query needs the native node: a metadata-only row count, `tail()`, or
tooling that inspects the scan node. Collect within the hour, and treat the plan as a secret.

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

With `resolve="now"` the call then sends one `resolve` request for each of the N files. With the
default `resolve="collect"` these requests are sent when a query runs:

| Query on N files | `resolve` requests |
| --- | --- |
| first use of the schema (`collect()`, `collect_schema()`, `explain()`) | 1, for the first file; 0 for a single-file URI |
| a full scan, a projection, a filter, a row count | N; the URL of the first file is used again if it is less than 5 minutes old, so N - 1 right after the schema |
| `head(n)` | the files of the groups it reads: 1, then 4, 16, 64 |
| a second `collect()` of the same LazyFrame | N again (N - 1 within 5 minutes) |

`resolve` requests count in the Hub's "resolvers" rate limit. A scan of N files uses N of them each
time it runs. A query sends at most 64 at once, and the next 64 only when the group before is
read (except in the one-group cases of [When the URLs are resolved](#when-the-urls-are-resolved)).

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
single-file URI: an invalid URI or glob, a missing bucket or path, nothing matched, an empty file
found in the listing, no access. With `resolve="now"` it raises all errors of the table.

With the default `resolve="collect"`, the `resolve` requests of the listed files are sent by the
query, so `collect()` raises their errors: a file that was deleted after the listing
(`FileNotFoundError`), a `401` / `403` (`PermissionError`), a rate limit or server error that the
retries did not clear (`HfHubHTTPError`), a file that the Hub does not redirect (`RuntimeError`).
The messages are the same. How the exception arrives depends on Polars:

- Polars 2 (tested with 2.0.0rc2) raises the exception itself.
- Polars 1.x wraps every exception of an IO source: `collect()` raises
  `polars.exceptions.ComputeError`, and its message holds the type name and the message of the
  original exception (`... FileNotFoundError: no such file: 'hf://buckets/...'`).
- All Polars versions wrap an error of the schema read (the `resolve` request of the first file)
  in a `ComputeError` that starts with `schema callable failed`.

A read error of Polars itself (a presigned URL that the CDN refuses, a file that is not parquet)
keeps its type. In the default mode its message names the `hf://` URI of the file; the presigned
URL is removed.

## Performance

Bucket reads fetch range requests from the XET CDN.

- **The IO-plugin node.** In the default mode the native scan runs inside an IO source, and whole
  DataFrames cross into Python. On HF Jobs (`cpu-performance`, Polars 2.0.0rc2, 54 files, 126.7 GB,
  four columns, CDN-warm, one run each) a prototype of this wrapper took 244 s and the native node
  301 s, with the same result: no overhead was measured for a full scan. A row count on a 3-file
  fixture took 2.5 s through the wrapper and 1.8 s on the native node, because Polars asks an IO
  source for one column to count rows. `tail()` and slices with an offset scan all files in the
  default mode. `resolve="now"` gives the native node for such queries.
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
  29.1 and 33.6 s without) and about 8% on Polars 1.44.2. On Polars 1.x you can set
  it yourself, before the first cloud read of the process.

[`benchmarks/`](benchmarks/README.md) has the scripts, what each number means, and the measured
tables with their job ids.

## Scan options

Extra keyword arguments to `scan_bucket` are forwarded to `pl.scan_parquet` (to the native scan of
every group in the default mode) — e.g. heterogeneous schemas across globbed files:

```python
plhf.scan_bucket(uri, missing_columns="insert", extra_columns="ignore")
```

or `storage_options={"max_retries": 5}` for flaky connections (this replaces the deprecated
`retries=` option of Polars; it applies to the data requests Polars makes, not to the Hub requests
above).

`include_file_paths="file"` adds the `hf://buckets/...` URI of the file of every row. (With
`resolve="now"` the column holds the presigned URL, signature included.) `hive_partitioning=` sees
the presigned CDN URLs, not the bucket paths, so it finds no partition columns in either mode.

## Limitations

- Reads cover parquet; writes cover parquet/csv/ipc/ndjson. Delta/Iceberg are out of scope.
- Hive-style partition columns are not inferred from paths on read (the presigned CDN URLs don't
  preserve the bucket paths) — but partitioned writes include the key columns in the files by
  default, so round-trips keep the data.
- The default read mode is a Polars IO-plugin node. Polars pushes a projection, a predicate and
  `head(n)` into it, nothing else: a row count reads one column, `tail()` and slices with an
  offset scan all files, and on Polars 1.x an error of a query is a `ComputeError` that quotes
  the original exception. `resolve="now"` gives the native node, with the presigned URLs in the
  plan. See [When the URLs are resolved](#when-the-urls-are-resolved) and [Errors](#errors).

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
(`expire_signed_urls()`), like the real CDN after an hour. File uploads do not go over HTTP; the fake has one
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
