# polars-hf

Read and write [Hugging Face Hub buckets](https://huggingface.co/docs/hub/storage-buckets) with
[Polars](https://pola.rs), as a pure-Python **IO plugin**. No fork of Polars, no compiled
extensions — just install and scan.

> **Status:** alpha, pre-release (not on PyPI yet — install from git, see below).
> Reads (`scan_bucket`) and writes (`sink_bucket`, including partitioned) are implemented.

## Why

Stock Polars already reads `hf://datasets/...` and `hf://spaces/...` natively. It does **not** yet
read `hf://buckets/...`. `polars-hf` fills that gap from the outside.

It returns a **native** `pl.scan_parquet` LazyFrame: bucket files are XET-backed, so `scan_bucket`
follows the authenticated Hub `resolve` redirect to a presigned `cas-bridge.xethub.hf.co` URL and
hands that to Polars. Polars' own Rust object store then does async, concurrent, **range-read**
scans — so **projection, predicate, and slice pushdown**, streaming, and multi-file concurrency all
work natively and only the column chunks actually needed are transferred. (This is the same read
mechanism upstream's `hf://` reader uses; we just resolve the signed URL in Python because stock
Polars can't attach a bearer token to a generic `https://` URL.)

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

Requires `polars>=1.40,<3`, `huggingface_hub>=1.12,<3` and `httpx>=0.27,<1`. Writes that use no
local disk need `huggingface_hub>=1.19` (see [Writing](#writing)).

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

# A single file, a glob, or a whole bucket/directory (expanded to **/*.parquet):
lf = plhf.scan_bucket("hf://buckets/my-namespace/my-bucket/data/*.parquet")

df = (
    lf.filter(pl.col("label") == 1)   # predicate pushdown
      .select("text", "label")        # projection pushdown
      .head(100)                       # row-limit pushdown
      .collect()
)
```

`scan_bucket` returns a lazy `LazyFrame` and works with the streaming engine.

### Writing

```python
# Single file (parquet/csv/ipc/ndjson; format inferred from the extension):
plhf.sink_bucket(lf, "hf://buckets/ns/name/out.parquet")

# Partitioned: pass a base prefix + partition options (native pl.PartitionBy):
plhf.sink_bucket(lf, "hf://buckets/ns/name/by_year", partition_by="year")
plhf.sink_bucket(lf, "hf://buckets/ns/name/shards", max_rows_per_file=1_000_000)

# Replace the content of a prefix: files this call does not write are deleted.
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
bytes), and a null key is `__HIVE_DEFAULT_PARTITION__`. A partitioned write whose query returns
no rows writes one file with the schema and no rows, `{prefix}/00000000.{extension}`, so the prefix
can be scanned afterwards.

**`mode`** sets what happens to objects that are already at the destination:

| `mode` | Single file | Partitioned (base prefix) |
| --- | --- | --- |
| `"append"` (default) | the object is replaced | existing files stay; a file with the same name as a new file is replaced |
| `"overwrite"` | same as `"append"` | as `"append"`, then the files that were below the prefix before the write, and that this call did not write, are deleted |
| `"error"` | `FileExistsError` if the object exists | `FileExistsError` if a file exists at the prefix itself or anywhere below it |

`"overwrite"` needs a directory below the bucket root: `hf://buckets/ns/name` and
`hf://buckets/ns/name/` are refused with a `ValueError`. It lists the prefix before the write and
deletes only files from that listing, so a file that another writer adds during the write is kept.
A query that returns no rows still deletes the previous files and leaves one file with the schema
and no rows.
A file that another writer *replaces* during the write is still deleted. A file at the prefix itself
(`out` as a file when the write goes to `out/...`) is neither listed nor deleted. The check of
`"error"` is also a listing before the write: it does not exclude a concurrent writer.

**Paths.** The Hub refuses a path with a backslash, an empty segment (`a//b`) or a `.` / `..`
segment. `sink_bucket` also refuses a control character in a path and a path segment of more than
255 bytes, with both backends, because the `"hub"` backend cannot create such a name on a local
file system. It raises a `ValueError` for such a destination before it sends a request. A partition
column name or value can produce such a path too (a backslash is not percent-encoded, and an
encoded value counts with its `key=` prefix towards the 255 bytes); the error names the `key=value`
segment, and no file of that write is registered.

**`backend`** sets how the output reaches the bucket:

| `backend` | How | Local disk | Needs |
| --- | --- | --- | --- |
| `"xet"` | each output file is streamed into Xet storage with `hf_xet` while Polars writes it, then all files are registered in the bucket | none for the output | `huggingface_hub>=1.19` and the `hf_xet` that it requires (installed with it on x86_64 and arm64) |
| `"hub"` | Polars writes the output to a temporary directory; the public `HfApi.batch_bucket_files` uploads and registers it | the size of the complete output | any supported `huggingface_hub` |

The default (`backend=None`) is the environment variable `POLARS_HF_SINK_BACKEND` if set, else
`"xet"` when the installed packages support it, else `"hub"`. An explicit `"xet"` that cannot run
raises an error; it does not fall back.

With both backends, the deletes of `mode="overwrite"` are sent to the bucket batch endpoint
(`/api/buckets/{id}/batch`) directly, not through `HfApi.batch_bucket_files`: `huggingface_hub` 1.x
does not report a rejected delete, and the direct request lets `sink_bucket` check the answer.

Notes on the `"xet"` backend:

- It uses private helpers of `huggingface_hub.utils._xet` and builds the
  `/api/buckets/{id}/batch` request itself, because the public API has no streaming upload. If a
  later `huggingface_hub` or `hf_xet` changes these, `sink_bucket` raises a `RuntimeError` that
  names the installed versions and points to `backend="hub"`; it does not fall back silently.
- A `KeyboardInterrupt` during a write aborts the process-wide Xet session, as `huggingface_hub`
  does in its own uploads. Other Xet uploads and downloads that run in the same process are
  cancelled.
- All upload streams stay open until the query ends, because Polars does not signal when one file is
  complete.

Notes on the `"hub"` backend:

- It stages in the directory named by `POLARS_HF_STAGING_DIR`, else in the system temporary
  directory, and removes the files when the write ends.
- With `huggingface_hub` 1.x, `HfApi.batch_bucket_files` does not report files that the bucket
  rejected. The backend then lists the destination once after the upload (one extra listing of the
  file's string prefix, or of the base prefix, per write) and raises if a file is missing or has
  another size. `huggingface_hub` 2.x reports rejected files itself, and no listing is made.

**Memory.** In laptop runs with outputs up to 4.5 GB, the `"xet"` backend used 0.5–1.8 GB more
memory than Polars alone. A bound on its memory use is not proven. See
[Measurements](#measurements).

**What a failure leaves behind.** With both backends, no file is registered in the bucket before the
Polars sink has finished without an error. If the query fails or is interrupted, the destination is
unchanged: an existing object keeps its content, and no partial or empty file appears. An upload
that fails before the first registration request also leaves the destination unchanged. A failure
of a registration request does not: see the list below.

A failed registration or delete request raises `polars_hf.BucketRegistrationError` (a
`RuntimeError`) with both backends, and so does a failed upload of the `"hub"` backend (an HTTP
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

| Output | Polars alone | `"hub"` | `"xet"` |
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

- `{path}` may be a single `.parquet` file, a glob (`data/*.parquet`), or a directory / the whole
  bucket (expanded to `**/*.parquet`).
- Buckets have **no** revision concept, so `@revision` is rejected (matching the Hub).
- `hf://datasets/...` and `hf://spaces/...` are read natively by Polars — use
  `pl.scan_parquet(...)` for those.

Signed URLs are resolved when `scan_bucket` is called and are valid for ~1 hour. Collect within that
window; for long-lived query plans, call `scan_bucket` again to refresh.

## Performance

Bucket reads fetch many small range requests from the XET CDN. Two things dominate:

- **Concurrency.** polars' default cloud-IO concurrency (`max(cpu_threads, 10)`) is low for
  high-latency object stores. `polars-hf` raises `POLARS_CONCURRENCY_BUDGET` to `64` by default
  (override by setting it yourself). This is a large win on warm/repeated scans and ~15% on cold.
- **Cold vs warm CDN.** The *first* read of freshly written/copied data pays a cold-CDN penalty
  (the bytes aren't at the edge yet); subsequent reads are much faster. For repeated large-scale
  reads, consider [pre-warming](https://huggingface.co/docs/hub/storage-buckets#pre-warming-and-cdn)
  the bucket.

## Scan options

Extra keyword arguments to `scan_bucket` are forwarded to `pl.scan_parquet`, so native options
work as-is — e.g. heterogeneous schemas across globbed files:

```python
plhf.scan_bucket(uri, missing_columns="insert", extra_columns="ignore")
```

or `retries=5` for flaky connections. Options that derive meaning from the file *path*
(`hive_partitioning=`, `include_file_paths=`) see the presigned CDN URLs, not the bucket paths,
so they are not useful here.

## Limitations

- Reads cover parquet; writes cover parquet/csv/ipc/ndjson. Delta/Iceberg are out of scope.
- Hive-style partition columns are not inferred from paths on read (the presigned CDN URLs don't
  preserve the bucket paths) — but partitioned writes include the key columns in the files by
  default, so round-trips keep the data.

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
be scripted to fail (`429`, `503`, `403`, ...). File uploads do not go over HTTP; the fake has one
seam per sink backend. For the `"hub"` backend, a patched `HfApi._batch_bucket_files` stores the
files in memory, so the client-side chunking of the public method stays real. For the `"xet"`
backend, an in-memory object replaces the `hf_xet` upload commit, and the backend's own
registration request goes to the fake's `POST /api/buckets/{id}/batch` route. The fake also reports
the `"xet"` backend as available, so the offline suite runs both backends with every supported
`huggingface_hub`. The real `hf_xet` upload is covered by the staging tests only. Use the
`fake_hub` / `fake_bucket` fixtures. `tests/sinks.py` is the one place that names the write design
(backends and failure switches); write tests go through its helpers.

**Staging tests** (`-m staging`) do real round-trips against `https://hub-ci.huggingface.co`, the
instance `huggingface_hub` uses for its own tests. `tests/conftest.py` sets `HF_ENDPOINT` and
`HF_TOKEN` to the staging endpoint and its public CI token *before* `huggingface_hub` is imported,
so the test process never talks to `huggingface.co` and never reads your own token. Each test
creates a uniquely named bucket and deletes it afterwards. `tests/test_staging_sinks.py` runs the
same write scenarios for both sink backends; the `xet` cases are skipped when the installed
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

**Known bugs** are in `tests/test_known_bugs.py` as `@pytest.mark.xfail(strict=True)` tests. Each one
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
