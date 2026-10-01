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
follows the authenticated Hub `resolve` redirect to a presigned CDN URL
(`us.aws.cdn.hf.co/xet-bridge-*`) and hands that to Polars. Polars' own Rust object store then does
async, concurrent, **range-read** scans — so **projection, predicate, and slice pushdown**,
streaming, and multi-file concurrency all work natively and only the column chunks actually needed
are transferred. (This is the same read
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

Requires `polars>=1.40,<3` and `huggingface_hub>=1.12,<3`.

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

### Writing

```python
# Single file (parquet/csv/ipc/ndjson; format inferred from the extension):
plhf.sink_bucket(lf, "hf://buckets/ns/name/out.parquet")

# Partitioned: pass a base prefix + partition options (native pl.PartitionBy):
plhf.sink_bucket(lf, "hf://buckets/ns/name/by_year", partition_by="year")
plhf.sink_bucket(lf, "hf://buckets/ns/name/shards", max_rows_per_file=1_000_000)
```

`sink_bucket` accepts a `LazyFrame` (streaming) or a `DataFrame`. Partitioned writes split by key
(hive `key=value/` layout), by size, or both. Two modes:

- `atomic=True` (default) — stage partitions locally, upload in one commit; bounded by local disk.
- `atomic=False` — stream each partition straight to the bucket; handles bigger-than-disk, one commit
  per file (cheap on buckets, which are not git-backed).

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

Signed URLs are resolved when `scan_bucket` is called and are valid for ~1 hour (the URL carries
its own expiry time). Collect within that window; for long-lived query plans, call `scan_bucket`
again to refresh.

### Hub requests

`scan_bucket` reads no file data. It makes these requests to the Hub:

| `{path}` | Requests |
| --- | --- |
| one file, any extension | 1 `resolve` (HEAD) |
| a directory of N parquet files (`data`) | 1 `resolve` (answered "not found") + 1 listing per page of results + N `resolve` |
| the same with a trailing slash (`data/`), or the whole bucket | 1 listing per page + N `resolve` |
| a glob that selects N files | 1 listing per page + N `resolve` |
| a path that does not exist | 1 `resolve` + 1 listing |

A glob whose only glob segment is the last one (`data/*.parquet`) lists that directory only. Other
globs list the subtree below the text before their first glob character. An invalid glob raises
before any request.

`resolve` requests count in the Hub's "resolvers" rate limit, so a scan of N files uses N of them.

**Retries.** A `408`, `429` or `5xx` answer to a `resolve` request or to a listing page is retried
up to 5 times; only the failed request is sent again. `scan_bucket` sends the listing requests
itself (it does not call `HfApi.list_bucket_tree`), so the same limits apply to every page and on
every supported `huggingface_hub` version. The wait before a retry is the
one the Hub asks for (rate-limit reset, `Retry-After`), else 1 s doubling up to 8 s. When the Hub
asks for more than 5 s, one warning per `scan_bucket` call announces the wait, so a paused scan is
not silent. If your warning filter turns warnings into errors, the message is logged (logger
`polars_hf.read`) instead, and the scan continues.

One `scan_bucket` call waits only while the wait ends within 10 minutes of its start. A rate-limit
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

or `storage_options={"max_retries": 5}` for flaky connections (this replaces the deprecated
`retries=` option of Polars; it applies to the data requests Polars makes, not to the Hub requests
above). Options that derive meaning from the file *path*
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
be scripted to fail (`429`, `503`, `403`, ...). Uploads are stored in memory through a patched
`HfApi._batch_bucket_files`, so the client-side chunking of the public method stays real. Use the
`fake_hub` / `fake_bucket` fixtures.

**Staging tests** (`-m staging`) do real round-trips against `https://hub-ci.huggingface.co`, the
instance `huggingface_hub` uses for its own tests. `tests/conftest.py` sets `HF_ENDPOINT` and
`HF_TOKEN` to the staging endpoint and its public CI token *before* `huggingface_hub` is imported,
so the test process never talks to `huggingface.co` and never reads your own token. Each test
creates a uniquely named bucket and deletes it afterwards. Staging can answer
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
