# Reading from a bucket: reference

Details behind [`scan_bucket`](../README.md#read) and `count_rows`. The README has the short version.

## When the URLs are resolved

`scan_bucket(uri, resolve=...)` has two modes:

| | `resolve="collect"` (default) | `resolve="now"` |
| --- | --- | --- |
| LazyFrame | a Polars IO-plugin node that holds the bucket paths | the native `scan_parquet` node over the presigned URLs |
| Needs | nothing | the environment variable `POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1` |
| `scan_bucket()` does | the listing (see [Hub requests](#hub-requests)) | the listing and one `resolve` request per file |
| Presigned URLs in `explain()`, `serialize()`, error messages | no | yes, with their signature |
| Validity | a URL is resolved shortly before it is used; the LazyFrame does not expire | ~1 hour from the `scan_bucket` call; call it again for a new plan |
| `head(5)` on N files | resolves 1 file | resolves N files |
| `include_file_paths="col"` | the `hf://buckets/...` URI of the file | the presigned URL |
| `select(pl.len())` | reads one whole column of every file | reads the footers only |
| `tail(n)`, `slice(offset, n)` | scan all files | read only the files they need |

Without `resolve=`, the mode is the value of the environment variable `POLARS_HF_RESOLVE` if it is
set (`collect` or `now`), else `"collect"`.

**`resolve="now"` is an opt-in.** It is the faster path for metadata-heavy work (row counts,
`tail()`, slices with an offset). It puts a read-only URL of every file, valid for about 60
minutes, into `explain()`, `serialize()`, the messages of read errors, logs and an
`include_file_paths=` column: anyone who sees one can read that file until it expires. The mode is
therefore refused with a `ValueError`, before any request, unless the environment variable
`POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1` is set. Only the exact value `1` enables it (`true`, `yes`
or an empty value do not). Nothing falls back to it. Collect within the
hour, and treat the plan and its logs as secrets.

Polars prints the URLs it scans to stderr when `POLARS_VERBOSE=1` is set, in both modes. The
package cannot prevent that: treat verbose logs as secrets for an hour.

The same holds for DEBUG logging of the HTTP client that `huggingface_hub` uses (logger `httpcore`
or `httpcore2`): it logs response headers, and the Hub's redirect answer carries the presigned URL
in its `Location` header. With DEBUG logging enabled for that logger, presigned URLs are printed in
every mode.

The Hub token itself is held in memory by the package while it talks to the Hub, as in
`huggingface_hub`, so a tool that captures the local variables of traceback frames can record it
on a Hub error. What this section promises is about presigned URLs: none in plans, messages,
exception chains and traceback frames of the default mode.

**What the default mode costs.** Full scans, `head()` and selective queries take about as long as
on the native node. A row count through the LazyFrame and `tail()` do not.

Measured on HF Jobs (`cpu-performance`, Polars 2.0.0rc2, 12 files, 28 GB, job
`6abec96b404719ba3761a56b`):

| Query | `resolve="collect"` | `resolve="now"` (native node) |
| --- | --- | --- |
| row count, `select(pl.len())` | 30–33 s, 27.7 GB downloaded | 1.1 s, 0.07 GB downloaded |
| `tail(5)` | 19 GB peak memory | 2.6 GB peak memory |
| one small column | 3.3–4.8 s | 2.1–2.9 s |
| `head(5)` | 0.8 s | 0.7 s |
| full scan of four columns | 22 s | 24 s |

**Row counts.** In the default mode `lf.select(pl.len())` is not a metadata query: Polars asks an
IO source for one column to count its rows. The package does not choose that column, and it can be
the largest one (in the measurement above it was nearly all of the data). Use
`plhf.count_rows(uri)` for a row count: it reads the parquet footers only and returns an `int`.

```python
plhf.count_rows("hf://buckets/my-namespace/my-bucket/data/*.parquet")
```

`count_rows` has the path rules and the errors of `scan_bucket`. It sends one `resolve` request per
file and keeps nothing. It does not compare the schemas of the files.

With `resolve="collect"` a query runs like this:

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
   is resolved and scanned once more, provided that it has not returned rows yet. A kept URL
   names the content that the file had when it was resolved: if a file is replaced in the bucket,
   a later query on the same LazyFrame reads the old content until the URL is 5 minutes old; after
   that, and from a new `scan_bucket` call, it reads the new content. Queries of several threads on
   one LazyFrame resolve a file once.

A consumer that stops reading a `collect_batches()` iterator does not stop the query: Polars keeps
running it, so the following groups are still resolved and scanned.

All files are one group (all URLs are resolved at the start of the query) with `row_index_name=` or
`n_rows=`, and for a row limit *before* a predicate (`lf.head(n).filter(...)`).

Every group is scanned with the schema of the first file of the scan (passed to the native scan as
`schema=`), unless you give `schema=` yourself. `missing_columns=`, `extra_columns=` and
`cast_options=` then apply to every file as in one native scan; the tests compare both modes on
files with different columns and types. One difference remains for files with different columns or
types and no such option: a row count, or a slice that lies outside of the rows, is answered by the
native node from the footers and raises a schema error in the default mode, because the default
mode reads a column of every file.

`LazyFrame.serialize()` of the default mode needs the `cloudpickle` package (Polars pickles an IO
source with it). The serialized plan holds the endpoint, the bucket paths and the scan options
except `storage_options=` and `credential_provider=`, which can hold credentials. It does not hold
the token or a presigned URL. A deserialized query therefore uses the token of its own
environment, also if `token=` was given to `scan_bucket`, and runs without `storage_options` and
`credential_provider`.

Use `resolve="now"` (with its acknowledgement variable) when a query needs the native node:
`tail()`, a slice with an offset, or tooling that inspects the scan node.

## Paths

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

## Hub requests

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
| a full scan, a projection, a filter | N - 1: the URL of the first file is known from the schema |
| `head(n)` | the files of the groups it reads: 1, then 4, 16, 64 |
| a second query on the same LazyFrame | 0 for the files whose URL is less than 5 minutes old; 1 for each other file |
| `count_rows(uri)` | N, in the call |

`resolve` requests count in the Hub's "resolvers" rate limit. A scan of N files uses N of them,
and N again when it runs more than 5 minutes later. A query sends at most 64 at once, and the next
64 only when the group before is read (except in the one-group cases of
[When the URLs are resolved](#when-the-urls-are-resolved)).

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

## Errors

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
found in the listing, no access, an unknown `resolve` mode, and `resolve="now"` without its
acknowledgement variable. With `resolve="now"` it raises all errors of the table.

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
URL is removed from the message, and the exception has no `__cause__` or `__context__` that holds
it. (Polars' own verbose log, `POLARS_VERBOSE=1`, still prints the URLs.)

## Performance

Bucket reads fetch range requests from the XET CDN.

- **The IO-plugin node.** In the default mode the native scan runs inside an IO source, and whole
  DataFrames cross into Python. On HF Jobs (`cpu-performance`, Polars 2.0.0rc2, 54 files, 126.7 GB,
  four columns, CDN-warm, one run each) a prototype of this wrapper took 244 s and the native node
  301 s, with the same result: no overhead was measured for a full scan. Queries that Polars does
  not push into an IO source cost more: a row count reads a whole column (use `plhf.count_rows`),
  and `tail()` and slices with an offset scan all files (`resolve="now"` reads only what they
  need). The table in
  [When the URLs are resolved](#when-the-urls-are-resolved) has the measured numbers.
- **Download, then scan.** In the same run, downloading the files with `hf_xet` in a rolling window
  and scanning them from local disk took 114 s (7.3 GB peak RSS, 15.1 GB of staging disk).
  `polars-hf` does not do this; [`benchmarks/read_paths.py`](../benchmarks/read_paths.py) has that arm
  for comparison.
- **Cold vs warm CDN.** The *first* read of freshly written/copied data pays a cold-CDN penalty
  (the bytes aren't at the edge yet); subsequent reads are much faster. For repeated large-scale
  reads, consider [pre-warming](https://huggingface.co/docs/hub/storage-buckets#pre-warming-and-cdn)
  the bucket.
- **`POLARS_CONCURRENCY_BUDGET`.** `polars-hf` does not set Polars environment variables. Earlier
  versions set `POLARS_CONCURRENCY_BUDGET=64` at import, for every cloud scan of the process. On
  HF Jobs (14 GB) that setting made no difference on Polars 2.0.0rc2 (31.0 and 32.1 s with it,
  29.1 and 33.6 s without) and about 8% on Polars 1.44.2
  ([job ids](../benchmarks/README.md#polars_concurrency_budget)). On Polars 1.x you can set
  it yourself, before the first cloud read of the process.

[`benchmarks/`](../benchmarks/README.md) has the scripts, what each number means, and the measured
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
`resolve="now"` the column holds the presigned URL, signature included.) The URI is found from
the presigned URL of the file: if two files with the same content get the same URL and are in the
same group of 64 files, their rows show the URI of one of them. In different groups each shows its
own URI. `hive_partitioning=` sees
the presigned CDN URLs, not the bucket paths, so it finds no partition columns in either mode.
