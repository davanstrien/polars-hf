# polars-hf

Read and write [Hugging Face Storage Buckets](https://huggingface.co/docs/hub/storage-buckets) with
[Polars](https://pola.rs). Pure Python, works with stock Polars: no fork, no compiled extension.

```python
import polars as pl
import polars_hf as plhf

lf = plhf.scan_bucket("hf://buckets/my-org/my-bucket/data/")

plhf.sink_bucket(
    lf.filter(pl.col("score") >= 3).select("id", "text", "score"),
    "hf://buckets/my-org/my-bucket/filtered/",
    max_bytes_per_file=1_000_000_000,
)
```

> **Status: beta.** The API can still change, and the package is not on PyPI yet (install from git,
> below). The streaming writer relies on a few private interfaces of `huggingface_hub` and `hf_xet`;
> if they change, a write fails with an error that says so.

## Why

Polars reads `hf://datasets/...` natively, but not `hf://buckets/...`. A bucket is a natural place
for the input and output of a processing job, so this package fills the gap from the outside:

- **Reads use Polars' parquet reader.** `scan_bucket` finds the files and gets a presigned URL for
  each one when the query runs; Polars does the range reads, so only the bytes a query needs are
  fetched.
- **Writes stream.** `sink_bucket` runs Polars' own `sink_parquet` (and csv / ipc / ndjson) and
  streams the output into the bucket as it is produced. No local copy of the output, and nothing
  appears in the bucket unless the query finished.

Native bucket support is proposed upstream
([pola-rs/polars#27611](https://github.com/pola-rs/polars/issues/27611),
[#26909](https://github.com/pola-rs/polars/issues/26909)). If that lands, most of this package
becomes unnecessary.

## Install

```bash
uv add "polars-hf @ git+https://github.com/davanstrien/polars-hf"
# or: pip install "git+https://github.com/davanstrien/polars-hf"
```

Requires `polars>=1.40,<3` and `huggingface_hub>=1.12,<3`. Writing without local disk needs
`huggingface_hub>=1.19`. CI runs Polars 1.40, 1.44 and the 2.0 release candidate; 2.0 is the main
target.

Authentication is whatever `huggingface_hub` finds (`HF_TOKEN`, or `hf auth login`); the functions
also take `token=`.

## Which path do I use?

**Reading**

| You want | Use | What it costs |
| --- | --- | --- |
| To scan, filter, select, `head` (the default) | `plhf.scan_bucket(uri)` | Nothing sensitive in the query plan. A row count *through the frame* reads a whole column. |
| A row count | `plhf.count_rows(uri)` | Reads parquet footers only. |
| Fast `tail()`, slices with an offset, or the native scan node | `scan_bucket(uri, resolve="now")` with `POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1` | Presigned URLs sit in the plan, error messages and logs for about an hour. |

**Writing**

| You want | Use |
| --- | --- |
| To fail if the destination exists (the default) | `mode="error"` |
| To add files to a directory | `mode="append"` |
| To replace a file, or the contents of a directory | `mode="overwrite"` |
| No local disk (the default) | `backend="stream"` |
| Only `huggingface_hub`'s public upload API, staged on local disk | `backend="staged"` |

## Read

```python
# one file, a directory (every .parquet / .pq file below it), or a glob
lf = plhf.scan_bucket("hf://buckets/my-org/my-bucket/data/2024/*.parquet")

df = (
    lf.filter(pl.col("label") == 1)   # predicate pushdown
      .select("text", "label")        # projection pushdown
      .head(100)                      # row-limit pushdown
      .collect()
)

n = plhf.count_rows("hf://buckets/my-org/my-bucket/data/2024/*.parquet")
```

`scan_bucket` returns a `LazyFrame`, so the streaming engine and the rest of Polars work on it.
Extra keyword arguments go to `pl.scan_parquet` (for example `missing_columns="insert"` for files
with different schemas).

Things worth knowing:

- **Nothing is downloaded when you call `scan_bucket`.** It lists the files. URLs are fetched when
  the query runs, a group of files at a time, so `head(5)` on a large bucket touches one file and a
  job that runs for hours keeps working.
- **No credential in the plan.** A presigned URL lets anyone who has it read that file for about an
  hour. In the default mode it does not appear in `explain()`, `serialize()`, error messages or
  tracebacks. Two things are outside the package's control: `POLARS_VERBOSE=1` makes Polars print
  the URLs it reads, and DEBUG logging of the HTTP client prints them too.
- **Use `count_rows` for counts.** In the default mode Polars asks for a whole column to count
  rows, which can mean downloading most of the data. `count_rows` reads only the footers.
- **Errors show up when the query runs.** A file deleted after the listing, or a token that lost
  access, raises from `collect()`. On Polars 1.x these arrive as a `ComputeError` that quotes the
  original error; Polars 2.0 raises the original.
- **Hub requests are retried with limits.** One request per file gets its URL; those count against
  the Hub's rate limit, and a scan waits for the limit to reset for up to 10 minutes per group
  before it raises.

`resolve="now"` is the old behaviour: the native scan node over presigned URLs. It is faster for
`tail()` and offset slices, and it is refused unless you set `POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1`,
because it puts those URLs where plans and logs end up.

Path rules, request counts, retries, the error table and the details of both modes are in
[docs/reading.md](docs/reading.md).

## Write

```python
# one file; the format comes from the extension (parquet, csv, ipc, ndjson)
plhf.sink_bucket(lf, "hf://buckets/my-org/my-bucket/out.parquet")

# many files: by key (hive layout), by size, or both
plhf.sink_bucket(lf, "hf://buckets/my-org/my-bucket/by_year/", partition_by="year")
plhf.sink_bucket(lf, "hf://buckets/my-org/my-bucket/shards/", max_bytes_per_file=1_000_000_000)

# a destination that exists is an error unless you say what should happen
plhf.sink_bucket(more, "hf://buckets/my-org/my-bucket/by_year/", partition_by="year", mode="append")
```

| `mode` | One file | Directory |
| --- | --- | --- |
| `"error"` (default) | raises if the file exists | raises if anything is there |
| `"append"` | not possible | adds new files; never replaces or deletes one |
| `"overwrite"` | replaces the file | writes the new files, then deletes the ones that were there before |

What you can rely on, and what you cannot:

- **A failed write changes nothing.** If the query fails, is interrupted, or the process is killed,
  an existing file keeps its content and no partial file appears.
- **Files are named the way Polars names them locally** (`year=2024/00000000.parquet`), so a
  directory written here reads back like any hive-partitioned dataset. `append` adds a random
  suffix per call so it cannot collide with what is there.
- **A write is not a transaction.** Buckets have none: files are registered in batches of 1,000,
  and `overwrite` deletes the old files after the new ones are in. A failure in between raises and
  can leave a mix. `overwrite` on the bucket root is refused.
- **`append` twice with the same data adds the rows twice.**

| `backend` | How the bytes get there | Local disk |
| --- | --- | --- |
| `"stream"` (default) | streamed into the bucket while Polars writes | none |
| `"staged"` | written to a temporary directory, then uploaded with `huggingface_hub`'s public API | the size of the output |

Both produce the same files. The reference, including exactly what each kind of failure leaves
behind, is in [docs/writing.md](docs/writing.md).

## On Hugging Face Jobs

A [PEP 723](https://peps.python.org/pep-0723/) script is enough; the Job installs the package from
git:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["polars-hf @ git+https://github.com/davanstrien/polars-hf@main"]
# ///
import polars as pl
import polars_hf as plhf

plhf.sink_bucket(
    plhf.scan_bucket("hf://buckets/my-org/raw/").filter(pl.col("score") >= 3),
    "hf://buckets/my-org/filtered/",
    max_bytes_per_file=1_000_000_000,
)
```

```bash
hf jobs uv run --secrets HF_TOKEN --flavor cpu-upgrade my_script.py
```

## How fast is it

I've only run a handful of benchmarks so far, so treat these as a rough guide rather than a
promise. On a 32-vCPU Hugging Face Job (`cpu-performance`), with Polars 2.0.0rc2 and parquet shards
of about 2 GB:

- Scanning ran at somewhere between a few hundred MB/s and a bit over 1 GB/s. It varied a lot from
  day to day, and the first read of new data is slower than later ones.
- A scan, filter and write of roughly 100 GB took a few minutes, with no local disk used.
- Memory during large writes stayed in the low tens of GB in these runs. That is what I observed,
  not a guaranteed bound.

Reading looks limited by the network path rather than by Polars: a scan used only a fraction of the
cores.

The scripts, the exact numbers and their job ids are in [benchmarks/](benchmarks/README.md), so you
can rerun them on your own data.

## What to know before relying on it

- **Parquet only for reads.** Writes cover parquet, csv, ipc and ndjson. Delta and Iceberg are out
  of scope.
- **Hive columns are not inferred on read.** Polars sees presigned URLs, not bucket paths.
  Partitioned writes keep the key columns in the files, so a round trip keeps the data.
- **The default read mode is a Polars IO plugin.** Polars pushes a projection, a filter and
  `head(n)` into it and nothing else: a count through the frame reads a column, and `tail()` and
  offset slices scan every file.
- **A replaced file can be read stale for up to 5 minutes** by a later query on the same
  `LazyFrame`, because URLs are reused for that long. A new `scan_bucket` call reads the new file.
- **Private interfaces.** The `"stream"` backend uses helpers from `huggingface_hub.utils._xet` and
  the bucket batch endpoint directly, because the public upload API takes a file or bytes, not a
  stream. A change there shows up as an error that names the installed versions;
  `backend="staged"` uploads through the public API.
- **No transactions**, as described under [Write](#write).

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest            # offline suite: a local fake Hub, no network, no token
```

The offline suite covers the read and write paths against a fake Hub, including injected failures,
and property tests with Hypothesis. Live tests, the slower tests and the CI setup are described in
[docs/development.md](docs/development.md).

## License

MIT
