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

Requires `polars>=1.40,<1.50` and `huggingface_hub>=1.12`.

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

## Limitations

- **Heterogeneous schemas across globbed files are not yet supported** (the equivalent of native
  Polars `missing_columns="insert"`). Like Polars' own default, mismatched schemas raise. Planned.
- Reads cover parquet; writes cover parquet/csv/ipc/ndjson. Delta/Iceberg are out of scope.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest                 # network tests skip without an HF token
```

## License

MIT
