# polars-hf

Read [Hugging Face Hub buckets](https://huggingface.co/docs/hub/storage-backends) with
[Polars](https://pola.rs), as a pure-Python **IO plugin**. No fork of Polars, no compiled
extensions — just `pip install` and scan.

> **Status:** alpha. Reads are implemented; writes (`sink_bucket`) are next.

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

## Install

```bash
uv add polars-hf        # or: pip install polars-hf
```

Requires `polars>=1.40,<1.50` and `huggingface_hub>=1.12`.

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
