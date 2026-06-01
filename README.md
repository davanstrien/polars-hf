# polars-hf

Read [Hugging Face Hub buckets](https://huggingface.co/docs/hub/storage-backends) with
[Polars](https://pola.rs), as a pure-Python **IO plugin**. No fork of Polars, no compiled
extensions — just `pip install` and scan.

> **Status:** alpha. Reads are implemented; writes (`sink_bucket`) are next.

## Why

Stock Polars already reads `hf://datasets/...` and `hf://spaces/...` natively. It does **not** yet
read `hf://buckets/...`. `polars-hf` fills that gap from the outside, using Polars'
[`register_io_source`](https://docs.pola.rs/user-guide/plugins/io_plugins/) plugin API and
`huggingface_hub`'s bucket support.

The heavy lifting stays in Rust: parquet decoding runs in Polars, and bytes are fetched lazily via
`HfFileSystem` range requests, so **projection, predicate, and row-limit pushdown** only transfer the
column chunks actually needed.

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

## Limitations

- **Heterogeneous schemas across globbed files are not yet supported** (the equivalent of native
  Polars `missing_columns="insert"`). Like Polars' own default, mismatched schemas raise. Planned.
- Writes (`sink_bucket`) are not implemented yet.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest                 # network tests skip without an HF token
```

## License

MIT
