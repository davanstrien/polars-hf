"""Scan parquet from Hugging Face buckets as a polars ``LazyFrame``.

Implemented as a pure-Python polars IO plugin via ``register_io_source``. The
heavy work stays in Rust: parquet decode runs in polars, and bytes are fetched
lazily through ``HfFileSystem`` seekable range requests, so projection and
row-limit pushdown only transfer the column chunks actually needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from huggingface_hub import HfFileSystem
from polars.io.plugins import register_io_source

from polars_hf._uri import parse_bucket_uri

if TYPE_CHECKING:
    from collections.abc import Iterator


def _list_files(fs: HfFileSystem, uri: str) -> list[str]:
    """Resolve a bucket URI to a sorted list of parquet file paths."""
    bp = parse_bucket_uri(uri)

    if bp.is_glob:
        files = fs.glob(bp.fs_path)
    elif bp.path.endswith(".parquet"):
        # A concrete single file.
        return [bp.fs_path]
    else:
        # A directory (or the whole bucket): expand to all parquet files.
        pattern = f"{bp.fs_path.rstrip('/')}/**/*.parquet"
        files = fs.glob(pattern)

    if not files:
        raise FileNotFoundError(f"no parquet files matched: {uri!r}")
    return sorted(files)


def scan_bucket(uri: str, *, token: str | None = None) -> pl.LazyFrame:
    """Lazily scan parquet file(s) from a Hugging Face bucket.

    Parameters
    ----------
    uri
        An ``hf://buckets/{namespace}/{name}/{path}`` URI. ``path`` may be a
        single ``.parquet`` file, a glob (e.g. ``data/*.parquet``), or a
        directory / the whole bucket (expanded to ``**/*.parquet``).
    token
        Hugging Face token. If ``None``, the token is resolved by
        ``huggingface_hub`` (the ``HF_TOKEN`` env var or the cached login).

    Returns
    -------
    LazyFrame
        Supports projection, predicate, and row-limit pushdown, and works with
        the streaming engine.

    Examples
    --------
    >>> import polars_hf as plhf
    >>> lf = plhf.scan_bucket("hf://buckets/me/data/*.parquet")  # doctest: +SKIP
    >>> lf.filter(pl.col("label") == 1).head(5).collect()  # doctest: +SKIP
    """
    fs = HfFileSystem(token=token)
    files = _list_files(fs, uri)

    with fs.open(files[0], "rb") as f:
        schema = pl.read_parquet_schema(f)

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,  # noqa: ARG001 (hint only)
    ) -> Iterator[pl.DataFrame]:
        # `remaining` tracks the row-limit budget across files. polars only
        # pushes `n_rows` when it is safe (i.e. no filter sits below the limit),
        # so when `predicate` is set `n_rows` is typically None and the engine
        # applies the final limit itself.
        remaining = n_rows
        for path in files:
            if remaining is not None and remaining <= 0:
                return
            with fs.open(path, "rb") as fh:
                df = pl.read_parquet(fh, columns=with_columns, n_rows=remaining)
            rows_read = df.height  # source rows read, before filtering
            if predicate is not None:
                df = df.filter(predicate)
            yield df
            if remaining is not None:
                remaining -= rows_read

    return register_io_source(source, schema=schema, is_pure=True)
