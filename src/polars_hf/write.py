"""Write a polars frame to a single file in a Hugging Face bucket.

Unlike reads, writes have no signed-URL shortcut: bucket files are XET-backed
(content-defined chunking + dedup + commit), so uploads must go through
``huggingface_hub``. We open the bucket path with ``HfFileSystem`` in ``"wb"``
mode and hand that file object to polars' streaming ``sink_*``. polars writes
incrementally (a ``SinkTarget::Dyn`` writer), and ``HfFileSystem`` spills to a
local temp file then commits via ``hf_xet`` on close — so memory stays bounded
(bigger-than-RAM is fine), while a single file is bounded by local disk.

Partitioned / sharded writes (``PartitionBy``, bigger-than-disk) are a separate
follow-up; polars requires a path (not a file object) when partitioning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from huggingface_hub import HfFileSystem

from polars_hf._uri import parse_bucket_uri

if TYPE_CHECKING:
    import polars as pl

# Map file extension -> polars streaming sink format.
_EXT_FORMAT = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".ipc": "ipc",
    ".arrow": "ipc",
    ".feather": "ipc",
    ".ndjson": "ndjson",
    ".jsonl": "ndjson",
}
_SINK_METHOD = {
    "parquet": "sink_parquet",
    "csv": "sink_csv",
    "ipc": "sink_ipc",
    "ndjson": "sink_ndjson",
}


def _infer_format(path: str) -> str:
    """Infer the sink format from a file path's extension."""
    lower = path.lower()
    for ext, fmt in _EXT_FORMAT.items():
        if lower.endswith(ext):
            return fmt
    raise ValueError(
        f"could not infer format from {path!r}; pass format= as one of "
        f"{sorted(set(_SINK_METHOD))}"
    )


def sink_bucket(
    frame: pl.LazyFrame | pl.DataFrame,
    uri: str,
    *,
    format: str | None = None,
    token: str | None = None,
    **kwargs: object,
) -> None:
    """Write a polars frame to a single file in a Hugging Face bucket.

    Parameters
    ----------
    frame
        A ``LazyFrame`` (written via the streaming engine) or an eager
        ``DataFrame`` (converted with ``.lazy()``).
    uri
        Destination ``hf://buckets/{namespace}/{name}/{path}`` URI. The format is
        inferred from the file extension unless ``format`` is given.
    format
        One of ``"parquet"``, ``"csv"``, ``"ipc"``, ``"ndjson"``. Overrides the
        extension-based inference.
    token
        Hugging Face token. If ``None``, resolved by ``huggingface_hub``.
    **kwargs
        Forwarded to the underlying polars ``sink_*`` (e.g. ``compression`` for
        parquet, ``separator`` for csv).

    Notes
    -----
    The bucket must already exist. Memory stays bounded for arbitrarily large
    inputs, but the output file is staged to local disk before upload, so a
    single file is bounded by available disk — shard large outputs across files.

    Examples
    --------
    >>> import polars_hf as plhf
    >>> plhf.sink_bucket(lf, "hf://buckets/me/data/out.parquet")  # doctest: +SKIP
    """
    bp = parse_bucket_uri(uri)
    if not bp.path:
        raise ValueError(f"a file path within the bucket is required, got {uri!r}")

    fmt = format or _infer_format(bp.path)
    if fmt not in _SINK_METHOD:
        raise ValueError(
            f"unsupported format {fmt!r}; expected one of {sorted(set(_SINK_METHOD))}"
        )

    lf = frame.lazy()
    sink = getattr(lf, _SINK_METHOD[fmt])

    fs = HfFileSystem(token=token)
    with fs.open(bp.fs_path, "wb") as f:
        sink(f, **kwargs)
