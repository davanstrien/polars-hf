"""Write a polars frame to a Hugging Face bucket — single file or partitioned.

Single-file writes open the bucket path with ``HfFileSystem`` in ``"wb"`` mode and
hand the file object to polars' streaming ``sink_*`` (incremental, bounded memory;
``HfFileSystem`` spills to a local temp file then commits via ``hf_xet`` on close).

Partitioned writes delegate all splitting (by key, by size, or both) to native
``pl.PartitionBy`` and land the files in the bucket one of two ways:

* ``atomic=True`` (default) — polars writes partitions to a local temp dir, then a
  single ``batch_bucket_files`` commit uploads them all. One atomic commit; total
  output is bounded by local disk.
* ``atomic=False`` — a ``file_path_provider`` returns an ``HfFileSystem`` ``"wb"``
  object per partition, so polars streams each partition straight to the bucket.
  Disk-light (handles bigger-than-disk) at the cost of one commit per file — which
  is cheap on buckets, since they are not git-backed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, HfFileSystem

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
_FORMAT_EXT = {"parquet": "parquet", "csv": "csv", "ipc": "ipc", "ndjson": "ndjson"}


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


def _partition_by(
    base_path: str,
    *,
    key: object,
    max_rows_per_file: int | None,
    max_bytes_per_file: int | None,
    file_path_provider: object = None,
) -> object:
    import polars as pl

    kwargs: dict = {}
    if key is not None:
        kwargs["key"] = key
    if max_rows_per_file is not None:
        kwargs["max_rows_per_file"] = max_rows_per_file
    if max_bytes_per_file is not None:
        kwargs["approximate_bytes_per_file"] = max_bytes_per_file
    if file_path_provider is not None:
        kwargs["file_path_provider"] = file_path_provider
    return pl.PartitionBy(base_path, **kwargs)


def _sink_partitioned_atomic(
    lf, bucket_id, prefix, fmt, *, key, max_rows, max_bytes, sink_kwargs, token
) -> None:
    """Design A: native local partition, then one batched bucket commit."""
    tmpdir = tempfile.mkdtemp(prefix="polars-hf-")
    try:
        part = _partition_by(
            tmpdir, key=key, max_rows_per_file=max_rows, max_bytes_per_file=max_bytes
        )
        getattr(lf, _SINK_METHOD[fmt])(part, **sink_kwargs)
        adds = []
        for root, _, files in os.walk(tmpdir):
            for fn in files:
                local = os.path.join(root, fn)
                rel = os.path.relpath(local, tmpdir)
                remote = f"{prefix}/{rel}" if prefix else rel
                adds.append((local, remote))
        if not adds:
            return
        HfApi(token=token).batch_bucket_files(bucket_id, add=adds)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _sink_partitioned_stream(
    lf, base, fmt, *, key, max_rows, max_bytes, sink_kwargs, token
) -> None:
    """Design B: stream each partition straight to the bucket via a provider."""
    fs = HfFileSystem(token=token)
    ext = _FORMAT_EXT[fmt]
    root = base.rstrip("/")

    def provider(args):
        pk = args.partition_keys  # 1-row DataFrame of this partition's key columns
        sub = "/".join(f"{c}={pk[c][0]}" for c in pk.columns)
        rel = f"{sub}/" if sub else ""
        return fs.open(f"{root}/{rel}{args.index_in_partition:08d}.{ext}", "wb")

    part = _partition_by(
        root,
        key=key,
        max_rows_per_file=max_rows,
        max_bytes_per_file=max_bytes,
        file_path_provider=provider,
    )
    getattr(lf, _SINK_METHOD[fmt])(part, **sink_kwargs)


def sink_bucket(
    frame: pl.LazyFrame | pl.DataFrame,
    uri: str,
    *,
    format: str | None = None,
    token: str | None = None,
    partition_by: str | list[str] | None = None,
    max_rows_per_file: int | None = None,
    max_bytes_per_file: int | None = None,
    atomic: bool = True,
    **kwargs: object,
) -> None:
    """Write a polars frame to a Hugging Face bucket.

    Without any partition argument, ``uri`` is a single destination file and the
    frame is written there. If ``partition_by``, ``max_rows_per_file``, or
    ``max_bytes_per_file`` is given, ``uri`` is treated as a **base prefix** and the
    output is split into multiple files via native ``pl.PartitionBy``.

    Parameters
    ----------
    frame
        A ``LazyFrame`` (streaming) or eager ``DataFrame`` (converted with ``.lazy()``).
    uri
        Destination ``hf://buckets/{namespace}/{name}/{path}`` URI: a file path for
        single-file writes, or a base prefix for partitioned writes.
    format
        ``"parquet"`` (default for partitioned), ``"csv"``, ``"ipc"``, or ``"ndjson"``.
        For single-file writes it is inferred from the extension if omitted.
    token
        Hugging Face token. If ``None``, resolved by ``huggingface_hub``.
    partition_by
        Column name(s) to partition by (hive ``key=value/`` layout).
    max_rows_per_file, max_bytes_per_file
        Split each partition further so files stay under these limits.
    atomic
        For partitioned writes: ``True`` (default) stages partitions locally and
        uploads them in one commit (bounded by local disk); ``False`` streams each
        partition straight to the bucket (handles bigger-than-disk; one commit per
        file, which is cheap on buckets).
    **kwargs
        Forwarded to the underlying polars ``sink_*``.

    Examples
    --------
    >>> import polars_hf as plhf
    >>> plhf.sink_bucket(lf, "hf://buckets/me/data/out.parquet")  # doctest: +SKIP
    >>> plhf.sink_bucket(  # doctest: +SKIP
    ...     lf, "hf://buckets/me/data/by_year", partition_by="year"
    ... )
    """
    bp = parse_bucket_uri(uri)
    partitioned = (
        partition_by is not None
        or max_rows_per_file is not None
        or max_bytes_per_file is not None
    )
    lf = frame.lazy()

    if not partitioned:
        if not bp.path:
            raise ValueError(f"a file path within the bucket is required, got {uri!r}")
        fmt = format or _infer_format(bp.path)
        if fmt not in _SINK_METHOD:
            raise ValueError(f"unsupported format {fmt!r}")
        fs = HfFileSystem(token=token)
        with fs.open(bp.fs_path, "wb") as f:
            getattr(lf, _SINK_METHOD[fmt])(f, **kwargs)
        return

    fmt = format or "parquet"
    if fmt not in _SINK_METHOD:
        raise ValueError(f"unsupported format {fmt!r}")

    if atomic:
        _sink_partitioned_atomic(
            lf,
            bp.bucket_id,
            bp.path.rstrip("/"),
            fmt,
            key=partition_by,
            max_rows=max_rows_per_file,
            max_bytes=max_bytes_per_file,
            sink_kwargs=kwargs,
            token=token,
        )
    else:
        _sink_partitioned_stream(
            lf,
            bp.fs_path,
            fmt,
            key=partition_by,
            max_rows=max_rows_per_file,
            max_bytes=max_bytes_per_file,
            sink_kwargs=kwargs,
            token=token,
        )
