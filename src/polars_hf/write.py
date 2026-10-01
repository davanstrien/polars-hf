"""Write a polars frame to a Hugging Face bucket — single file or partitioned.

``sink_bucket`` runs the Polars streaming sink and hands the output to a sink
backend (see ``_sinks.py``). The backend makes nothing visible in the bucket
until the Polars sink has returned without an error, so a query that fails
leaves the destination as it was.

Partitioned writes delegate all splitting (by key, by size, or both) to native
``pl.PartitionBy``. The object names are the ones Polars writes to a local
directory, with either backend.

The bucket API has no transactions. Files are registered in requests of at
most 1,000 operations; ``mode="overwrite"`` deletes stale files in later
requests. A failure between two requests leaves the earlier ones applied.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from polars_hf import _sinks
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
_MODES = ("append", "overwrite", "error")


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


def _check_sink_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject keyword arguments that ``sink_bucket`` cannot honour."""
    if "atomic" in kwargs:
        raise TypeError(
            "sink_bucket() no longer accepts atomic=; see the backend= and mode= "
            "parameters"
        )
    # With lazy=True Polars returns a plan and writes nothing: the destination
    # would be registered as an empty object.
    if kwargs.pop("lazy", False):
        raise ValueError(
            "sink_bucket() runs the query before it returns; lazy=True is not supported"
        )


def _make_run_sink(lf: pl.LazyFrame, fmt: str, sink_kwargs: dict[str, Any]) -> Any:
    """Build the function a backend calls to run the Polars sink."""

    def run_sink(target: Any) -> None:
        result = getattr(lf, _SINK_METHOD[fmt])(target, **sink_kwargs)
        if result is not None:
            # The sink did not run (it returned a plan): the backend must not
            # register anything.
            raise TypeError(
                f"{_SINK_METHOD[fmt]} returned {type(result).__name__} instead of "
                "running the query; sink_bucket() cannot defer a write"
            )

    return run_sink


def _files_with_prefix(backend: _sinks.SinkBackend, prefix: str) -> Iterator[str]:
    """Paths of the files whose path starts with the string ``prefix``."""
    items = backend.api.list_bucket_tree(
        backend.bucket_id, prefix=prefix or None, recursive=True
    )
    for item in items:
        if getattr(item, "type", None) == "file":
            yield item.path


def _files_below(backend: _sinks.SinkBackend, prefix: str) -> list[str]:
    """Paths of the files in the directory ``prefix`` (``""`` is the bucket)."""
    paths = []
    for path in _files_with_prefix(backend, prefix):
        # The listing matches a string prefix: drop siblings such as "out2/x"
        # for the prefix "out".
        if prefix == "" or path.startswith(prefix + "/"):
            paths.append(path)
    return paths


def _raise_if_file_exists(backend: _sinks.SinkBackend, path: str, uri: str) -> None:
    for existing in _files_with_prefix(backend, path):
        if existing == path:
            raise FileExistsError(f"{uri!r} already exists (mode='error')")


def _raise_if_prefix_exists(backend: _sinks.SinkBackend, prefix: str, uri: str) -> None:
    """Raise if a file is at ``prefix`` itself or anywhere below it."""
    below = []
    for path in _files_with_prefix(backend, prefix):
        if prefix != "" and path == prefix:
            raise FileExistsError(f"{uri!r} already exists as a file (mode='error')")
        if prefix == "" or path.startswith(prefix + "/"):
            below.append(path)
    if below:
        raise FileExistsError(
            f"{uri!r} already holds {len(below)} file(s), for example "
            f"{below[0]!r} (mode='error')"
        )


def _partition_prefix(path: str, uri: str, mode: str) -> str:
    """The base prefix of a partitioned write (``""`` is the bucket root)."""
    if path != "" and path.strip("/") == "":
        raise ValueError(
            f"invalid destination {uri!r}: the path consists of slashes only"
        )
    prefix = path[:-1] if path.endswith("/") else path
    if prefix == "":
        if mode == "overwrite":
            # It would delete every other object of the bucket.
            raise ValueError(
                f"mode='overwrite' needs a directory below the bucket root, got {uri!r}"
            )
        return prefix
    _sinks.validate_destination(prefix)
    return prefix


def sink_bucket(
    frame: pl.LazyFrame | pl.DataFrame,
    uri: str,
    *,
    format: str | None = None,
    token: str | None = None,
    partition_by: str | list[str] | None = None,
    max_rows_per_file: int | None = None,
    max_bytes_per_file: int | None = None,
    mode: str = "append",
    backend: str | None = None,
    **kwargs: Any,
) -> None:
    """Write a polars frame to a Hugging Face bucket.

    Without any partition argument, ``uri`` is a single destination file and the
    frame is written there. If ``partition_by``, ``max_rows_per_file``, or
    ``max_bytes_per_file`` is given, ``uri`` is treated as a **base prefix** and the
    output is split into multiple files via native ``pl.PartitionBy``.

    The query runs before the function returns. Files become visible in the
    bucket only after the Polars sink has finished without an error: if the
    query fails, nothing is written and existing objects are not changed.

    Parameters
    ----------
    frame
        A ``LazyFrame`` (streaming) or eager ``DataFrame`` (converted with ``.lazy()``).
    uri
        Destination ``hf://buckets/{namespace}/{name}/{path}`` URI: a file path for
        single-file writes, or a base prefix for partitioned writes.
    format
        ``"parquet"`` (default for partitioned), ``"csv"``, ``"ipc"``, or ``"ndjson"``
        (case-insensitive). For single-file writes it is inferred from the
        extension if omitted.
    token
        Hugging Face token. If ``None``, resolved by ``huggingface_hub``.
    partition_by
        Column name(s) to partition by. The layout is the one Polars writes
        locally: ``key=value/`` directories with percent-encoded values and
        ``__HIVE_DEFAULT_PARTITION__`` for a null key, and files named
        ``00000000.parquet``, ``00000001.parquet``, ... (``.jsonl`` for ndjson).
    max_rows_per_file
        Split each partition further: at most this many rows per file.
    max_bytes_per_file
        Split each partition further, with this size as a target. It is
        Polars' ``approximate_bytes_per_file``: an estimate made while the
        rows are written, so a file can be larger than the value.
    mode
        What to do with objects that are already at the destination:

        * ``"append"`` (default): keep them. An object with the same name as a
          new file is replaced.
        * ``"overwrite"``: list the files below the base prefix before the
          write; after all new files are registered, delete the listed files
          that this call did not write. A file that another writer adds
          during the write is not deleted. The base prefix must be a
          directory below the bucket root. A query that returns no rows
          still deletes the listed files and leaves one file with the schema
          and no rows. For a single-file write this is the same as
          ``"append"``.
        * ``"error"``: raise ``FileExistsError`` before anything is written if
          the destination file exists or, for a partitioned write, if a file
          exists at the base prefix or anywhere below it. The check and the
          write are separate requests, so a concurrent writer is not excluded.
    backend
        ``"stream"`` streams every output file straight into Xet storage and uses
        no local disk for the output. ``"staged"`` writes the output to a local
        temporary directory first and uploads it with
        ``HfApi.batch_bucket_files``, so it needs as much free disk as the
        output is large. ``None`` (default) reads the environment variable
        ``POLARS_HF_SINK_BACKEND`` and otherwise uses ``"stream"`` when the
        installed ``huggingface_hub`` and ``hf_xet`` support it
        (huggingface_hub>=1.19), else ``"staged"``. The ``"staged"`` backend stages
        in the directory named by the environment variable
        ``POLARS_HF_STAGING_DIR``, else in the system temporary directory.
    **kwargs
        Forwarded to the underlying polars ``sink_*``. ``lazy=True`` is rejected.

    Raises
    ------
    ValueError
        For an invalid ``uri``, ``format``, ``mode`` or ``backend``, for
        ``lazy=True``, for ``mode="overwrite"`` on the bucket root, and for a
        destination path the Hub refuses (a backslash, an empty segment, a
        ``.`` or ``..`` segment), including one built from a partition value.
        Such a path is rejected before any file is registered.
    FileExistsError
        With ``mode="error"``, if the destination exists.
    BucketRegistrationError
        If the upload or the registration request fails, or if the bucket
        rejects some of the files (see ``failures``). It is a ``RuntimeError``.
    RuntimeError
        If ``backend="stream"`` is requested and the installed packages do not
        support it or are not compatible with it.

    Notes
    -----
    The write is not transactional. Files are registered in requests of at most
    1,000 operations, and ``mode="overwrite"`` deletes stale files afterwards.
    If one of these requests fails, the error is raised and the earlier
    requests stay applied: the destination can then hold a part of the new
    files, and with ``mode="overwrite"`` it can hold new and stale files
    together.

    A partitioned write whose query returns no rows writes one file with the
    schema and no rows, ``{prefix}/00000000.{extension}``, so the prefix can
    be scanned afterwards.

    Examples
    --------
    >>> import polars_hf as plhf
    >>> plhf.sink_bucket(lf, "hf://buckets/me/data/out.parquet")  # doctest: +SKIP
    >>> plhf.sink_bucket(  # doctest: +SKIP
    ...     lf, "hf://buckets/me/data/by_year", partition_by="year", mode="overwrite"
    ... )
    """
    import polars as pl

    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
    sink_kwargs = dict(kwargs)
    _check_sink_kwargs(sink_kwargs)

    bp = parse_bucket_uri(uri)
    partitioned = (
        partition_by is not None
        or max_rows_per_file is not None
        or max_bytes_per_file is not None
    )
    lf = frame.lazy()

    if format is not None:
        fmt = format.lower()
    elif partitioned:
        fmt = "parquet"
    else:
        fmt = None

    if partitioned:
        prefix = _partition_prefix(bp.path, uri, mode)
    else:
        if not bp.path:
            raise ValueError(f"a file path within the bucket is required, got {uri!r}")
        if bp.path.endswith("/"):
            raise ValueError(
                f"{uri!r} names a directory; a single-file write needs a file path "
                "(or pass a partition argument)"
            )
        _sinks.validate_destination(bp.path)
        if fmt is None:
            fmt = _infer_format(bp.path)
    if fmt not in _SINK_METHOD:
        raise ValueError(f"unsupported format {format!r}")

    sink = _sinks.make_backend(backend, bp.bucket_id, token)
    run_sink = _make_run_sink(lf, fmt, sink_kwargs)

    if not partitioned:
        if mode == "error":
            _raise_if_file_exists(sink, bp.path, uri)
        sink.write_file(run_sink, bp.path)
        return

    if mode == "error":
        _raise_if_prefix_exists(sink, prefix, uri)
    # Listed before the write: only these files can be deleted afterwards, so
    # a file that another writer adds in the meantime is kept.
    existing_before = []
    if mode == "overwrite":
        existing_before = _files_below(sink, prefix)

    spec = _sinks.PartitionSpec(
        key=partition_by,
        max_rows_per_file=max_rows_per_file,
        max_bytes_per_file=max_bytes_per_file,
        extension=_sinks.PARTITION_EXTENSION[fmt],
    )
    written = sink.write_partitioned(run_sink, prefix, spec)
    if not written:
        # No row, so Polars opened no file. Write the schema alone: the prefix
        # then scans back as an empty frame instead of "no such file".
        empty = pl.LazyFrame(schema=lf.collect_schema())
        empty_path = _sinks.join_path(prefix, f"00000000.{spec.extension}")
        written = sink.write_file(_make_run_sink(empty, fmt, sink_kwargs), empty_path)

    if mode == "overwrite":
        kept = set(written)
        stale = []
        for path in existing_before:
            if path not in kept:
                stale.append(path)
        sink.delete(stale)
