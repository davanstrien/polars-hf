"""Sink backends: how the files that Polars writes reach a bucket.

``sink_bucket`` builds the Polars sink; a backend decides where Polars writes
and how the result is uploaded. Both backends follow the same rule: nothing is
visible in the bucket until the Polars sink has returned without an error.

* :class:`XetBackend` hands Polars write-only file objects. Every ``write()``
  goes into an ``hf_xet`` upload stream, so no output is staged on local disk.
  When the sink has returned, the streams are finished and the files are
  registered with the bucket ``/batch`` endpoint.
* :class:`HubBackend` lets Polars write into a local temporary directory and
  uploads it with the public ``HfApi.batch_bucket_files``.

The two backends produce the same object names for the same call. The hub
backend gets the partitioned layout from Polars itself (``pl.PartitionBy`` on
a local directory); the xet backend rebuilds the same names in
:func:`partition_file_name`.

Neither backend is transactional: the bucket API has no transactions. Files
are registered in requests of at most 1,000 operations, and a failure between
two requests leaves the earlier ones applied.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status, http_backoff

if TYPE_CHECKING:
    import polars as pl

BACKEND_ENV_VAR = "POLARS_HF_SINK_BACKEND"
STAGING_DIR_ENV_VAR = "POLARS_HF_STAGING_DIR"
BACKEND_NAMES = ("xet", "hub")

# The Hub client sends at most this many operations per ``/batch`` request.
BATCH_CHUNK_SIZE = 1000

# File extension Polars gives the files of a partitioned sink, per format.
PARTITION_EXTENSION = {
    "parquet": "parquet",
    "csv": "csv",
    "ipc": "ipc",
    "ndjson": "jsonl",
}

# Polars' hive layout (crates/polars-io/src/hive.rs): a null key is written as
# this name, and the bytes below are percent-encoded in a key value. Control
# characters and all non-ASCII bytes are encoded too.
HIVE_NULL_VALUE = "__HIVE_DEFAULT_PARTITION__"
_HIVE_ENCODED_ASCII = frozenset(b"/=%: ")


class BucketRegistrationError(RuntimeError):
    """The bucket rejected some of the files of a write.

    Attributes
    ----------
    failures
        The ``failed`` entries of the ``/batch`` response: one dict per
        rejected operation, with the keys ``path`` and ``error``.
    """

    def __init__(self, bucket_id: str, failures: list[dict]) -> None:
        self.failures = failures
        lines = []
        for failure in failures[:10]:
            lines.append(f"  - {failure.get('path')}: {failure.get('error')}")
        if len(failures) > 10:
            lines.append(f"  - ... and {len(failures) - 10} more")
        super().__init__(
            f"bucket {bucket_id!r} rejected {len(failures)} operation(s); the other "
            "operations of the same request were applied:\n" + "\n".join(lines)
        )


# ---- object names -----------------------------------------------------------


def hive_encode(value: str | None) -> str:
    """Encode one partition key value the way Polars' hive writer does."""
    if value is None:
        return HIVE_NULL_VALUE
    encoded = []
    for byte in value.encode("utf-8"):
        is_control = byte < 0x20 or byte == 0x7F
        if is_control or byte >= 0x80 or byte in _HIVE_ENCODED_ASCII:
            encoded.append(f"%{byte:02X}")
        else:
            encoded.append(chr(byte))
    return "".join(encoded)


def partition_file_name(
    partition_keys: pl.DataFrame, index_in_partition: int, extension: str
) -> str:
    """The path Polars' own hive provider gives a partition file.

    ``partition_keys`` is the one-row frame of the key columns that Polars
    passes to a ``file_path_provider`` (no columns when the sink only splits by
    size). The result is relative to the base prefix, for example
    ``"g=a%2Fb/00000000.parquet"``.
    """
    import polars as pl

    directories = []
    for name in partition_keys.columns:
        # The same cast Polars applies: booleans become "true"/"false", etc.
        value = partition_keys.get_column(name).cast(pl.String).item()
        directories.append(f"{name}={hive_encode(value)}/")
    return f"{''.join(directories)}{index_in_partition:08x}.{extension}"


# ---- what a backend is asked to write ---------------------------------------


@dataclass(frozen=True)
class PartitionSpec:
    """The partition options of one ``sink_bucket`` call."""

    key: str | list[str] | None
    max_rows_per_file: int | None
    max_bytes_per_file: int | None
    extension: str

    def partition_by(self, base_path: str, file_path_provider: Any = None) -> Any:
        """Build the ``pl.PartitionBy`` target for this spec."""
        import polars as pl

        kwargs: dict[str, Any] = {}
        if self.key is not None:
            kwargs["key"] = self.key
        if self.max_rows_per_file is not None:
            kwargs["max_rows_per_file"] = self.max_rows_per_file
        if self.max_bytes_per_file is not None:
            kwargs["approximate_bytes_per_file"] = self.max_bytes_per_file
        if file_path_provider is not None:
            kwargs["file_path_provider"] = file_path_provider
        return pl.PartitionBy(base_path, **kwargs)


# Runs the Polars sink into the given target (a path, a file object or a
# ``pl.PartitionBy``). It raises if the query fails.
RunSink = Callable[[Any], None]


def join_path(prefix: str, relative: str) -> str:
    return f"{prefix}/{relative}" if prefix else relative


class SinkBackend(ABC):
    """Writes the output of one ``sink_bucket`` call to one bucket."""

    name: str

    def __init__(self, bucket_id: str, token: str | None) -> None:
        self.bucket_id = bucket_id
        self.token = token
        self.api = HfApi(token=token)

    @abstractmethod
    def write_file(self, run_sink: RunSink, path: str) -> list[str]:
        """Write one file to ``path``; return the bucket paths written."""

    @abstractmethod
    def write_partitioned(
        self, run_sink: RunSink, prefix: str, spec: PartitionSpec
    ) -> list[str]:
        """Write a partitioned output below ``prefix``; return the paths.

        The list is empty when the query produced no file.
        """

    @abstractmethod
    def delete(self, paths: list[str]) -> None:
        """Delete objects from the bucket."""


# ---- hub backend ------------------------------------------------------------


class HubBackend(SinkBackend):
    """Stage the output on local disk, then upload it with the public API.

    Local disk use equals the size of the complete output. The staging
    directory is removed when the write ends, with or without an error.
    """

    name = "hub"

    def __init__(
        self, bucket_id: str, token: str | None, staging_dir: str | None = None
    ) -> None:
        super().__init__(bucket_id, token)
        self.staging_dir = staging_dir or os.environ.get(STAGING_DIR_ENV_VAR) or None

    @contextmanager
    def _staging(self) -> Iterator[str]:
        directory = tempfile.mkdtemp(prefix="polars-hf-", dir=self.staging_dir)
        try:
            yield directory
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def write_file(self, run_sink: RunSink, path: str) -> list[str]:
        with self._staging() as directory:
            local = os.path.join(directory, "data")
            run_sink(local)
            self.api.batch_bucket_files(self.bucket_id, add=[(local, path)])
        return [path]

    def write_partitioned(
        self, run_sink: RunSink, prefix: str, spec: PartitionSpec
    ) -> list[str]:
        with self._staging() as directory:
            run_sink(spec.partition_by(directory))
            additions = []
            for root, _, names in os.walk(directory):
                for name in names:
                    local = os.path.join(root, name)
                    relative = os.path.relpath(local, directory).replace(os.sep, "/")
                    additions.append((local, join_path(prefix, relative)))
            additions.sort(key=lambda addition: addition[1])
            if additions:
                self.api.batch_bucket_files(self.bucket_id, add=additions)
        return [destination for _, destination in additions]

    def delete(self, paths: list[str]) -> None:
        if paths:
            self.api.batch_bucket_files(self.bucket_id, delete=paths)


# ---- xet backend ------------------------------------------------------------

_XET_REQUIREMENT = (
    "the 'xet' sink backend needs huggingface_hub>=1.19 with hf_xet>=1.5.1 "
    "(installed by huggingface_hub on x86_64 and arm64)"
)


def xet_unavailable_reason() -> str | None:
    """Why the xet backend cannot run here, or ``None`` if it can.

    The backend needs ``XetUploadCommit.start_upload_stream`` from ``hf_xet``
    and the Xet session helpers of ``huggingface_hub.utils._xet`` (a private
    module; the helpers exist from huggingface_hub 1.19.0).
    """
    try:
        import hf_xet
    except ImportError:
        return "hf_xet is not installed"
    upload_commit = getattr(hf_xet, "XetUploadCommit", None)
    if not hasattr(upload_commit, "start_upload_stream"):
        return "the installed hf_xet has no streaming upload"
    if not hasattr(hf_xet, "SKIP_SHA256"):
        return "the installed hf_xet has no SKIP_SHA256"
    try:
        from huggingface_hub.utils._xet import (  # noqa: F401
            XetTokenType,
            get_xet_session,
            xet_connection_info_refresh_url,
            xet_headers_without_auth,
        )
    except ImportError:
        return (
            f"huggingface_hub {huggingface_hub.__version__} has no Xet session helpers"
        )
    return None


class _XetCommit:
    """One ``hf_xet`` upload commit, reduced to what the backend uses.

    The offline tests replace :func:`open_xet_commit` with a function that
    returns an in-memory object with these four methods.
    """

    def __init__(self, commit: Any) -> None:
        self._commit = commit

    def open_stream(self, name: str) -> Any:
        """Start an upload stream: ``write(bytes)``, then ``finish()``.

        ``finish()`` returns an object whose ``xet_info.hash`` is the Xet hash
        of the bytes written.
        """
        from hf_xet import SKIP_SHA256

        return self._commit.start_upload_stream(name=name, sha256=SKIP_SHA256)

    def wait_to_finish(self) -> None:
        """Block until every finished stream is stored. Call once."""
        self._commit.wait_to_finish()

    def abort(self) -> None:
        """Cancel the commit; later writes to its streams raise."""
        self._commit.abort()

    def interrupt(self) -> None:
        """Stop the shared Xet session after a ``KeyboardInterrupt``."""
        try:
            from huggingface_hub.utils._xet import abort_xet_session
        except ImportError:
            self._commit.abort()
        else:
            abort_xet_session()


def open_xet_commit(endpoint: str, bucket_id: str, headers: dict[str, str]) -> Any:
    """Open an upload commit for ``bucket_id`` on the shared Xet session."""
    from huggingface_hub.utils._xet import (
        XetTokenType,
        get_xet_session,
        xet_connection_info_refresh_url,
        xet_headers_without_auth,
    )

    refresh_url = xet_connection_info_refresh_url(
        token_type=XetTokenType.WRITE,
        repo_id=bucket_id,
        repo_type="bucket",
        endpoint=endpoint,
    )
    commit = get_xet_session().new_upload_commit(
        token_refresh_url=refresh_url,
        token_refresh_headers=headers,
        custom_headers=xet_headers_without_auth(headers),
    )
    return _XetCommit(commit)


class _StreamWriter:
    """Write-only file object for Polars; the bytes go to one upload stream.

    Polars does not call ``close()`` on a Python file object, so the stream is
    finished by the backend after the sink has returned.
    """

    def __init__(self, upload: _XetUpload, path: str, stream: Any) -> None:
        self._upload = upload
        self.path = path
        self.stream = stream

    def write(self, data: Any) -> int:
        chunk = data if isinstance(data, bytes) else bytes(data)
        try:
            self.stream.write(chunk)
        except BaseException as error:
            # Polars reports a failed write as its own ComputeError; keep the
            # original so the backend can raise it.
            self._upload.record_write_error(error)
            raise
        return len(chunk)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False


class _XetUpload:
    """The open streams of one write, and the error of the first failed one."""

    def __init__(self, commit: Any) -> None:
        self.commit = commit
        self.writers: list[_StreamWriter] = []
        self.write_error: BaseException | None = None
        self._lock = threading.Lock()

    def open(self, path: str) -> _StreamWriter:
        writer = _StreamWriter(self, path, self.commit.open_stream(path))
        with self._lock:
            self.writers.append(writer)
        return writer

    def record_write_error(self, error: BaseException) -> None:
        with self._lock:
            if self.write_error is None:
                self.write_error = error


def _chunks(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class XetBackend(SinkBackend):
    """Stream every output file into Xet storage; no local staging.

    All upload streams stay open until the Polars sink returns, because Polars
    gives no signal when one file is complete. The files are registered in the
    bucket only after every stream has been stored.
    """

    name = "xet"

    def __init__(self, bucket_id: str, token: str | None) -> None:
        super().__init__(bucket_id, token)
        self.headers = build_hf_headers(token=token)

    def _upload(self, write: Callable[[_XetUpload], None]) -> list[str]:
        """Run ``write`` against a new commit; register its files on success."""
        commit = open_xet_commit(self.api.endpoint, self.bucket_id, self.headers)
        upload = _XetUpload(commit)
        operations = []
        try:
            write(upload)
            if not upload.writers:
                commit.abort()
                return []
            for writer in upload.writers:
                result = writer.stream.finish()
                operations.append(self._add_operation(writer.path, result))
            commit.wait_to_finish()
        except KeyboardInterrupt:
            commit.interrupt()
            raise
        except BaseException as error:
            try:
                commit.abort()
            except Exception:
                pass  # the commit is already unusable; raise the first error
            write_error = upload.write_error
            if write_error is not None and write_error is not error:
                raise write_error from error
            raise
        self._post_batch(operations)
        return [operation["path"] for operation in operations]

    def _add_operation(self, path: str, result: Any) -> dict:
        operation = {
            "type": "addFile",
            "path": path,
            "xetHash": result.xet_info.hash,
            "mtime": int(time.time() * 1000),
        }
        content_type = mimetypes.guess_type(path)[0]
        if content_type is not None:
            operation["contentType"] = content_type
        return operation

    def _post_batch(self, operations: list[dict]) -> None:
        """Send operations to the bucket, at most 1,000 per request."""
        url = f"{self.api.endpoint}/api/buckets/{self.bucket_id}/batch"
        headers = {"Content-Type": "application/x-ndjson", **self.headers}
        for chunk in _chunks(operations, BATCH_CHUNK_SIZE):
            lines = []
            for operation in chunk:
                lines.append(json.dumps(operation).encode() + b"\n")
            body = b"".join(lines)
            response = http_backoff("POST", url, headers=headers, content=body)
            # Rejected operations are listed in the body of a 200 (some failed)
            # or of a 422 (all failed).
            if response.status_code in (200, 422):
                try:
                    failures = response.json().get("failed", [])
                except ValueError:
                    failures = []
                if failures:
                    raise BucketRegistrationError(self.bucket_id, failures)
            hf_raise_for_status(response)

    def write_file(self, run_sink: RunSink, path: str) -> list[str]:
        def write(upload: _XetUpload) -> None:
            run_sink(upload.open(path))

        return self._upload(write)

    def write_partitioned(
        self, run_sink: RunSink, prefix: str, spec: PartitionSpec
    ) -> list[str]:
        def write(upload: _XetUpload) -> None:
            def provider(args: Any) -> _StreamWriter:
                relative = partition_file_name(
                    args.partition_keys, args.index_in_partition, spec.extension
                )
                return upload.open(join_path(prefix, relative))

            # The base path is not used: the provider returns file objects.
            run_sink(spec.partition_by("unused", file_path_provider=provider))

        return sorted(self._upload(write))

    def delete(self, paths: list[str]) -> None:
        operations = []
        for path in paths:
            operations.append({"type": "deleteFile", "path": path})
        self._post_batch(operations)


# ---- selection --------------------------------------------------------------


def resolve_backend_name(backend: str | None) -> str:
    """Pick the backend: the argument, then the environment, then the default.

    The default is ``"xet"`` when it can run here and ``"hub"`` otherwise. An
    explicit ``"xet"`` that cannot run raises instead of falling back.
    """
    requested = backend
    if requested is None:
        requested = os.environ.get(BACKEND_ENV_VAR) or None
    if requested is None:
        return "xet" if xet_unavailable_reason() is None else "hub"
    if requested not in BACKEND_NAMES:
        raise ValueError(
            f"unknown sink backend {requested!r}; expected one of {BACKEND_NAMES}"
        )
    if requested == "xet":
        reason = xet_unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"{_XET_REQUIREMENT}; {reason}. Use backend='hub'.")
    return requested


def make_backend(
    backend: str | None,
    bucket_id: str,
    token: str | None,
    staging_dir: str | None = None,
) -> SinkBackend:
    name = resolve_backend_name(backend)
    if name == "xet":
        return XetBackend(bucket_id, token)
    return HubBackend(bucket_id, token, staging_dir=staging_dir)
