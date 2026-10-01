"""Sink backends: how the files that Polars writes reach a bucket.

``sink_bucket`` builds the Polars sink; a backend decides where Polars writes
and how the result is uploaded. Both backends follow the same rule: nothing is
visible in the bucket until the Polars sink has returned without an error.

* :class:`StreamBackend` hands Polars write-only file objects. Every ``write()``
  goes into an ``hf_xet`` upload stream, so no output is staged on local disk.
  When the sink has returned, the streams are finished and the files are
  registered with the bucket ``/batch`` endpoint.
* :class:`StagedBackend` lets Polars write into a local temporary directory and
  uploads it with the public ``HfApi.batch_bucket_files``.

The two backends produce the same object names for the same call: both name
the files of a partitioned write with :func:`partition_file_name`, which
rebuilds the layout Polars' own hive provider writes, and both check every
name with :func:`validate_destination` when Polars asks for the file. Deletes
(``mode="overwrite"``) are one checked ``/batch`` request in both backends.

Neither backend is transactional: the bucket API has no transactions. Files
are registered in requests of at most 1,000 operations, and a failure between
two requests leaves the earlier ones applied.

The stream backend relies on private parts of its dependencies: the session
helpers of ``huggingface_hub.utils._xet`` and the ``/api/buckets/{id}/batch``
request, which it builds itself the way ``HfApi._batch_bucket_files`` does.
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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status, http_backoff

if TYPE_CHECKING:
    import polars as pl

BACKEND_ENV_VAR = "POLARS_HF_SINK_BACKEND"
STAGING_DIR_ENV_VAR = "POLARS_HF_STAGING_DIR"
BACKEND_NAMES = ("stream", "staged")

# The Hub client sends at most this many operations per ``/batch`` request.
BATCH_CHUNK_SIZE = 1000

# Longest path segment accepted, in bytes: the file-name limit of the common
# local file systems, where the staged backend stages the output.
MAX_SEGMENT_BYTES = 255

# Threads that finish the upload streams of one stream write.
_FINISH_THREADS = 16

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
    """An upload or a registration of a write failed.

    Raised by both backends when the request that registers files in the
    bucket (or deletes stale ones) fails, gets no answer, gets an answer that
    does not confirm it, or is answered with rejected operations. The staged
    backend also raises it when its upload fails, with the error of
    ``huggingface_hub`` as the ``__cause__``. An error of the ``hf_xet``
    stream upload of the stream backend is raised as ``hf_xet`` reports it.

    Attributes
    ----------
    failures
        The ``failed`` entries of the ``/batch`` response: one dict per
        rejected operation, with the keys ``path`` and ``error``. Empty when
        the request failed as a whole.
    """

    def __init__(self, message: str, failures: list[dict] | None = None) -> None:
        self.failures = list(failures or [])
        lines = [message]
        for failure in self.failures[:10]:
            lines.append(f"  - {failure.get('path')}: {failure.get('error')}")
        if len(self.failures) > 10:
            lines.append(f"  - ... and {len(self.failures) - 10} more")
        super().__init__("\n".join(lines))


def _rejected(bucket_id: str, failures: list[dict], sent: int) -> str:
    """Message for a request of ``sent`` operations with rejected ones."""
    message = (
        f"bucket {bucket_id!r} rejected {len(failures)} of {sent} operation(s) "
        "of a request"
    )
    applied = sent - len(failures)
    if applied > 0:
        message += f"; the other {applied} were applied"
    return message + ":"


def incompatible_xet_error(error: BaseException) -> RuntimeError:
    """The error for a private Xet API that no longer has the expected shape."""
    try:
        from importlib.metadata import version

        hf_xet_version = version("hf_xet")
    except Exception:
        hf_xet_version = "unknown"
    return RuntimeError(
        "the 'stream' sink backend is not compatible with the installed "
        f"huggingface_hub {huggingface_hub.__version__} / hf_xet {hf_xet_version} "
        f"({type(error).__name__}: {error}). Use backend='staged'."
    )


# Raised by Python when a private function or object has changed its
# signature or lost an attribute.
_API_SHAPE_ERRORS = (TypeError, AttributeError)


def _network_errors() -> tuple[type[BaseException], ...]:
    """Timeouts and connection errors of the HTTP clients huggingface_hub uses."""
    import httpx

    errors: list[type[BaseException]] = [OSError, httpx.HTTPError]
    try:
        import httpx2  # huggingface_hub 2.x
    except ImportError:
        pass
    else:
        errors.append(httpx2.HTTPError)
    return tuple(errors)


_UNKNOWN_STATE = (
    "The state of the destination is unknown: the request may or may not have "
    "been applied. List the destination (for example with "
    "HfApi.list_bucket_tree) to see which files are registered"
)


def _confirms_success(body: Any, sent: int) -> bool:
    """Whether a ``/batch`` answer says that all ``sent`` operations were applied.

    A proxy page, an empty object or a body without these fields is not a
    confirmation, whatever the status code.
    """
    if not isinstance(body, dict):
        return False
    if body.get("failed") != [] or body.get("success") is not True:
        return False
    return body.get("processed") == sent and body.get("succeeded") == sent


def hub_reports_rejected_files() -> bool:
    """Whether ``HfApi.batch_bucket_files`` raises for rejected operations.

    huggingface_hub 2.x raises ``BucketBatchError`` for the ``failed`` entries
    of a ``/batch`` answer. Older versions ignore them, so a rejected file is
    silently missing after a call that returned normally.
    """
    return hasattr(huggingface_hub.errors, "BucketBatchError")


# ---- destination paths ------------------------------------------------------


def validate_destination(path: str) -> None:
    """Raise ``ValueError`` for a bucket path that the Hub refuses.

    The Hub rejects a backslash, an empty segment (a leading or trailing
    slash, ``//``) and the segments ``.`` and ``..``. Control characters and
    segments of more than 255 bytes are refused here as well: the staged backend
    cannot stage such a name on a local file system, and both backends must
    accept the same destinations. For a partitioned write the offending
    segment is the ``key=value`` directory, so the message names the partition
    column and value.
    """
    for segment in path.split("/"):
        if segment == "":
            reason = "an empty path segment"
        elif segment in (".", ".."):
            reason = f"the path segment {segment!r}"
        elif "\\" in segment:
            reason = f"a backslash in the path segment {segment!r}"
        elif _has_control_character(segment):
            reason = f"a control character in the path segment {segment!r}"
        elif len(segment.encode("utf-8")) > MAX_SEGMENT_BYTES:
            reason = (
                f"a path segment of more than {MAX_SEGMENT_BYTES} bytes "
                f"({segment[:40]!r}...)"
            )
        else:
            continue
        raise ValueError(
            f"invalid bucket path {path!r}: sink_bucket does not accept {reason}"
        )


def _has_control_character(text: str) -> bool:
    for character in text:
        if ord(character) < 0x20 or ord(character) == 0x7F:
            return True
    return False


class _Destinations:
    """The bucket paths one partitioned write has handed to Polars."""

    def __init__(self, prefix: str, extension: str) -> None:
        self.prefix = prefix
        self.extension = extension
        self.paths: list[str] = []
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, args: Any) -> tuple[str, str]:
        """Name the file Polars asks for: ``(relative path, bucket path)``.

        ``args`` is what Polars passes to a ``file_path_provider``. Raises for
        a path the bucket would refuse and for a path asked for twice (a
        second file would replace the first one under the same name).
        """
        relative = partition_file_name(
            args.partition_keys, args.index_in_partition, self.extension
        )
        path = join_path(self.prefix, relative)
        validate_destination(path)
        with self._lock:
            if path in self._seen:
                raise RuntimeError(
                    f"Polars asked twice for the output file {path!r}; refusing to "
                    "write two files under one name"
                )
            self._seen.add(path)
            self.paths.append(path)
        return relative, path


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
        self.headers = build_hf_headers(token=token)

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

    def delete(self, paths: list[str]) -> None:
        """Delete objects from the bucket (a missing path is not an error)."""
        operations = []
        for path in paths:
            operations.append({"type": "deleteFile", "path": path})
        self._post_batch(operations)

    def _post_batch(self, operations: list[dict]) -> None:
        """Send operations to the bucket, at most 1,000 per request."""
        url = f"{self.api.endpoint}/api/buckets/{self.bucket_id}/batch"
        headers = {"Content-Type": "application/x-ndjson", **self.headers}
        for chunk in _chunks(operations, BATCH_CHUNK_SIZE):
            lines = []
            for operation in chunk:
                lines.append(json.dumps(operation).encode() + b"\n")
            body = b"".join(lines)
            # http_backoff retries 429 and 5xx answers and connection errors.
            # A repeated request is harmless: the operations are idempotent.
            try:
                response = http_backoff("POST", url, headers=headers, content=body)
            except _network_errors() as error:
                raise BucketRegistrationError(
                    f"the registration request to bucket {self.bucket_id!r} got "
                    f"no answer ({type(error).__name__}: {error}). {_UNKNOWN_STATE}."
                ) from error
            self._check_batch_response(response, sent=len(chunk))

    def _check_batch_response(self, response: Any, sent: int) -> None:
        """Raise unless the bucket confirms that it applied all ``sent`` operations.

        The Hub answers ``{"success", "processed", "succeeded", "failed"}``.
        Rejected operations are listed in ``failed``, in the body of a 200
        (some failed) or of a 422 (all failed).
        """
        body = None
        if response.status_code in (200, 422):
            try:
                body = response.json()
            except ValueError:
                body = None
        if isinstance(body, dict) and isinstance(body.get("failed"), list):
            failures = body["failed"]
            if failures:
                message = _rejected(self.bucket_id, failures, sent)
                raise BucketRegistrationError(message, failures)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as error:
            raise BucketRegistrationError(
                f"the registration in bucket {self.bucket_id!r} failed: {error}"
            ) from error
        if not _confirms_success(body, sent):
            raise BucketRegistrationError(
                f"bucket {self.bucket_id!r} answered {response.status_code} to a "
                "registration request with a body that is not the expected "
                f"confirmation of {sent} operation(s). {_UNKNOWN_STATE}."
            )


# ---- staged backend ------------------------------------------------------------


class StagedBackend(SinkBackend):
    """Stage the output on local disk, then upload it with the public API.

    Local disk use equals the size of the complete output. The staging
    directory is removed when the write ends, with or without an error.
    """

    name = "staged"

    def __init__(self, bucket_id: str, token: str | None) -> None:
        super().__init__(bucket_id, token)
        self.staging_dir = os.environ.get(STAGING_DIR_ENV_VAR) or None

    def _add(self, additions: list[tuple[str, str]], listing_prefix: str) -> None:
        """Upload ``(local path, bucket path)`` pairs with the public API.

        ``listing_prefix`` is a string prefix of all the bucket paths; it is
        used to check the result when the client does not report rejections.
        """
        sizes = {}
        for local, destination in additions:
            sizes[destination] = os.path.getsize(local)
        try:
            self.api.batch_bucket_files(self.bucket_id, add=additions)
        except HfHubHTTPError as error:
            raise BucketRegistrationError(
                f"the upload to bucket {self.bucket_id!r} failed: {error}",
                getattr(error, "failures", None),
            ) from error
        except _network_errors() as error:
            raise BucketRegistrationError(
                f"the upload to bucket {self.bucket_id!r} got no answer "
                f"({type(error).__name__}: {error}). {_UNKNOWN_STATE}."
            ) from error
        if not hub_reports_rejected_files():
            self._verify_added(sizes, listing_prefix)

    def _verify_added(self, sizes: dict[str, int], listing_prefix: str) -> None:
        """Raise if a file of ``{bucket path: size}`` is not in the bucket.

        huggingface_hub 1.x returns normally when the bucket rejects single
        files, so the destination is listed once after the upload.
        """
        listed = {}
        items = self.api.list_bucket_tree(
            self.bucket_id, prefix=listing_prefix or None, recursive=True
        )
        for item in items:
            if getattr(item, "type", None) == "file":
                listed[item.path] = item.size
        failures = []
        for path, size in sizes.items():
            if path not in listed:
                error = "not in the bucket after the upload"
            elif listed[path] != size:
                error = f"has {listed[path]} bytes in the bucket, expected {size}"
            else:
                continue
            failures.append({"path": path, "error": error})
        if failures:
            raise BucketRegistrationError(
                _rejected(self.bucket_id, failures, len(sizes)), failures
            )

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
            self._add([(local, path)], listing_prefix=path)
        return [path]

    def write_partitioned(
        self, run_sink: RunSink, prefix: str, spec: PartitionSpec
    ) -> list[str]:
        destinations = _Destinations(prefix, spec.extension)
        additions = []

        def provider(args: Any) -> str:
            # The same names, checks and errors as the stream backend. Polars
            # creates the directories below the staging directory.
            relative, path = destinations.claim(args)
            additions.append((relative, path))
            return relative

        with self._staging() as directory:
            run_sink(spec.partition_by(directory, file_path_provider=provider))
            local_additions = []
            for relative, path in sorted(additions, key=lambda item: item[1]):
                local = os.path.join(directory, *relative.split("/"))
                local_additions.append((local, path))
            if local_additions:
                self._add(local_additions, listing_prefix=prefix)
        return [path for _, path in local_additions]


# ---- stream backend ------------------------------------------------------------

_XET_REQUIREMENT = (
    "the 'stream' sink backend needs huggingface_hub>=1.19 and the hf_xet that it "
    "requires (installed by huggingface_hub on x86_64 and arm64)"
)


def stream_unavailable_reason() -> str | None:
    """Why the stream backend cannot run here, or ``None`` if it can.

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
    returns an in-memory object with these three methods.
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


def abort_xet_session() -> None:
    """Stop the shared Xet session after a ``KeyboardInterrupt``.

    The session is process-wide: other Xet uploads and downloads that run in
    the same process are cancelled too. ``huggingface_hub`` does the same in
    its own uploads.
    """
    from huggingface_hub.utils import _xet

    _xet.abort_xet_session()


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

    def write(self, data: bytes) -> int:
        # Polars passes ``bytes`` (checked for the four formats on Polars
        # 1.40.0, 1.41.2, 1.44.2 and 2.0.0rc2), which the stream takes as is.
        try:
            self.stream.write(data)
        except BaseException as error:
            # Polars reports a failed write as its own ComputeError, or not at
            # all; keep the original so the backend can raise it.
            self._upload.record_write_error(error)
            raise
        return len(data)

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
        validate_destination(path)
        try:
            stream = self.commit.open_stream(path)
        except _API_SHAPE_ERRORS as error:
            raise incompatible_xet_error(error) from error
        writer = _StreamWriter(self, path, stream)
        with self._lock:
            self.writers.append(writer)
        return writer

    def record_write_error(self, error: BaseException) -> None:
        with self._lock:
            if self.write_error is None:
                self.write_error = error


def _finish_streams(writers: list[_StreamWriter]) -> list[Any]:
    """Finish every upload stream; return the results in the order of ``writers``.

    ``finish()`` blocks on the network for each stream, so the calls run in a
    small thread pool. The first error is raised after all calls have ended.
    """
    if len(writers) <= 1:
        return [writer.stream.finish() for writer in writers]
    workers = min(_FINISH_THREADS, len(writers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(writer.stream.finish) for writer in writers]
    return [future.result() for future in futures]


def _chunks(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class StreamBackend(SinkBackend):
    """Stream every output file into Xet storage; no local staging.

    All upload streams stay open until the Polars sink returns, because Polars
    gives no signal when one file is complete. The files are registered in the
    bucket only after every stream has been stored.
    """

    name = "stream"

    def _upload(self, write: Callable[[_XetUpload], None]) -> list[str]:
        """Run ``write`` against a new commit; register its files on success."""
        try:
            commit = open_xet_commit(self.api.endpoint, self.bucket_id, self.headers)
        except _API_SHAPE_ERRORS as error:
            raise incompatible_xet_error(error) from error
        upload = _XetUpload(commit)
        operations = []
        try:
            write(upload)
            # Do not rely on Polars to raise for a write() that failed.
            if upload.write_error is not None:
                raise upload.write_error
            if not upload.writers:
                commit.abort()
                return []
            try:
                results = _finish_streams(upload.writers)
                for writer, result in zip(upload.writers, results, strict=True):
                    operations.append(self._add_operation(writer.path, result))
                commit.wait_to_finish()
            except _API_SHAPE_ERRORS as error:
                raise incompatible_xet_error(error) from error
        except KeyboardInterrupt:
            # Cancel this commit, then the shared session. Neither call may
            # replace the KeyboardInterrupt.
            for stop in (commit.abort, abort_xet_session):
                try:
                    stop()
                except Exception:
                    pass
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

    def write_file(self, run_sink: RunSink, path: str) -> list[str]:
        def write(upload: _XetUpload) -> None:
            run_sink(upload.open(path))

        return self._upload(write)

    def write_partitioned(
        self, run_sink: RunSink, prefix: str, spec: PartitionSpec
    ) -> list[str]:
        destinations = _Destinations(prefix, spec.extension)

        def write(upload: _XetUpload) -> None:
            def provider(args: Any) -> _StreamWriter:
                _, path = destinations.claim(args)
                return upload.open(path)

            # The base path is not used: the provider returns file objects.
            run_sink(spec.partition_by("unused", file_path_provider=provider))

        return sorted(self._upload(write))


# ---- selection --------------------------------------------------------------


def resolve_backend_name(backend: str | None) -> str:
    """Pick the backend: the argument, then the environment, then the default.

    The default is ``"stream"`` when it can run here and ``"staged"`` otherwise. An
    explicit ``"stream"`` that cannot run raises instead of falling back.
    """
    requested = backend
    if requested is None:
        requested = os.environ.get(BACKEND_ENV_VAR) or None
    if requested is None:
        return "stream" if stream_unavailable_reason() is None else "staged"
    if requested not in BACKEND_NAMES:
        raise ValueError(
            f"unknown sink backend {requested!r}; expected one of {BACKEND_NAMES}"
        )
    if requested == "stream":
        reason = stream_unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"{_XET_REQUIREMENT}; {reason}. Use backend='staged'.")
    return requested


def make_backend(backend: str | None, bucket_id: str, token: str | None) -> SinkBackend:
    name = resolve_backend_name(backend)
    if name == "stream":
        return StreamBackend(bucket_id, token)
    return StagedBackend(bucket_id, token)
