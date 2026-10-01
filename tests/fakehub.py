"""An offline stand-in for the parts of the Hugging Face Hub that polars-hf uses.

Two local HTTP servers run on different ports, so they are different origins:

* the **hub** server answers the bucket API (``/api/buckets/...``) and the
  ``/buckets/{id}/resolve/{path}`` redirect, and answers 401 unless the
  ``Authorization`` header carries an accepted token, like the real Hub does
  for private buckets;
* the **cdn** server stands in for ``cas-bridge.xethub.hf.co``: it serves the
  "presigned" URLs with HTTP range support and no authentication.

Every request is recorded (:attr:`FakeHub.requests`) and faults can be scripted
per route (:meth:`FakeHub.add_fault`), so tests can assert on request counts,
bytes transferred and error handling without any network access.

File data does not go over HTTP: the real client uploads to Xet storage with
a native extension. :meth:`FakeHub.patch_uploads` adds one seam per sink
backend of ``sink_bucket``:

* **staged backend.** The private ``HfApi._batch_bucket_files`` (one upload + one
  ``/batch`` request) is replaced with a function that stores the files in the
  fake bucket. The public ``HfApi.batch_bucket_files`` stays real, so its
  client-side chunking (1,000 operations per call) and its non-transactional
  behaviour are exercised.
* **stream backend.** ``polars_hf._sinks.open_xet_commit`` is replaced with a
  function that returns a :class:`MemoryCommit`: its streams keep the bytes in
  memory and put them in the fake content store when the commit finishes. The
  Xet protocol itself is not faked. Everything after the upload is real: the
  backend registers the files with ``POST /api/buckets/{id}/batch`` over HTTP,
  and that route stores the content the ``xetHash`` names. The patch also
  reports the stream backend as available, so the offline suite covers it with
  every supported ``huggingface_hub``; the real upload is covered by the
  staging tests.

Both seams append to :attr:`FakeHub.batch_calls`.

The ``/batch`` route was compared with the Hub CI instance on 2026-10-01:

* the reply is ``{"success", "processed", "succeeded", "failed"}``; a request
  with some rejected operations answers 200, lists them in ``failed`` and
  applies the others;
* ``deleteFile`` of a missing path succeeds;
* the server accepts more than 1,000 operations in one request (the limit of
  1,000 is the client's), and so does the fake.

Two differences: the fake rejects an ``addFile`` whose ``xetHash`` it never
received (the Hub CI instance accepts any hash), and the 422 for a request
whose operations all fail follows the ``huggingface_hub`` client, not an
observation.

The listing semantics were copied from the Hub CI instance
(``hub-ci.huggingface.co``) on 2026-10-01:

* the ``tree`` prefix is a plain *string* prefix, not a directory name;
* a non-recursive listing of a directory returns its direct children;
* a non-recursive listing of anything else returns the entries of the parent
  directory whose path starts with the prefix;
* an unknown prefix returns ``[]`` with status 200;
* ``HEAD /buckets/{id}/resolve/{path}`` answers 302 for a file and 404
  ``EntryNotFound`` for a directory or a missing path;
* ``HEAD /buckets/{id}/tree/{path}`` (the directory web page) answers 401 to a
  token;
* a batch on a missing bucket answers 404 ``RepoNotFound``;
* a batch answers 422 for a destination that is empty, starts or ends with
  ``/``, contains ``//``, a ``..`` segment or a backslash.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import polars as pl

HUB = "hub"
CDN = "cdn"

SIGNATURE = "fakesignature"
_TIMESTAMP = "2026-01-01T00:00:00.000Z"
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class ScriptedUploadError(RuntimeError):
    """Raised by a patched upload (hub or stream backend) when told to fail."""


@dataclass(frozen=True)
class RecordedRequest:
    """One request received by the hub or the cdn server."""

    origin: str
    method: str
    path: str
    query: str
    range: str | None
    authorization: str | None
    status: int
    body_bytes: int

    @property
    def has_authorization(self) -> bool:
        return self.authorization is not None


@dataclass
class BatchCall:
    """One batch of bucket operations (one chunk).

    ``via`` is ``"client"`` for a call to the patched
    ``HfApi._batch_bucket_files`` and ``"http"`` for a ``POST .../batch``
    request. ``failed`` is true if any operation of the batch was refused.
    """

    bucket_id: str
    added: dict[str, int]
    deleted: list[str]
    failed: bool = False
    via: str = "client"


@dataclass
class _Fault:
    origin: str
    method: str
    pattern: re.Pattern[str]
    status: int
    remaining: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


@dataclass
class _Reply:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


def _json_reply(payload: object, status: int = 200) -> _Reply:
    body = json.dumps(payload).encode()
    return _Reply(status, {"Content-Type": "application/json; charset=utf-8"}, body)


def _error_reply(status: int, code: str, message: str) -> _Reply:
    reply = _json_reply({"error": message}, status)
    reply.headers["X-Error-Code"] = code
    reply.headers["X-Error-Message"] = message
    return reply


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _MemoryStream:
    """Stand-in for ``hf_xet.XetStreamUpload``: the bytes stay in memory."""

    def __init__(self, commit: MemoryCommit, name: str) -> None:
        self._commit = commit
        self.name = name
        self.chunks: list[bytes] = []
        self.finished = False

    def write(self, data: bytes) -> None:
        hub = self._commit.hub
        if self._commit.aborted:
            raise RuntimeError("write to a stream of an aborted commit")
        if hub._count("stream_writes") == hub.fail_stream_write_on_call:
            raise ScriptedUploadError(f"scripted failure of a write to {self.name!r}")
        self.chunks.append(bytes(data))

    def finish(self) -> SimpleNamespace:
        hub = self._commit.hub
        if hub._count("stream_finishes") == hub.fail_stream_finish_on_call:
            raise ScriptedUploadError(f"scripted failure of the upload {self.name!r}")
        self.finished = True
        data = b"".join(self.chunks)
        info = SimpleNamespace(hash=_content_hash(data), file_size=len(data))
        return SimpleNamespace(xet_info=info)


class MemoryCommit:
    """Stand-in for the upload commit of the stream backend.

    The content of the finished streams reaches the fake content store in
    :meth:`wait_to_finish`. A ``/batch`` request sent before that, or after
    :meth:`abort`, names hashes the fake does not know and is refused.
    """

    def __init__(self, hub: FakeHub, bucket_id: str) -> None:
        self.hub = hub
        self.bucket_id = bucket_id
        self.streams: list[_MemoryStream] = []
        self.finished = False
        self.aborted = False

    def open_stream(self, name: str) -> _MemoryStream:
        stream = _MemoryStream(self, name)
        with self.hub._lock:
            self.streams.append(stream)
        return stream

    def wait_to_finish(self) -> None:
        if self.aborted:
            raise RuntimeError("wait_to_finish on an aborted commit")
        for stream in self.streams:
            if not stream.finished:
                raise RuntimeError(f"stream {stream.name!r} was not finished")
            self.hub._store_blob(b"".join(stream.chunks))
        self.finished = True

    def abort(self) -> None:
        self.aborted = True


def _is_valid_destination(path: str) -> bool:
    """Whether the real Hub accepts ``path`` as a bucket file path."""
    if path == "" or path.startswith("/") or path.endswith("/"):
        return False
    if "//" in path or "\\" in path:
        return False
    return ".." not in path.split("/")


class FakeHub:
    """In-memory buckets served over two local HTTP origins.

    Use as a context manager, or call :meth:`start` and :meth:`stop`.

    Parameters
    ----------
    token
        The token the hub server accepts (``Authorization: Bearer <token>``).
        More tokens can be added with :meth:`accept_token`.

    Attributes
    ----------
    requests
        Every request received, in arrival order.
    batch_calls
        Every batch of bucket operations, in arrival order: the calls to the
        patched ``HfApi._batch_bucket_files`` and the ``POST .../batch``
        requests. One per chunk of at most 1,000 operations.
    fail_batch_on_call
        If set to ``N``, the ``N``-th batch (1-based) fails and stores
        nothing: the patched method raises :class:`ScriptedUploadError`, the
        HTTP route answers 403.
    reject_paths
        ``addFile`` operations of a ``POST .../batch`` request for these paths
        are refused and listed in ``failed``; the others are applied.
    drop_paths
        The patched ``HfApi._batch_bucket_files`` does not store these paths
        and raises nothing, like huggingface_hub 1.x when the bucket rejects
        single files of a request.
    commits
        Every :class:`MemoryCommit` opened by the stream backend.
    session_aborts
        How often the stream backend stopped the shared Xet session (what it
        does after a ``KeyboardInterrupt``). The real session is not touched.
    fail_stream_write_on_call, fail_stream_finish_on_call
        If set to ``N``, the ``N``-th ``write()`` / ``finish()`` (1-based)
        over all streams of all commits raises :class:`ScriptedUploadError`.
    before_batch
        Optional callable run at the start of every batch, before the files
        are read. Tests use it to measure local staging.
    """

    def __init__(self, token: str) -> None:
        self._tokens = {token}
        self.requests: list[RecordedRequest] = []
        self.batch_calls: list[BatchCall] = []
        self.fail_batch_on_call: int | None = None
        self.reject_paths: set[str] = set()
        self.drop_paths: set[str] = set()
        self.commits: list[MemoryCommit] = []
        self.session_aborts = 0
        self.fail_stream_write_on_call: int | None = None
        self.fail_stream_finish_on_call: int | None = None
        self.before_batch = None
        self._counters: dict[str, int] = {}
        self._buckets: dict[str, dict[str, bytes]] = {}
        # sha256 -> content, for every object ever stored (the "CAS").
        self._blobs: dict[str, bytes] = {}
        self._faults: list[_Fault] = []
        self._lock = threading.Lock()
        self._servers: dict[str, ThreadingHTTPServer] = {}
        self._threads: list[threading.Thread] = []

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> FakeHub:
        for origin in (HUB, CDN):
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self, origin))
            server.daemon_threads = True
            # A short poll interval keeps shutdown (once per test) fast.
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            thread.start()
            self._servers[origin] = server
            self._threads.append(thread)
        return self

    def stop(self) -> None:
        for server in self._servers.values():
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(timeout=5)
        self._servers.clear()
        self._threads.clear()

    def __enter__(self) -> FakeHub:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def endpoint(self) -> str:
        """Base URL of the hub server (the value for ``constants.ENDPOINT``)."""
        return f"http://127.0.0.1:{self._servers[HUB].server_address[1]}"

    @property
    def cdn_endpoint(self) -> str:
        """Base URL of the server that serves the presigned URLs."""
        return f"http://127.0.0.1:{self._servers[CDN].server_address[1]}"

    def accept_token(self, token: str) -> None:
        """Make the hub server accept one more token."""
        with self._lock:
            self._tokens.add(token)

    # ---- bucket contents ---------------------------------------------------

    def create_bucket(self, bucket_id: str) -> str:
        with self._lock:
            self._buckets.setdefault(bucket_id, {})
        return bucket_id

    def put(self, bucket_id: str, path: str, data: bytes) -> None:
        """Store raw bytes at ``path`` (the bucket is created if needed)."""
        with self._lock:
            self._buckets.setdefault(bucket_id, {})[path] = data
            self._blobs[_content_hash(data)] = data

    def put_parquet(self, bucket_id: str, path: str, df: pl.DataFrame, **kw) -> int:
        """Store ``df`` as a parquet file and return its size in bytes."""
        buffer = io.BytesIO()
        df.write_parquet(buffer, **kw)
        data = buffer.getvalue()
        self.put(bucket_id, path, data)
        return len(data)

    def files(self, bucket_id: str, prefix: str = "") -> list[str]:
        """Sorted paths in the bucket that start with ``prefix``."""
        with self._lock:
            paths = list(self._buckets.get(bucket_id, {}))
        return sorted(p for p in paths if p.startswith(prefix))

    def read(self, bucket_id: str, path: str) -> bytes:
        with self._lock:
            return self._buckets[bucket_id][path]

    # ---- request log -------------------------------------------------------

    def reset_log(self) -> None:
        with self._lock:
            self.requests.clear()

    def matching(
        self,
        *,
        origin: str | None = None,
        method: str | None = None,
        path_contains: str | None = None,
    ) -> list[RecordedRequest]:
        """Recorded requests that match every given criterion."""
        with self._lock:
            recorded = list(self.requests)
        selected = []
        for request in recorded:
            if origin is not None and request.origin != origin:
                continue
            if method is not None and request.method != method:
                continue
            if path_contains is not None and path_contains not in request.path:
                continue
            selected.append(request)
        return selected

    @property
    def cdn_bytes_served(self) -> int:
        """Total body bytes sent by the cdn server (file data transferred)."""
        return sum(r.body_bytes for r in self.matching(origin=CDN, method="GET"))

    # ---- scripted faults ---------------------------------------------------

    def add_fault(
        self,
        origin: str,
        method: str,
        pattern: str,
        status: int,
        *,
        times: int = 1,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        """Answer the next ``times`` matching requests with ``status``.

        ``pattern`` is a regular expression searched in the percent-decoded
        request path. Faults are consumed in the order they were added, so
        ``add_fault(..., 429)`` followed by a normal bucket file scripts
        "429, then 200".
        """
        fault = _Fault(
            origin=origin,
            method=method,
            pattern=re.compile(pattern),
            status=status,
            remaining=times,
            headers=headers or {},
            body=body,
        )
        with self._lock:
            self._faults.append(fault)

    def _take_fault(self, origin: str, method: str, path: str) -> _Fault | None:
        with self._lock:
            for fault in self._faults:
                if fault.remaining <= 0:
                    continue
                if fault.origin != origin or fault.method != method:
                    continue
                if fault.pattern.search(path) is None:
                    continue
                fault.remaining -= 1
                return fault
        return None

    # ---- uploads -----------------------------------------------------------

    def _count(self, name: str) -> int:
        """Increment the counter ``name`` and return its new value."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + 1
            return self._counters[name]

    def _abort_session(self) -> None:
        with self._lock:
            self.session_aborts += 1

    def _store_blob(self, data: bytes) -> None:
        with self._lock:
            self._blobs[_content_hash(data)] = data

    def open_commit(self, endpoint: str, bucket_id: str, headers: dict) -> MemoryCommit:
        """What replaces ``polars_hf._sinks.open_xet_commit``."""
        commit = MemoryCommit(self, bucket_id)
        with self._lock:
            self.commits.append(commit)
        return commit

    def patch_uploads(self, monkeypatch) -> None:
        """Replace the upload of both sink backends with in-memory uploads."""
        from huggingface_hub import HfApi

        from polars_hf import _sinks

        hub = self

        monkeypatch.setattr(_sinks, "open_xet_commit", self.open_commit)
        monkeypatch.setattr(_sinks, "stream_unavailable_reason", lambda: None)
        monkeypatch.setattr(_sinks, "abort_xet_session", self._abort_session)

        def _batch_bucket_files(
            api, bucket_id, *, add=None, copy=None, delete=None, token=None, **_
        ) -> None:
            hub._batch(api.endpoint, bucket_id, add or [], copy or [], delete or [])

        # This relies on a PRIVATE method of huggingface_hub. Its name and
        # keyword arguments were verified on 1.12.0, 1.17.0 and 2.0.0; check
        # them again when the supported range moves.
        monkeypatch.setattr(HfApi, "_batch_bucket_files", _batch_bucket_files)

    def _raise_http_error(self, url: str, status: int, message: str) -> None:
        """Raise what the real client raises for this ``/batch`` response.

        The exception is built directly (not with ``hf_raise_for_status``):
        the HTTP library behind ``huggingface_hub`` differs between versions.
        """
        import httpx
        from huggingface_hub import errors

        response = httpx.Response(
            status, json={"error": message}, request=httpx.Request("POST", url)
        )
        error_class = errors.HfHubHTTPError
        if status == 404:
            # Older clients have no bucket-specific error class.
            error_class = getattr(errors, "BucketNotFoundError", error_class)
        raise error_class(f"{status} Client Error: {message}", response=response)

    def _batch(
        self, endpoint: str, bucket_id: str, add: list, copy: list, delete: list
    ) -> None:
        if copy:
            raise NotImplementedError("the fake bucket does not implement copy=")
        if self.before_batch is not None:
            self.before_batch()

        # The public method passes tuples and strings; accept the client's
        # internal operation objects too.
        additions: list[tuple[object, str]] = []
        for item in add:
            if isinstance(item, tuple):
                additions.append(item)
            else:
                additions.append((item.source, item.destination))
        deletions = [item if isinstance(item, str) else item.path for item in delete]

        call = BatchCall(bucket_id=bucket_id, added={}, deleted=deletions)
        with self._lock:
            self.batch_calls.append(call)
            call_number = len(self.batch_calls)
            bucket_exists = bucket_id in self._buckets
        if self.fail_batch_on_call == call_number:
            call.failed = True
            raise ScriptedUploadError(f"scripted failure of upload call {call_number}")

        url = f"{endpoint}/api/buckets/{bucket_id}/batch"
        if not bucket_exists:
            call.failed = True
            self._raise_http_error(url, 404, "Repository not found")
        for _, destination in additions:
            if not _is_valid_destination(destination):
                call.failed = True
                self._raise_http_error(url, 422, "Invalid file path")

        # Read every source before the store changes: a real call is one
        # request, sent after all the data is uploaded.
        contents: dict[str, bytes] = {}
        for source, destination in additions:
            if isinstance(source, bytes):
                contents[destination] = source
            else:
                contents[destination] = Path(source).read_bytes()

        with self._lock:
            bucket = self._buckets[bucket_id]
            for destination, data in contents.items():
                if destination in self.drop_paths:
                    continue
                bucket[destination] = data
                self._blobs[_content_hash(data)] = data
                call.added[destination] = len(data)
            for path in deletions:
                bucket.pop(path, None)

    def _http_batch(self, bucket_id: str, body: bytes) -> _Reply:
        """``POST /api/buckets/{id}/batch``: NDJSON addFile / deleteFile."""
        if self.before_batch is not None:
            self.before_batch()
        operations = []
        for line in body.splitlines():
            if line.strip():
                operations.append(json.loads(line))

        call = BatchCall(bucket_id=bucket_id, added={}, deleted=[], via="http")
        failures = []
        with self._lock:
            self.batch_calls.append(call)
            if self.fail_batch_on_call == len(self.batch_calls):
                call.failed = True
                return _error_reply(403, "Forbidden", "scripted failure of a batch")
            bucket = self._buckets[bucket_id]
            for operation in operations:
                path = operation.get("path", "")
                kind = operation.get("type")
                if kind == "deleteFile":
                    bucket.pop(path, None)
                    call.deleted.append(path)
                    continue
                if kind != "addFile":
                    error = f"the fake bucket does not implement {kind!r}"
                elif not _is_valid_destination(path):
                    error = "Invalid file path"
                elif path in self.reject_paths:
                    error = "scripted rejection"
                elif operation.get("xetHash") not in self._blobs:
                    error = "the fake bucket has no content for this xetHash"
                else:
                    data = self._blobs[operation["xetHash"]]
                    bucket[path] = data
                    call.added[path] = len(data)
                    continue
                failures.append({"path": path, "error": error})

        call.failed = bool(failures)
        payload = {
            "success": not failures,
            "processed": len(operations),
            "succeeded": len(operations) - len(failures),
            "failed": failures,
        }
        all_failed = bool(operations) and len(failures) == len(operations)
        return _json_reply(payload, 422 if all_failed else 200)

    # ---- hub routes --------------------------------------------------------

    def _file_entry(self, path: str, data: bytes) -> dict:
        return {
            "type": "file",
            "path": path,
            "size": len(data),
            "xetHash": _content_hash(data),
            "mtime": _TIMESTAMP,
            "uploadedAt": _TIMESTAMP,
        }

    def _list_tree(self, files: dict[str, bytes], prefix: str, recursive: bool) -> list:
        if recursive:
            matched = sorted(p for p in files if p.startswith(prefix))
            return [self._file_entry(p, files[p]) for p in matched]

        directory = prefix.rstrip("/")
        is_directory = directory == "" or any(
            p.startswith(directory + "/") for p in files
        )
        if not is_directory:
            # Not a directory: list the parent, keep string-prefix matches.
            directory = prefix.rpartition("/")[0]
        base = directory + "/" if directory else ""

        entries: dict[str, dict] = {}
        for path in sorted(files):
            if not path.startswith(base):
                continue
            if not is_directory and not path.startswith(prefix):
                continue
            child, _, rest = path[len(base) :].partition("/")
            child_path = base + child
            if rest:
                entries.setdefault(
                    child_path,
                    {"type": "directory", "path": child_path, "uploadedAt": _TIMESTAMP},
                )
            else:
                entries[child_path] = self._file_entry(path, files[path])
        return list(entries.values())

    def _hub_reply(self, method: str, path: str, query: str, body: bytes) -> _Reply:
        parts = path.strip("/").split("/")

        if parts[:2] == ["api", "buckets"] and len(parts) >= 4:
            bucket_id = f"{parts[2]}/{parts[3]}"
            with self._lock:
                files = dict(self._buckets.get(bucket_id, {}))
                exists = bucket_id in self._buckets
            if not exists:
                return _error_reply(404, "RepoNotFound", "Repository not found")
            action = parts[4] if len(parts) > 4 else ""

            if action == "" and method == "GET":
                info = {
                    "id": bucket_id,
                    "private": True,
                    "createdAt": _TIMESTAMP,
                    "size": sum(len(data) for data in files.values()),
                    "totalFiles": len(files),
                }
                return _json_reply(info)

            if action == "batch" and method == "POST":
                return self._http_batch(bucket_id, body)

            if action == "tree" and method == "GET":
                # The client sends the prefix as ONE percent-encoded segment.
                prefix = unquote("/".join(parts[5:]))
                flag = parse_qs(query).get("recursive", ["false"])[0]
                recursive = flag.lower() in ("true", "1")
                return _json_reply(self._list_tree(files, prefix, recursive))

        if parts[:1] == ["buckets"] and len(parts) >= 5 and parts[3] == "resolve":
            bucket_id = f"{parts[1]}/{parts[2]}"
            file_path = unquote("/".join(parts[4:]))
            with self._lock:
                data = self._buckets.get(bucket_id, {}).get(file_path)
                exists = bucket_id in self._buckets
            if not exists:
                return _error_reply(404, "RepoNotFound", "Repository not found")
            if data is None:
                return _error_reply(404, "EntryNotFound", "File not found")
            digest = _content_hash(data)
            location = (
                f"{self.cdn_endpoint}/xet-bridge-us/{digest}"
                f"?X-Amz-Expires=3600&X-Amz-Signature={SIGNATURE}"
            )
            headers = {
                "Location": location,
                "X-Xet-Hash": digest,
                "X-Linked-Etag": f'"{digest}"',
                "X-Linked-Size": str(len(data)),
                "Content-Disposition": (
                    f'inline; filename="{posixpath.basename(file_path)}"'
                )
                .encode("ascii", "replace")
                .decode(),
            }
            return _Reply(302, headers)

        if parts[:1] == ["buckets"] and len(parts) >= 4 and parts[3] == "tree":
            # The web page of a bucket directory (what HfFileSystem.url()
            # returns for a directory). It uses cookie auth: a token gets 401.
            return _Reply(401, {"Content-Type": "text/html; charset=utf-8"}, b"<html>")

        return _error_reply(404, "NotFound", f"no fake route for {method} {path}")

    # ---- cdn routes --------------------------------------------------------

    def _cdn_reply(self, path: str, query: str, range_header: str | None) -> _Reply:
        parts = path.strip("/").split("/")
        signature = parse_qs(query).get("X-Amz-Signature", [""])[0]
        if signature != SIGNATURE:
            return _Reply(401, {"Content-Type": "text/plain"}, b"Unauthorized")
        if len(parts) != 2 or parts[0] != "xet-bridge-us":
            return _Reply(404, {"Content-Type": "text/plain"}, b"Not Found")

        with self._lock:
            data = self._blobs.get(parts[1])
        if data is None:
            return _Reply(404, {"Content-Type": "text/plain"}, b"Not Found")

        headers = {"Accept-Ranges": "bytes", "ETag": f'"{parts[1]}"'}
        if range_header is None:
            return _Reply(200, headers, data)

        match = _RANGE.fullmatch(range_header.strip())
        size = len(data)
        if match is None or (match.group(1) == "" and match.group(2) == ""):
            return _Reply(416, {"Content-Range": f"bytes */{size}"})
        if match.group(1) == "":
            # Suffix range: the last N bytes.
            start = max(size - int(match.group(2)), 0)
            end = size - 1
        else:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
            end = min(end, size - 1)
        if start > end:
            return _Reply(416, {"Content-Range": f"bytes */{size}"})
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return _Reply(206, headers, data[start : end + 1])

    # ---- shared request handling -------------------------------------------

    def _handle(self, origin: str, method: str, raw_path: str, headers, body: bytes):
        split = urlsplit(raw_path)
        decoded_path = unquote(split.path)
        range_header = headers.get("Range")
        authorization = headers.get("Authorization")
        with self._lock:
            accepted = {f"Bearer {token}" for token in self._tokens}

        fault = self._take_fault(origin, method, decoded_path)
        if fault is not None:
            reply = _Reply(fault.status, dict(fault.headers), fault.body)
        elif origin == HUB and authorization not in accepted:
            reply = _error_reply(401, "Unauthorized", "Invalid username or password.")
        elif origin == HUB:
            reply = self._hub_reply(method, split.path, split.query, body)
        elif method in ("GET", "HEAD"):
            reply = self._cdn_reply(split.path, split.query, range_header)
        else:
            reply = _Reply(405)

        sent = 0 if method == "HEAD" else len(reply.body)
        record = RecordedRequest(
            origin=origin,
            method=method,
            path=decoded_path,
            query=split.query,
            range=range_header,
            authorization=authorization,
            status=reply.status,
            body_bytes=sent,
        )
        with self._lock:
            self.requests.append(record)
        return reply


def _make_handler(hub: FakeHub, origin: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Send each reply at once (headers and body in one buffered write, no
        # Nagle delay); otherwise every request costs tens of milliseconds.
        wbufsize = -1
        disable_nagle_algorithm = True

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            reply = hub._handle(origin, self.command, self.path, self.headers, body)
            self.send_response(reply.status)
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(reply.body)
            self.wfile.flush()

        do_GET = _serve
        do_HEAD = _serve
        do_POST = _serve

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler
