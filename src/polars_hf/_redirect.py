"""A local HTTP server that redirects polars to presigned URLs.

``scan_bucket(resolve="redirect")`` gives polars URLs of this form::

    http://127.0.0.1:{port}/{scan id}/valid-only-while-pid-{pid}-runs/{bucket path}

The server of this module answers every request for such a URL with
``302 Location: <presigned URL>``. Polars follows the redirect and reads the
file from the CDN itself, with its native parquet scan. The query plan holds
the local URL only: no signature and no token.

One server runs per process, on the loopback interface and a port that the
operating system chooses. It is an ``asyncio`` server on its own event loop in
a daemon thread, and it starts with the first registration. The presigned URLs
are resolved by a function that the caller registers (``read.py`` passes one
that asks the Hub); the calls run in a thread pool of this module.

What the server answers:

* a registered file: ``302`` with the presigned URL, for ``GET`` and ``HEAD``.
  The URL of a file is resolved by one call also when many requests for it
  arrive together, is kept, and is resolved again when it is older than
  ``_URL_MAX_AGE``;
* everything else: a ``4xx`` answer. Polars retries a ``5xx`` answer, a
  ``408``, a ``429`` and a dropped connection ten times (5 to 30 s), and does
  not retry another ``4xx``. So only a transient failure of the resolver is
  answered with ``503``. Polars prints the body of an error answer to a
  ``GET`` request and has none for ``HEAD``, so the reason of a failed
  resolve is also logged (logger ``polars_hf._redirect``, level WARNING).

The scan id is random. It is a capability: who knows a local URL can get the
presigned URL of that one file while the process lives. The server listens on
``127.0.0.1`` only, serves only the exact file paths of a registration, and
refuses a request whose ``Host`` header is not the loopback address (a web
page cannot reach it through DNS rebinding).

A local URL works only while the process that registered it runs, and only
on its machine. Polars runs the plan natively, so this module gets no control
when a plan is used elsewhere (a serialized plan, another machine, a child
process after the parent has exited): polars then fails to connect, retries
for several seconds and raises an error that names the local URL. The
``valid-only-while-pid-...-runs`` segment of the URL is there for that
message. In the child of a fork the inherited listening socket is closed and
a new server is started for new scans.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import quote, unquote_to_bytes

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"
# Pending connections that the kernel keeps for the server. Polars opens many
# connections at once; with a small backlog they are dropped and retried
# after seconds.
_BACKLOG = 4096
# Longest request head (request line and headers) that is read.
_MAX_REQUEST_BYTES = 16 * 1024
_MAX_RESOLVE_WORKERS = 32
# A presigned URL is valid for about 60 minutes. A URL older than this many
# seconds is not served: the file is resolved again.
_URL_MAX_AGE = 1800.0
# After a failure the resolver is not called again for the same file for this
# many seconds; the requests in between get the same answer.
_FAILURE_HOLD = 5.0
# Registrations kept; the one that was used least recently is dropped first.
_MAX_SCANS = 1024

_REASONS = {
    302: "Found",
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    424: "Failed Dependency",
    503: "Service Unavailable",
}
_UNKNOWN_SCAN = (
    "unknown scan. A scan_bucket LazyFrame of the redirect mode is valid only "
    "in the process that made it, and while it is one of the "
    f"{_MAX_SCANS} scans used last. Call scan_bucket again, or use "
    'resolve="collect" for a plan that other processes can run.'
)


class _Refusal(Exception):
    """The server answers a request with ``status`` and ``message``."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class _Scan:
    """One registration: the files of one ``scan_bucket`` call.

    ``resolve(path)`` returns the presigned URL of a file. ``describe(error)``
    turns an exception of ``resolve`` into the status and the text of the
    answer. The dictionaries are used by the event loop thread only.
    """

    scan_id: str
    marker: str
    paths: frozenset[str]
    resolve: Callable[[str], str]
    describe: Callable[[Exception], tuple[int, str]]
    # path -> (presigned URL, time it was resolved)
    urls: dict[str, tuple[str, float]] = field(default_factory=dict)
    # path -> (refusal, time until which it is repeated)
    failures: dict[str, tuple[_Refusal, float]] = field(default_factory=dict)
    # path -> the task that resolves it now
    resolving: dict[str, asyncio.Task] = field(default_factory=dict)


def _has_control_character(text: str) -> bool:
    for character in text:
        if ord(character) < 32 or ord(character) == 127:
            return True
    return False


def _is_safe_location(url: str) -> bool:
    """Whether ``url`` can be written into a ``Location`` header."""
    if not url.startswith(("http://", "https://")):
        return False
    for character in url:
        if ord(character) <= 32 or ord(character) >= 127:
            return False
    return True


def _response(status: int, *, location: str | None = None, body: str = "") -> bytes:
    """The bytes of one HTTP/1.1 answer. A redirect has no body."""
    payload = body.encode("utf-8")
    lines = [f"HTTP/1.1 {status} {_REASONS[status]}"]
    if location is not None:
        lines.append(f"Location: {location}")
    if payload:
        lines.append("Content-Type: text/plain; charset=utf-8")
    if status == 405:
        lines.append("Allow: GET, HEAD")
    lines.append(f"Content-Length: {len(payload)}")
    head = "\r\n".join(lines) + "\r\n\r\n"
    return head.encode("latin-1") + payload


class _RedirectServer:
    """The redirect server of this process. Use :func:`register`."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.port = 0
        self.requests = 0
        self._scans: OrderedDict[str, _Scan] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=_MAX_RESOLVE_WORKERS, thread_name_prefix="polars-hf-resolve"
        )
        self._loop = asyncio.new_event_loop()
        self._server: asyncio.AbstractServer | None = None
        self._listening: list[int] = []
        self._stopped = False
        self._start_error: BaseException | None = None
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(ready,), name="polars-hf-redirect", daemon=True
        )
        self._thread.start()
        ready.wait()
        if self._start_error is not None:
            raise RuntimeError(
                "polars-hf could not start its local redirect server on "
                f'{_HOST}: {self._start_error}. Use resolve="collect".'
            )

    # ---- lifecycle ---------------------------------------------------------

    def _run(self, ready: threading.Event) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._server = self._loop.run_until_complete(
                asyncio.start_server(
                    self._handle,
                    _HOST,
                    0,
                    backlog=_BACKLOG,
                    limit=_MAX_REQUEST_BYTES,
                )
            )
            self.port = self._server.sockets[0].getsockname()[1]
            self._listening = [sock.fileno() for sock in self._server.sockets]
        except BaseException as error:
            self._start_error = error
            ready.set()
            return
        ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _shutdown(self) -> None:
        if self._server is not None:
            self._server.close()
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._loop.stop()

    def stop(self) -> None:
        """Stop the server and wait for its thread. Safe to call twice."""
        if self._stopped:
            return
        self._stopped = True
        if self._thread.is_alive() and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            except RuntimeError:
                # The loop was closed in between.
                pass
            self._thread.join(timeout=5)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def close_inherited_sockets(self) -> None:
        """In a forked child: close the listening sockets of the parent.

        The child has no server thread. With the sockets open in the child,
        connections would be accepted by the kernel and never answered once
        the parent is gone.
        """
        for descriptor in self._listening:
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._listening = []

    # ---- registrations -----------------------------------------------------

    def register(
        self,
        paths: list[str],
        resolve: Callable[[str], str],
        describe: Callable[[Exception], tuple[int, str]],
        *,
        first_url: str | None = None,
    ) -> list[str]:
        """Register the files of one scan and return their local URLs.

        Every call makes a new scan id with no presigned URL, so a new
        ``scan_bucket`` call never reads through a URL of an earlier one.
        ``first_url`` is the presigned URL of ``paths[0]``, if the caller
        has just resolved it.
        """
        scan = _Scan(
            scan_id=secrets.token_urlsafe(16),
            marker=f"valid-only-while-pid-{self.pid}-runs",
            paths=frozenset(paths),
            resolve=resolve,
            describe=describe,
        )
        if first_url is not None:
            # Before the scan is published: no request can see it yet.
            scan.urls[paths[0]] = (first_url, time.monotonic())
        with self._lock:
            self._scans[scan.scan_id] = scan
            while len(self._scans) > _MAX_SCANS:
                self._scans.popitem(last=False)

        base = f"http://{_HOST}:{self.port}/{scan.scan_id}/{scan.marker}/"
        # "=" stays as it is: hive partition directories are "key=value".
        return [base + quote(path, safe="/=") for path in paths]

    def registered_scans(self) -> int:
        with self._lock:
            return len(self._scans)

    def _find_scan(self, scan_id: str) -> _Scan | None:
        with self._lock:
            scan = self._scans.get(scan_id)
            if scan is not None:
                self._scans.move_to_end(scan_id)
            return scan

    def forget_urls(self) -> None:
        """Drop every presigned URL and held failure (for tests)."""

        def clear() -> None:
            with self._lock:
                scans = list(self._scans.values())
            for scan in scans:
                scan.urls.clear()
                scan.failures.clear()

        done = threading.Event()

        def run() -> None:
            clear()
            done.set()

        self._loop.call_soon_threadsafe(run)
        done.wait(timeout=5)

    # ---- requests ----------------------------------------------------------

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Answer the requests of one connection until it is closed."""
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError:
                    # The client closed the connection.
                    return
                except (asyncio.LimitOverrunError, ValueError):
                    writer.write(_response(400, body="polars-hf: request too large"))
                    await writer.drain()
                    return
                self.requests += 1
                answer, keep_open = await self._answer(head)
                writer.write(answer)
                await writer.drain()
                if not keep_open:
                    return
        except (
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
            asyncio.IncompleteReadError,
        ):
            pass
        except Exception:
            # Nothing of a failed request may reach stderr.
            logger.debug("redirect server: a request failed", exc_info=True)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _answer(self, head: bytes) -> tuple[bytes, bool]:
        """The answer to one request head, and whether the connection stays open.

        Every failure becomes an HTTP answer here: no exception leaves.
        """
        is_head = head.startswith(b"HEAD ")
        try:
            _, path, scan = self._parse(head)
            location = await self._url(scan, path)
            if not _is_safe_location(location):
                raise _Refusal(424, "the Hub gave an invalid presigned URL")
            return _response(302, location=location), True
        except _Refusal as refusal:
            body = ""
            if not is_head:
                body = f"polars-hf: {refusal.message}"
            # After a request that was not understood, the rest of the
            # stream cannot be trusted.
            keep_open = refusal.status not in (400, 405)
            return _response(refusal.status, body=body), keep_open
        except Exception:
            logger.debug("redirect server: unexpected error", exc_info=True)
            return _response(400, body="polars-hf: the request failed"), False

    def _parse(self, head: bytes) -> tuple[str, str, _Scan]:
        """The method, the bucket path and the scan of a request head."""
        lines = head.decode("latin-1").split("\r\n")
        request_line = lines[0].split(" ")
        if len(request_line) != 3 or not request_line[2].startswith("HTTP/1."):
            raise _Refusal(400, "malformed request line")
        method, target, _ = request_line
        if method not in ("GET", "HEAD"):
            raise _Refusal(405, "only GET and HEAD are answered")

        host = None
        for line in lines[1:]:
            name, _, value = line.partition(":")
            if name.strip().lower() == "host":
                host = value.strip().lower()
        if host not in (f"{_HOST}:{self.port}", f"localhost:{self.port}"):
            # A browser that was sent here through DNS rebinding sends the
            # host name of the web page.
            raise _Refusal(403, "the Host header is not the loopback address")

        if not target.startswith("/"):
            raise _Refusal(400, "malformed request target")
        raw_path = target.partition("?")[0]
        try:
            # Decoded once: "%252F" is the three characters "%2F" of a name.
            decoded = unquote_to_bytes(raw_path).decode("utf-8")
        except UnicodeDecodeError:
            raise _Refusal(400, "the path is not UTF-8") from None
        if _has_control_character(decoded):
            raise _Refusal(400, "the path has a control character")

        scan_id, _, rest = decoded[1:].partition("/")
        marker, _, path = rest.partition("/")
        scan = self._find_scan(scan_id)
        if scan is None:
            raise _Refusal(404, _UNKNOWN_SCAN)
        if marker != scan.marker or path not in scan.paths:
            # The path is not repeated in the answer.
            raise _Refusal(404, "not a file of this scan")
        return method, path, scan

    async def _url(self, scan: _Scan, path: str) -> str:
        """The presigned URL of ``path``: kept, or resolved by one call."""
        now = time.monotonic()
        kept = scan.urls.get(path)
        if kept is not None and now - kept[1] <= _URL_MAX_AGE:
            return kept[0]
        held = scan.failures.get(path)
        if held is not None and now < held[1]:
            raise held[0]

        task = scan.resolving.get(path)
        if task is None:
            task = asyncio.ensure_future(self._resolve(scan, path))
            scan.resolving[path] = task
            # A task that nobody waits for any more must not log its error.
            task.add_done_callback(_ignore_result)
        # shield: a client that goes away does not cancel the resolve that
        # other requests wait for.
        return await asyncio.shield(task)

    async def _resolve(self, scan: _Scan, path: str) -> str:
        loop = asyncio.get_running_loop()
        try:
            url = await loop.run_in_executor(self._executor, scan.resolve, path)
        except Exception as error:
            try:
                status, message = scan.describe(error)
            except Exception:
                status, message = 424, "the presigned URL could not be resolved"
            if status not in _REASONS or status == 302:
                status = 424
            refusal = _Refusal(status, message)
            scan.failures[path] = (refusal, time.monotonic() + _FAILURE_HOLD)
            logger.warning(
                "polars-hf: the local redirect server answers HTTP %d: %s",
                status,
                message,
            )
            raise refusal from None
        finally:
            scan.resolving.pop(path, None)
        scan.failures.pop(path, None)
        scan.urls[path] = (url, time.monotonic())
        return url


def _ignore_result(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


# ---- the server of the process ---------------------------------------------

_server: _RedirectServer | None = None
_server_lock = threading.Lock()
_exit_hook_registered = False


def _stop_at_exit() -> None:
    server = _server
    if server is not None and server.pid == os.getpid():
        server.stop()


def _after_fork_in_child() -> None:
    """The child of a fork has no server thread: start again when needed."""
    global _server, _server_lock
    inherited = _server
    _server = None
    _server_lock = threading.Lock()
    if inherited is not None:
        inherited.close_inherited_sockets()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def get_server() -> _RedirectServer:
    """The redirect server of this process; started by the first call."""
    global _server, _exit_hook_registered
    with _server_lock:
        if _server is not None and _server.pid != os.getpid():
            # A fork without the at-fork hook.
            _server = None
        if _server is None:
            _server = _RedirectServer()
            if not _exit_hook_registered:
                atexit.register(_stop_at_exit)
                _exit_hook_registered = True
        return _server


def register(
    paths: list[str],
    resolve: Callable[[str], str],
    describe: Callable[[Exception], tuple[int, str]],
    *,
    first_url: str | None = None,
) -> list[str]:
    """Register the files of one scan with the server of this process.

    Returns the local URL of every path, in the order of ``paths``. See
    :meth:`_RedirectServer.register`.
    """
    return get_server().register(paths, resolve, describe, first_url=first_url)
