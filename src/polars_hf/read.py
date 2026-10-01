"""Scan parquet from Hugging Face buckets as a native polars ``LazyFrame``.

Bucket files are XET-backed: the Hub ``resolve`` URL 302-redirects (when
requested with auth) to a presigned CDN URL (``us.aws.cdn.hf.co/xet-bridge-*``
on huggingface.co) that needs no auth and supports HTTP range requests. We
follow that redirect in Python and hand the **signed URLs** to native
:func:`polars.scan_parquet`, so polars' Rust object store does async,
concurrent, range-read scans with full projection / predicate / slice pushdown
— the same mechanism the upstream ``hf://`` reader uses, but reachable from
stock polars.

Stock polars cannot authenticate a generic ``https://`` URL itself (bearer-token
injection is gated behind the ``hf://`` scheme), which is why we resolve the
signed URL here rather than passing the ``resolve`` URL directly.

All Hub requests (the bucket listing and the ``resolve`` requests) are sent
with the shared ``huggingface_hub`` session
(``huggingface_hub.utils.get_session``: proxies, offline mode, custom client
factory). Retries of 408, 429 and 5xx answers are done here, for every request
and every listing page, and no wait ends after the deadline of the
``scan_bucket`` call. ``HfApi.list_bucket_tree`` is not used: its retries are
not bounded and differ between huggingface_hub versions.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote, urlencode, urljoin, urlparse

import polars as pl
from huggingface_hub import constants
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import (
    build_hf_headers,
    get_session,
    hf_raise_for_status,
    parse_ratelimit_headers,
)

from polars_hf._glob import glob_to_regex, has_glob, literal_prefix
from polars_hf._uri import BucketPath, parse_bucket_uri

logger = logging.getLogger(__name__)

# The signed URL carries its own expiry (an ``Expires`` query parameter, about
# one hour after the resolve request), so URLs are resolved at scan time.
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_MAX_RESOLVE_WORKERS = 16
_MAX_REDIRECT_HOPS = 5
# Entries per listing page (the ``limit`` query parameter). ``None`` leaves
# the page size to the Hub; the staging tests set it to force several pages.
_LIST_PAGE_LIMIT: int | None = None

# Keep in sync with the parquet extensions accepted by sink_bucket
# (write._EXT_FORMAT, which matches case-insensitively).
_PARQUET_SUFFIXES = (".parquet", ".pq")

# Retry policy of the Hub requests (listing and resolve). Timeouts and
# connection errors are not retried.
_REQUEST_TIMEOUT = 30
_RETRY_STATUS_CODES = (408, 429, 500, 502, 503, 504)
_MAX_RETRIES = 5
# Without a hint from the server the wait doubles from the base up to the
# maximum backoff.
_RETRY_BASE_WAIT = 1.0
_RETRY_MAX_BACKOFF = 8.0
# The server can ask for a longer wait (rate-limit reset, Retry-After). A wait
# is made only if it ends before the deadline of the scan_bucket call; the
# scan fails instead of sleeping past it.
_SCAN_DEADLINE = 600.0
# A wait that the server asks for and that is longer than this is announced
# with a warning, so that a paused scan is not silent.
_WARN_WAIT = 5.0


def _is_parquet_name(path: str) -> bool:
    return path.lower().endswith(_PARQUET_SUFFIXES)


def _file_uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _status_code(error: HfHubHTTPError) -> int | None:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def _no_access_error(bucket_id: str, uri: str, status: int) -> PermissionError:
    return PermissionError(
        f"cannot read {uri!r}: the Hub refused the request (HTTP {status}); "
        f"the token is not valid or lacks access to the bucket {bucket_id!r}"
    )


def _empty_file_error(uri: str) -> ValueError:
    return ValueError(f"{uri!r} is empty (0 bytes): not a parquet file")


# ---- bounded retries -------------------------------------------------------


class _Budget:
    """Retry limits and progress of one ``scan_bucket`` call.

    One instance is shared by the listing and by all resolve threads, so the
    deadline applies to the call as a whole. ``operation`` names the public
    function in messages; the write path passes ``"sink_bucket"``.
    """

    def __init__(self, bucket_id: str, operation: str = "scan_bucket") -> None:
        self.bucket_id = bucket_id
        self.operation = operation
        self.deadline = time.monotonic() + _SCAN_DEADLINE
        self.files_total = 0
        self._files_resolved = 0
        self._wait_announced = False
        self._lock = threading.Lock()

    def file_resolved(self) -> None:
        with self._lock:
            self._files_resolved += 1

    def announce_wait(self, message: str) -> None:
        """Warn about a long wait, once per ``scan_bucket`` call.

        All resolve threads usually wait for the same rate-limit reset: one
        warning is enough. Under a warnings-as-errors filter ``warnings.warn``
        raises; the message is then logged instead, so that the filter cannot
        abort the scan from inside a retry.
        """
        with self._lock:
            if self._wait_announced:
                return
            self._wait_announced = True
        try:
            warnings.warn(message, stacklevel=4)
        except Warning:
            logger.warning(message)

    def progress(self) -> str:
        """``"; 3 of 20 files were resolved"``, or ``""`` before the resolves."""
        if self.files_total == 0:
            return ""
        with self._lock:
            resolved = self._files_resolved
        return f"; {resolved} of {self.files_total} files were resolved"


def _finite_seconds(value: object) -> float | None:
    """``value`` as a number of seconds, or ``None`` if it is not a finite one."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return max(seconds, 0.0)


def _server_wait_hint(response) -> float | None:
    """Seconds the server asks to wait before the next attempt, if it says so.

    A hint that is not a finite number of seconds (``nan``, an HTTP date) is
    no hint: the caller then uses the backoff.
    """
    if response.status_code == 429:
        info = parse_ratelimit_headers(response.headers)
        if info is not None and info.remaining == 0:
            reset = _finite_seconds(info.reset_in_seconds)
            if reset is not None:
                # One more second: the reset time is rounded down.
                return reset + 1
    return _finite_seconds(response.headers.get("retry-after"))


def _retry_error(response, budget: _Budget, what: str, reason: str) -> HfHubHTTPError:
    """The error raised when a request is not retried (again)."""
    status = response.status_code
    if status == 429:
        quota = ""
        info = parse_ratelimit_headers(response.headers)
        if info is not None and info.limit is not None:
            quota = f" ({info.limit} requests per {info.window_seconds} s)"
        problem = f"the Hub rate limit for {what} requests was reached{quota}"
    else:
        problem = f"the Hub answered HTTP {status} to a {what} request"
    message = (
        f"{problem} while reading the bucket {budget.bucket_id!r}"
        f"{budget.progress()}: {reason}"
    )
    return HfHubHTTPError(message, response=response)


def _wait_before_retry(response, attempt: int, budget: _Budget, what: str) -> None:
    """Sleep before retry number ``attempt + 1``, or raise if it is not allowed.

    ``response`` is the 408 / 429 / 5xx answer of the attempt that failed.
    """
    if attempt >= _MAX_RETRIES:
        reason = f"no success after {_MAX_RETRIES} retries"
        raise _retry_error(response, budget, what, reason)

    hint = _server_wait_hint(response)
    wait = hint
    if wait is None:
        wait = min(_RETRY_MAX_BACKOFF, _RETRY_BASE_WAIT * 2**attempt)
    left = max(budget.deadline - time.monotonic(), 0.0)
    if wait > left:
        asked = "the Hub asks to wait" if hint is not None else "the next retry is in"
        reason = (
            f"{asked} {wait:.0f} s, but only {left:.0f} s are left of the "
            f"{_SCAN_DEADLINE:.0f} s allowed for one {budget.operation} call; "
            "try again later"
        )
        raise _retry_error(response, budget, what, reason)
    if hint is not None and wait > _WARN_WAIT:
        cause = "rate limit" if response.status_code == 429 else "busy server"
        budget.announce_wait(
            f"{budget.operation}: Hub {cause} (HTTP {response.status_code}) on a "
            f"{what} request for the bucket {budget.bucket_id!r}; waiting {wait:.0f} s "
            "before the next attempt"
        )
    time.sleep(wait)


# ---- requests --------------------------------------------------------------

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str | None, int | None]:
    """Scheme, host and port of ``url``; the default port is made explicit.

    ``urlparse`` lower-cases the scheme and the host, so their case is ignored.
    """
    parsed = urlparse(url)
    port = parsed.port
    if port is None:
        port = _DEFAULT_PORTS.get(parsed.scheme)
    return (parsed.scheme, parsed.hostname, port)


def _request(
    method: str,
    url: str,
    headers: dict[str, str],
    budget: _Budget,
    what: str,
):
    """One request with auth that does not follow redirects.

    The request is sent with the shared ``huggingface_hub`` session. A 408 /
    429 / 5xx answer is retried within the limits of ``budget``; when they are
    reached, ``HfHubHTTPError`` is raised. Any other answer is returned.

    Timeouts and connection errors are not retried and keep their type.
    """
    attempt = 0
    while True:
        response = get_session().request(
            method,
            url,
            headers=headers,
            follow_redirects=False,
            timeout=_REQUEST_TIMEOUT,
        )
        if response.status_code not in _RETRY_STATUS_CODES:
            return response
        _wait_before_retry(response, attempt, budget, what)
        attempt += 1


# ---- listing ---------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    """One entry of a bucket listing (the fields of the Hub's JSON we use)."""

    type: str
    path: str
    size: int | None = None
    xet_hash: str | None = None


# One value of a Link header: <target> followed by its ;name=value parameters.
# A quoted parameter value can hold commas and semicolons.
_LINK_VALUE = re.compile(r'<([^>]*)>((?:\s*;\s*[^=;,\s]+\s*=\s*(?:"[^"]*"|[^;,]*))*)')
_LINK_REL = re.compile(r';\s*rel\s*=\s*(?:"([^"]*)"|([^;,\s]*))', re.IGNORECASE)


def _next_page_url(response, page_url: str) -> str | None:
    """The target of the ``rel="next"`` link of a listing page, if any.

    The ``Link`` header is parsed here: ``rel`` is matched case-insensitively
    and can hold several relation types (``rel="prev next"``). A relative
    target is resolved against the URL of the page.
    """
    header = response.headers.get("link")
    if header is None:
        return None
    for value in _LINK_VALUE.finditer(header):
        rel = _LINK_REL.search(value.group(2))
        if rel is None:
            continue
        relations = (rel.group(1) or rel.group(2) or "").lower().split()
        if "next" in relations:
            return urljoin(page_url, value.group(1).strip())
    return None


def _entries_of_page(response, page_url: str) -> list[_Entry]:
    """The entries of one listing page; ``RuntimeError`` for another shape."""
    problem = None
    try:
        body = response.json()
    except ValueError:
        body = None
        problem = "the body is not JSON"
    if problem is None and not isinstance(body, list):
        problem = "the body is not a JSON list"

    entries = []
    if problem is None:
        for item in body:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("type"), str)
                or not isinstance(item.get("path"), str)
            ):
                problem = "an entry has no 'type' or no 'path'"
                break
            entries.append(
                _Entry(
                    type=item["type"],
                    path=item["path"],
                    size=item.get("size"),
                    xet_hash=item.get("xetHash"),
                )
            )
    if problem is not None:
        raise RuntimeError(
            f"unexpected answer of the Hub listing endpoint {page_url!r}: {problem}"
        )
    return entries


def _list_tree(
    endpoint: str,
    headers: dict[str, str],
    bucket_id: str,
    prefix: str,
    *,
    recursive: bool,
    uri: str,
    budget: _Budget,
) -> list[_Entry]:
    """All entries of a bucket listing (every page of :func:`_iter_tree_pages`)."""
    entries: list[_Entry] = []
    pages = _iter_tree_pages(
        endpoint,
        headers,
        bucket_id,
        prefix,
        recursive=recursive,
        uri=uri,
        budget=budget,
    )
    for page in pages:
        entries.extend(page)
    return entries


def _iter_tree_pages(
    endpoint: str,
    headers: dict[str, str],
    bucket_id: str,
    prefix: str,
    *,
    recursive: bool,
    uri: str,
    budget: _Budget,
) -> Iterator[list[_Entry]]:
    """The entries of a bucket listing, one page per request.

    A page is requested only when the caller asks for it, so a caller that
    stops after the first page sends one request.

    The request is the one of ``HfApi.list_bucket_tree``:
    ``GET /api/buckets/{id}/tree/{prefix}?recursive=...``, with the prefix as
    one percent-encoded segment. The answer is a JSON list; a
    ``Link: <url>; rel="next"`` header names the next page. Every page is
    requested with the retry limits of ``budget``.

    Raises ``RuntimeError`` when the next link leaves the Hub origin (the
    token is sent with every page), repeats a page, or when a page has an
    unexpected shape; ``TimeoutError`` when the deadline of ``budget`` passes
    between two pages.
    """
    url = f"{endpoint}/api/buckets/{bucket_id}/tree"
    if prefix:
        url = f"{url}/{quote(prefix, safe='')}"
    query = {"recursive": "true" if recursive else "false"}
    if _LIST_PAGE_LIMIT is not None:
        query["limit"] = str(_LIST_PAGE_LIMIT)
    # The complete URL of every page is kept, to detect a link that repeats.
    url = f"{url}?{urlencode(query)}"
    hub_origin = _origin(endpoint)

    listed = 0
    requested: set[str] = set()
    while url is not None:
        if time.monotonic() > budget.deadline:
            raise TimeoutError(
                f"the listing of {uri!r} did not finish within the "
                f"{_SCAN_DEADLINE:.0f} s allowed for one {budget.operation} call "
                f"({listed} entries were listed)"
            )
        requested.add(url)
        response = _request("GET", url, headers, budget, "listing")
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as error:
            status = _status_code(error)
            if status in (401, 403):
                raise _no_access_error(bucket_id, uri, status) from error
            if status == 404:
                raise FileNotFoundError(
                    f"bucket {bucket_id!r} not found (or the token has no access "
                    f"to it): {uri!r}"
                ) from error
            raise
        if response.status_code != 200:
            raise RuntimeError(
                f"unexpected answer of the Hub listing endpoint {url!r} "
                f"(HTTP {response.status_code})"
            )
        page = _entries_of_page(response, url)
        listed += len(page)

        # The link of the next page already holds the query parameters.
        next_url = _next_page_url(response, url)
        if next_url is not None:
            try:
                same_origin = _origin(next_url) == hub_origin
            except ValueError:
                # urlparse refuses the port of the link.
                raise RuntimeError(
                    f"the Hub listing of {uri!r} has an invalid next link: {next_url!r}"
                ) from None
            if not same_origin:
                # The token is sent with every page: never to another origin.
                raise RuntimeError(
                    f"the Hub listing of {uri!r} links to another origin: {next_url!r}"
                )
            if next_url in requested:
                raise RuntimeError(
                    f"the Hub listing of {uri!r} links back to a page that was "
                    f"already read: {next_url!r}"
                )
        yield page
        url = next_url


def _files(entries: list) -> list:
    files = []
    for entry in entries:
        if entry.type == "file":
            files.append(entry)
    return files


def _exact_file(files: list, path: str) -> list:
    for file in files:
        if file.path == path:
            return [file]
    return []


def _glob_is_in_one_directory(pattern: str) -> bool:
    """Whether only the last segment of ``pattern`` is a glob, without ``**``."""
    segments = pattern.split("/")
    for segment in segments[:-1]:
        if has_glob(segment):
            return False
    return "**" not in segments[-1]


def _directory_files(
    endpoint: str,
    headers: dict[str, str],
    bp: BucketPath,
    directory: str,
    uri: str,
    budget: _Budget,
) -> list[_Entry]:
    """The parquet files below ``directory`` (``""`` is the whole bucket)."""
    # The Hub matches a listing prefix as a plain string: 'train' would also
    # list 'train_full/...'. With the trailing slash the listing holds the
    # files of this directory only.
    prefix = f"{directory}/" if directory else ""
    entries = _list_tree(
        endpoint, headers, bp.bucket_id, prefix, recursive=True, uri=uri, budget=budget
    )
    selected = []
    for file in _files(entries):
        if file.path.startswith(prefix) and _is_parquet_name(file.path):
            selected.append(file)
    return selected


def _glob_files(
    endpoint: str,
    headers: dict[str, str],
    bp: BucketPath,
    pattern: str,
    uri: str,
    budget: _Budget,
) -> list[_Entry]:
    """The files that ``pattern`` names: an exact file name, else the matches.

    A file whose name is exactly ``pattern`` (``data[1].parquet``) is read
    literally. Raises ``ValueError`` for an invalid pattern before any request.
    """
    matcher = glob_to_regex(pattern)
    if _glob_is_in_one_directory(pattern):
        # A glob in the last segment only: list its directory, not the subtree.
        parent = pattern.rpartition("/")[0]
        entries = _list_tree(
            endpoint,
            headers,
            bp.bucket_id,
            parent,
            recursive=False,
            uri=uri,
            budget=budget,
        )
    else:
        # Every match starts with the text before the first glob character.
        entries = _list_tree(
            endpoint,
            headers,
            bp.bucket_id,
            literal_prefix(pattern),
            recursive=True,
            uri=uri,
            budget=budget,
        )
    files = _files(entries)

    selected = _exact_file(files, pattern)
    if selected:
        return selected
    for file in files:
        if matcher.fullmatch(file.path):
            selected.append(file)
    return selected


def _list_files(
    endpoint: str, headers: dict[str, str], bp: BucketPath, uri: str, budget: _Budget
) -> list[_Entry]:
    """List the bucket and return the files that ``bp`` names, sorted by path.

    A path with a glob character is a glob (see ``_glob_files``); any other
    path is a directory. The entries hold ``path``, ``size`` and ``xet_hash``.
    """
    path = bp.path.rstrip("/")
    if bp.is_glob and bp.path.endswith("/"):
        raise ValueError(
            f"a glob cannot end with '/': {uri!r} (a glob selects files; use "
            f"'{path}/*.parquet' for the files of the matching directories or "
            f"'{path}/**/*.parquet' for all files below them)"
        )

    if bp.is_glob:
        selected = _glob_files(endpoint, headers, bp, path, uri, budget)
    else:
        selected = _directory_files(endpoint, headers, bp, path, uri, budget)
    if not selected:
        hint = ""
        if bp.is_glob:
            hint = (
                " (the path is read as a glob; to match a literal '[', '*' or "
                "'?' of a file or directory name, write '[[]', '[*]' or '[?]')"
            )
        raise FileNotFoundError(f"no parquet files matched: {uri!r}{hint}")

    for file in selected:
        if file.size == 0:
            raise _empty_file_error(_file_uri(bp.bucket_id, file.path))
    return sorted(selected, key=lambda file: file.path)


# ---- resolve ---------------------------------------------------------------


def _resolve_url(endpoint: str, bucket_id: str, path: str) -> str:
    """The Hub ``resolve`` URL of a bucket file (the one ``HfApi`` requests)."""
    return f"{endpoint}/buckets/{bucket_id}/resolve/{quote(path, safe='')}"


def _signed_url(
    resolve_url: str, headers: dict[str, str], *, uri: str, budget: _Budget
) -> str:
    """Follow the authenticated resolve redirect to a range-readable signed URL.

    Uses HEAD — the same way ``HfApi.get_bucket_file_metadata`` probes this
    endpoint — so no file bytes are transferred. Redirects that stay on the
    Hub origin (relative, or absolute with the same scheme, host and port) are
    followed with auth; the first ``location`` on another origin is the
    presigned CDN URL, readable without auth. The auth header is never sent to
    another origin — including a scheme downgrade on the same host.

    ``uri`` is used in error messages only.

    Raises
    ------
    FileNotFoundError
        The Hub answers 404 for the file.
    PermissionError
        The Hub answers 401 or 403.
    ValueError
        The file is empty (0 bytes).
    RuntimeError
        The Hub serves the file itself instead of redirecting (the URL would
        need auth, which polars cannot send), or redirects too many times.
    huggingface_hub.errors.HfHubHTTPError
        Any other HTTP error, and a 408 / 429 / 5xx answer that the retries
        allowed by ``budget`` did not clear.
    """
    hub_origin = _origin(resolve_url)
    url = resolve_url
    for _ in range(_MAX_REDIRECT_HOPS):
        response = _request("HEAD", url, headers, budget, "resolve")
        if response.status_code in _REDIRECT_CODES and "location" in response.headers:
            # urljoin resolves relative *and* protocol-relative (//host/..)
            # locations; compare origins rather than sniffing the scheme prefix.
            location = urljoin(url, response.headers["location"])
            try:
                same_origin = _origin(location) == hub_origin
            except ValueError:
                # urlparse refuses the port of the location.
                raise RuntimeError(
                    f"the Hub redirected {uri!r} to an invalid URL: {location!r}"
                ) from None
            if not same_origin:
                if response.headers.get("x-linked-size") == "0":
                    raise _empty_file_error(uri)
                return location
            url = location
            continue

        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as error:
            status = _status_code(error)
            if status in (401, 403):
                raise _no_access_error(budget.bucket_id, uri, status) from error
            if status == 404:
                raise FileNotFoundError(f"no such file: {uri!r}") from error
            raise

        # No redirect and no error: the Hub serves the bytes itself. It does
        # so for an empty file and for a file that is not Xet-backed.
        size = response.headers.get("x-linked-size")
        if size is None:
            size = response.headers.get("content-length")
        if size == "0":
            raise _empty_file_error(uri)
        raise RuntimeError(
            f"the Hub did not redirect {uri!r} to a presigned URL (HTTP "
            f"{response.status_code}); polars cannot read a URL that needs the "
            "token"
        )
    raise RuntimeError(f"too many redirects while resolving {uri!r}")


def _signed_urls(
    endpoint: str, headers: dict[str, str], paths: list[str], budget: _Budget
) -> list[str]:
    """Resolve every path to its signed URL, one HEAD request per file."""
    bucket_id = budget.bucket_id
    budget.files_total = len(paths)

    def resolve(path: str) -> str:
        url = _signed_url(
            _resolve_url(endpoint, bucket_id, path),
            headers,
            uri=_file_uri(bucket_id, path),
            budget=budget,
        )
        budget.file_resolved()
        return url

    if len(paths) == 1:
        return [resolve(paths[0])]
    with ThreadPoolExecutor(max_workers=min(_MAX_RESOLVE_WORKERS, len(paths))) as pool:
        return list(pool.map(resolve, paths))


def scan_bucket(
    uri: str, *, token: str | None = None, **scan_kwargs: object
) -> pl.LazyFrame:
    """Lazily scan parquet file(s) from a Hugging Face bucket.

    Returns a native polars ``LazyFrame`` (via :func:`polars.scan_parquet` over
    presigned URLs), so projection, predicate, and slice pushdown, streaming, and
    multi-file concurrency all work natively — only the column chunks actually
    needed are transferred.

    Parameters
    ----------
    uri
        An ``hf://buckets/{namespace}/{name}/{path}`` URI. ``path`` is read as:

        * a glob, if it has a glob character (``*``, ``?`` or ``[``), e.g.
          ``data/*.parquet``: every *file* that matches. ``*``, ``?`` and
          ``[...]`` match inside one path segment; ``**`` must be a whole
          segment and matches any number of directories. Braces (``{a,b}``)
          are not expanded. Directories are never passed to the scan. A file
          whose name is exactly the pattern (``data[1].parquet``) is read
          instead of the matches;
        * else a single file, whatever its extension, if a file with exactly
          this name exists;
        * else a directory, or the whole bucket: every ``.parquet`` / ``.pq``
          file below it, at any depth, extension matched case-insensitively.
          A trailing ``/`` forces this reading.
    token
        Hugging Face token. If ``None``, resolved by ``huggingface_hub`` (the
        ``HF_TOKEN`` env var or cached login).
    **scan_kwargs
        Forwarded to :func:`polars.scan_parquet` — e.g.
        ``storage_options={"max_retries": 5}`` for flaky connections,
        ``missing_columns="insert"`` / ``extra_columns="ignore"`` for
        heterogeneous schemas across globbed files, ``schema=``, or
        ``cast_options=``. Options that derive meaning from the file *path*
        (``hive_partitioning=``, ``include_file_paths=``) see the presigned
        CDN URLs, not the bucket paths, so they are not useful here.

    Returns
    -------
    LazyFrame

    Raises
    ------
    ValueError
        The URI is not a valid bucket URI, a glob ends with ``/`` or uses
        ``**`` inside a path segment, or a matched file is empty (0 bytes).
    FileNotFoundError
        The bucket does not exist, or no file matches ``uri``.
    PermissionError
        The Hub answers 401 or 403: the token is not valid or lacks access to
        the bucket. The original ``HfHubHTTPError`` is the ``__cause__``.
    RuntimeError
        The Hub serves a file itself instead of redirecting to a presigned
        URL (a file that is not Xet-backed), or a listing answer is not what
        the Hub API documents.
    TimeoutError
        A listing with many pages did not finish within 10 minutes.
    huggingface_hub.errors.HfHubHTTPError
        Any other HTTP error of the Hub, and a rate-limit (429), timeout
        (408) or server (5xx) answer that the retries did not clear. The
        message of a rate-limit error names the bucket, the quota when the
        Hub sends it, and the number of files already resolved.

    Notes
    -----
    ``scan_bucket`` makes these Hub requests and reads no file data:

    * a file: one ``resolve`` request (HEAD);
    * a directory of N parquet files: one ``resolve`` request that the Hub
      answers "not found", one listing request per page of results, then N
      ``resolve`` requests. With a trailing ``/`` (and for the whole bucket)
      the first request is not made;
    * a glob that selects N files: one listing request per page, then N
      ``resolve`` requests. A glob whose only glob segment is the last one
      (``data/*.parquet``) lists that directory only; other globs list the
      subtree below the text before their first glob character.

    ``resolve`` requests count in the Hub's "resolvers" rate limit.

    The requests go through the shared HTTP session of ``huggingface_hub``, so
    ``HF_HUB_OFFLINE=1`` and a custom client factory
    (``huggingface_hub.set_client_factory``) apply to them.

    A 408, 429 or 5xx answer to any of these requests (every listing page,
    every ``resolve``) is retried up to 5 times. The wait is the one the Hub
    asks for (rate-limit reset, ``Retry-After``), else 1 s doubling up to 8 s;
    a wait of more than 5 s that the Hub asks for is announced with one
    warning per call (logged instead if warnings are turned into errors).
    ``scan_bucket`` raises instead of waiting when the wait would end more
    than 10 minutes after the call started. Timeouts and connection errors
    are not retried.

    Signed URLs are resolved when ``scan_bucket`` is called and are valid for
    ~1 hour. Collect within that window; for long-lived plans, call
    ``scan_bucket`` again to refresh.

    Examples
    --------
    >>> import polars_hf as plhf
    >>> lf = plhf.scan_bucket("hf://buckets/me/data/*.parquet")  # doctest: +SKIP
    >>> lf.filter(pl.col("label") == 1).head(5).collect()  # doctest: +SKIP
    """
    bp = parse_bucket_uri(uri)
    # Read at call time, like HfApi does when it is created.
    endpoint = constants.ENDPOINT
    headers = build_hf_headers(token=token)
    budget = _Budget(bp.bucket_id)

    if bp.path and not bp.is_glob and not bp.path.endswith("/"):
        # A file needs no listing: ask for its signed URL directly. When the
        # Hub answers "not found", the path can still be a directory.
        try:
            urls = _signed_urls(endpoint, headers, [bp.path], budget)
        except FileNotFoundError:
            pass
        else:
            return pl.scan_parquet(urls, **scan_kwargs)

    files = _list_files(endpoint, headers, bp, uri, budget)
    paths = [file.path for file in files]
    urls = _signed_urls(endpoint, headers, paths, budget)
    return pl.scan_parquet(urls, **scan_kwargs)
