"""Scan parquet from Hugging Face buckets with native polars parquet scans.

Bucket files are XET-backed: the Hub ``resolve`` URL 302-redirects (when
requested with auth) to a presigned CDN URL (``us.aws.cdn.hf.co/xet-bridge-*``
on huggingface.co) that needs no auth and supports HTTP range requests. We
follow that redirect in Python and hand the **signed URLs** to native
:func:`polars.scan_parquet`, so polars' Rust object store does async,
concurrent, range-read scans with projection / predicate / row-limit pushdown
— the same mechanism the upstream ``hf://`` reader uses, but reachable from
stock polars.

Stock polars cannot authenticate a generic ``https://`` URL itself (bearer-token
injection is gated behind the ``hf://`` scheme), which is why we resolve the
signed URL here rather than passing the ``resolve`` URL directly.

A signed URL holds a signature and is valid for about one hour. By default
(``resolve="collect"``) ``scan_bucket`` therefore returns a LazyFrame over a
polars IO-plugin source (:class:`_BucketSource`): the plan holds no URL, and
the URLs are resolved when the query runs, for one group of files at a time.
Each group is one native ``scan_parquet``; its output crosses into Python as
whole DataFrames. ``resolve="now"`` resolves every URL in ``scan_bucket`` and
returns the native scan node itself, with the signed URLs in the plan; it is
refused unless the environment variable
``POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN`` is ``1``.

All Hub requests (the bucket listing and the ``resolve`` requests) are sent
with the shared ``huggingface_hub`` session
(``huggingface_hub.utils.get_session``: proxies, offline mode, custom client
factory). Retries of 408, 429 and 5xx answers are done here, for every request
and every listing page, and no wait ends after the deadline of the
``scan_bucket`` call. ``HfApi.list_bucket_tree`` is not used: its retries are
not bounded and differ between huggingface_hub versions.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
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
from polars.io.plugins import register_io_source

from polars_hf._glob import glob_to_regex, has_glob, literal_prefix
from polars_hf._uri import BucketPath, parse_bucket_uri

logger = logging.getLogger(__name__)

# The signed URL carries its own expiry (an ``Expires`` query parameter, about
# one hour after the resolve request), so URLs are resolved at scan time.
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_RESOLVE_MODES = ("collect", "now")
# The mode of a scan_bucket call without resolve=, if this variable is set.
RESOLVE_ENV_VAR = "POLARS_HF_RESOLVE"
# resolve="now" puts presigned URLs into the query plan. It is refused unless
# this variable is "1".
ALLOW_SIGNED_URLS_ENV_VAR = "POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN"
# A collect-time scan resolves and scans the files in groups of this size: one
# group is one burst of resolve requests and one native multi-file scan. The
# URLs of a group are resolved when the group before it is exhausted, so a URL
# is never older than the scan of its own group.
_GROUP_FILES = 64
# A query with a row limit starts with a small group and multiplies the size
# up to _GROUP_FILES: head(5) resolves one file, not a full group.
_LIMITED_FIRST_GROUP = 1
_LIMITED_GROUP_GROWTH = 4
# A collect-time source keeps the signed URLs it resolved. A group uses the
# URL of a file again if it is not older than this many seconds when the scan
# of the group starts; an older URL is resolved again.
_URL_REUSE_SECONDS = 300.0
# scan_parquet options whose result depends on the rows of all earlier files.
# With one of them the files are scanned as one group.
_ONE_GROUP_OPTIONS = ("row_index_name", "n_rows")
# scan_parquet options that can hold credentials. They are passed to the
# native scans and left out of a pickled source.
_UNPICKLED_OPTIONS = ("storage_options", "credential_provider")
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

    A collect-time scan makes one instance per group of files (and one for
    the schema): ``scope`` then names in messages what the deadline applies
    to, and ``files_total`` / ``files_resolved`` carry the progress of the
    whole scan into the group.
    """

    def __init__(
        self,
        bucket_id: str,
        operation: str = "scan_bucket",
        *,
        scope: str | None = None,
        files_total: int = 0,
        files_resolved: int = 0,
    ) -> None:
        self.bucket_id = bucket_id
        self.operation = operation
        self.scope = scope if scope is not None else f"{operation} call"
        self.deadline = time.monotonic() + _SCAN_DEADLINE
        self.files_total = files_total
        # A total given by the caller is the total of the scan: a resolve of
        # some of its files does not replace it.
        self._total_is_given = files_total > 0
        self._files_resolved = files_resolved
        self._wait_announced = False
        self._lock = threading.Lock()

    def count_files(self, total: int) -> None:
        """Set the number of files to resolve, unless the caller gave a total."""
        if not self._total_is_given:
            self.files_total = total

    def file_resolved(self) -> None:
        with self._lock:
            self._files_resolved += 1

    def announce_wait(self, message: str) -> None:
        """Warn about a long wait, once per ``scan_bucket`` call or group.

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
            f"{_SCAN_DEADLINE:.0f} s allowed for one {budget.scope}; "
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
                f"{_SCAN_DEADLINE:.0f} s allowed for one {budget.scope} "
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
    budget.count_files(len(paths))

    def resolve(path: str) -> str:
        url = _signed_url(
            _resolve_url(endpoint, bucket_id, path),
            headers,
            uri=_file_uri(bucket_id, path),
            budget=budget,
        )
        budget.file_resolved()
        return url

    if not paths:
        return []
    if len(paths) == 1:
        return [resolve(paths[0])]
    with ThreadPoolExecutor(max_workers=min(_MAX_RESOLVE_WORKERS, len(paths))) as pool:
        return list(pool.map(resolve, paths))


# ---- collect-time source ---------------------------------------------------


def _group_bounds(
    n_files: int, *, limited: bool, one_group: bool
) -> Iterator[tuple[int, int]]:
    """The ``(start, stop)`` file indices of every group of a scan, in order.

    Without a row limit every group has ``_GROUP_FILES`` files. With one, the
    first group has ``_LIMITED_FIRST_GROUP`` file and the size is multiplied
    by ``_LIMITED_GROUP_GROWTH`` up to ``_GROUP_FILES``.
    """
    if one_group:
        yield (0, n_files)
        return
    size = _LIMITED_FIRST_GROUP if limited else _GROUP_FILES
    start = 0
    while start < n_files:
        stop = min(start + size, n_files)
        yield (start, stop)
        start = stop
        size = min(size * _LIMITED_GROUP_GROWTH, _GROUP_FILES)


# Names of query parameters that hold a signature or a credential of a
# presigned URL (CloudFront, S3 and Xet forms), matched case-insensitively.
_SIGNED_PARAMETER = (
    r"(?:Signature|Policy|Key-Pair-Id|X-Amz-[A-Za-z0-9-]+|X-Xet-[A-Za-z0-9-]+)"
)
# One such parameter with its value, in plain or percent-encoded form
# ("name=value" or "name%3Dvalue"; the value ends at "&" or "%26").
_SIGNED_PARAMETER_VALUE = re.compile(
    _SIGNED_PARAMETER + r"(?:=|%3D)(?:(?!%26)[^&\s\"'<>()\[\]])*", re.IGNORECASE
)
_REMOVED_URL = "<presigned URL removed>"
_REMOVED_PARAMETER = "<removed>"


def _scrub_signed_urls(text: str, uris_by_url: dict[str, str]) -> str:
    """``text`` without the signed URLs of ``uris_by_url`` and their parts.

    An exact URL is replaced by the ``hf://`` URI of its file. Any other text
    on the host of one of the URLs (a truncated, percent-encoded or re-quoted
    URL) and any signature-like query parameter is removed.
    """
    hosts = set()
    # Longest first: a URL can be the start of another one.
    for url in sorted(uris_by_url, key=len, reverse=True):
        text = text.replace(url, uris_by_url[url])
        host = urlparse(url).netloc
        if host:
            hosts.add(host)
    for host in hosts:
        # The host with an optional scheme ("https://" or "https%3A%2F%2F")
        # and everything up to the next space, quote or bracket.
        on_host = re.compile(
            r"(?:[a-z][a-z0-9+.-]*(?::|%3A)(?://|%2F%2F))?"
            + re.escape(host)
            + r"[^\s\"'<>()\[\]]*",
            re.IGNORECASE,
        )
        text = on_host.sub(_REMOVED_URL, text)
    return _SIGNED_PARAMETER_VALUE.sub(_REMOVED_PARAMETER, text)


def _exception_chain(error: BaseException) -> list[BaseException]:
    """``error`` and every exception linked to it as cause or context."""
    chain: list[BaseException] = []
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or any(current is seen for seen in chain):
            continue
        chain.append(current)
        pending.append(current.__cause__)
        pending.append(current.__context__)
    return chain


def _without_signed_urls(
    error: Exception, uris_by_url: dict[str, str]
) -> Exception | None:
    """A copy of ``error`` whose message names bucket files, not signed URLs.

    Polars puts the URL of a file in the message of a read error (an expired
    URL, a file that is not parquet). Returns ``None`` if no message in the
    exception chain of ``error`` holds a signed URL or a part of one.

    The copy has no cause and no context. The caller must raise it outside
    of the ``except`` block that caught ``error``: an exception raised inside
    the block gets ``error`` as its ``__context__``, also with ``from None``.
    """
    changed = False
    for linked in _exception_chain(error):
        text = str(linked)
        if _scrub_signed_urls(text, uris_by_url) != text:
            changed = True
    if not changed:
        return None
    clean = _scrub_signed_urls(str(error), uris_by_url)
    try:
        return type(error)(clean)
    except Exception:
        return RuntimeError(clean)


# Column types that are cheap to read, cheapest first.
_NARROW_TYPES = (
    pl.Boolean,
    pl.Int8,
    pl.UInt8,
    pl.Int16,
    pl.UInt16,
    pl.Int32,
    pl.UInt32,
    pl.Float32,
    pl.Date,
)


def _narrowest_column(schema: pl.Schema) -> str:
    """The name of a column of ``schema`` that is cheap to read.

    A column of the first type of ``_NARROW_TYPES`` that the schema has, else
    the first numeric or temporal column, else the first column.
    """
    for narrow in _NARROW_TYPES:
        for name, dtype in schema.items():
            if dtype == narrow:
                return name
    for name, dtype in schema.items():
        if dtype.is_numeric() or dtype.is_temporal():
            return name
    return schema.names()[0]


class _BucketSource:
    """The files of one collect-time ``scan_bucket`` call, as a polars IO source.

    Polars calls :meth:`schema` when it needs the schema of the LazyFrame, and
    calls the object when the query runs. The object holds the bucket paths;
    no signed URL is part of the query plan.

    The object keeps the signed URLs it resolved, to use them again in a
    query that starts within ``_URL_REUSE_SECONDS``.

    A pickled source (``LazyFrame.serialize``) holds the endpoint, the bucket
    id, the paths and the scan options without ``storage_options`` and
    ``credential_provider``. It does not hold the token, the request headers
    or a signed URL: a source that was unpickled sends its requests with the
    token that ``huggingface_hub`` finds in its environment.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        bucket_id: str,
        paths: list[str],
        scan_kwargs: dict[str, object],
        first_url: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.bucket_id = bucket_id
        self.paths = paths
        self.scan_kwargs = scan_kwargs
        self._init_unpickled_state(headers)
        if first_url is not None:
            self._urls[0] = (first_url, time.monotonic())

    def _init_unpickled_state(self, headers: dict[str, str] | None) -> None:
        self._headers = headers
        self._schema: pl.Schema | None = None
        # File index -> (signed URL, time it was resolved). At most one entry
        # per file.
        self._urls: dict[int, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._schema_lock = threading.Lock()

    def __getstate__(self) -> dict[str, object]:
        scan_kwargs = {}
        for name, value in self.scan_kwargs.items():
            if name not in _UNPICKLED_OPTIONS:
                scan_kwargs[name] = value
        return {
            "endpoint": self.endpoint,
            "bucket_id": self.bucket_id,
            "paths": self.paths,
            "scan_kwargs": scan_kwargs,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._init_unpickled_state(None)

    # ---- requests ----------------------------------------------------------

    def _hub_headers(self) -> dict[str, str]:
        with self._lock:
            if self._headers is None:
                self._headers = build_hf_headers()
            return self._headers

    def _uri(self, index: int) -> str:
        return _file_uri(self.bucket_id, self.paths[index])

    def _uris_by_url(self, start: int, urls: list[str]) -> dict[str, str]:
        uris_by_url = {}
        for offset, url in enumerate(urls):
            uris_by_url[url] = self._uri(start + offset)
        return uris_by_url

    def _forget_urls(self, start: int, stop: int) -> None:
        with self._lock:
            for index in range(start, stop):
                self._urls.pop(index, None)

    def _resolve(self, start: int, stop: int, scope: str) -> tuple[list[str], bool]:
        """The signed URLs of the files ``start`` to ``stop - 1``.

        A URL that this object resolved less than ``_URL_REUSE_SECONDS`` ago
        is used again; the other files are resolved, with their own retry
        budget (``_SCAN_DEADLINE`` from now). Returns the URLs and whether
        one of them was used again.
        """
        now = time.monotonic()
        urls: dict[int, str] = {}
        with self._lock:
            for index in range(start, stop):
                cached = self._urls.get(index)
                if cached is not None and now - cached[1] <= _URL_REUSE_SECONDS:
                    urls[index] = cached[0]
        reused = len(urls) > 0

        missing = []
        for index in range(start, stop):
            if index not in urls:
                missing.append(index)
        if missing:
            budget = _Budget(
                self.bucket_id,
                scope=scope,
                files_total=len(self.paths),
                files_resolved=start + len(urls),
            )
            paths = [self.paths[index] for index in missing]
            resolved = _signed_urls(self.endpoint, self._hub_headers(), paths, budget)
            resolved_at = time.monotonic()
            with self._lock:
                for index, url in zip(missing, resolved, strict=True):
                    urls[index] = url
                    self._urls[index] = (url, resolved_at)
        return [urls[index] for index in range(start, stop)], reused

    # ---- schema ------------------------------------------------------------

    def schema(self) -> pl.Schema:
        """The schema of the LazyFrame; read once, from the first file.

        It is the schema of a native scan of the first file with the scan
        options of the call: one ``resolve`` request (none if the URL of the
        file is recent) and the footer of that file, unless ``schema=`` makes
        the read unnecessary for polars.
        """
        with self._schema_lock:
            if self._schema is None:
                self._schema = self._read_schema()
            return self._schema

    def _read_schema(self) -> pl.Schema:
        scope = "schema read of a scan_bucket query"
        # A second attempt only if the first one used a URL again: the CDN
        # can refuse a URL before it is _URL_REUSE_SECONDS old.
        for attempt in (0, 1):
            urls, reused = self._resolve(0, 1, scope)
            lf = pl.scan_parquet(urls, **self.scan_kwargs)
            failure = None
            try:
                return lf.collect_schema()
            except Exception as error:
                self._forget_urls(0, 1)
                if reused and attempt == 0:
                    continue
                failure = _without_signed_urls(error, self._uris_by_url(0, urls))
                if failure is None:
                    raise
            # Raised outside of the except block: no __context__.
            raise failure
        raise AssertionError("unreachable")  # pragma: no cover

    def _file_schema(self) -> pl.Schema:
        """The columns of :meth:`schema` that are read from the files."""
        added = (
            self.scan_kwargs.get("include_file_paths"),
            self.scan_kwargs.get("row_index_name"),
        )
        columns = {}
        for name, dtype in self.schema().items():
            if name not in added:
                columns[name] = dtype
        return pl.Schema(columns)

    # ---- scan --------------------------------------------------------------

    def _scan_group(self, urls: list[str], uris_by_url: dict[str, str]) -> pl.LazyFrame:
        """The native scan of one group of files."""
        scan_kwargs = dict(self.scan_kwargs)
        if scan_kwargs.get("schema") is None:
            # Polars takes the schema of a scan from its first file. Every
            # group must use the schema of the first file of the whole scan.
            scan_kwargs["schema"] = self._file_schema()
        lf = pl.scan_parquet(urls, **scan_kwargs)
        path_column = scan_kwargs.get("include_file_paths")
        if path_column is not None:
            # Polars fills the column with the signed URLs. Two files of one
            # group that have the same URL (the same content) get one URI.
            bucket_uri = pl.col(path_column).replace_strict(
                uris_by_url, return_dtype=pl.String
            )
            lf = lf.with_columns(bucket_uri)
        return lf

    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """Scan the files group by group (the IO source function of polars).

        The projection, the predicate and the row limit that polars pushed
        into the source are applied to the native scan of every group. When
        polars gives a row limit and a predicate, the limit is on the rows
        before the predicate (``head(n).filter(...)``).
        """
        one_group = False
        for option in _ONE_GROUP_OPTIONS:
            if self.scan_kwargs.get(option) is not None:
                one_group = True
        if n_rows is not None and predicate is not None:
            # The batches hold the rows after the predicate: they do not tell
            # how many rows of the limit a group used.
            one_group = True
        if with_columns is not None and len(with_columns) == 0:
            # A frame without columns has no rows. Read one cheap column, so
            # that the frames have the row count of the files.
            with_columns = [_narrowest_column(self.schema())]
        groups = _group_bounds(
            len(self.paths), limited=n_rows is not None, one_group=one_group
        )
        scope = "group of files of a scan_bucket query"
        remaining = n_rows
        for start, stop in groups:
            # A second attempt only if the first one used a URL again and
            # failed before its first batch: the CDN can refuse a URL before
            # it is _URL_REUSE_SECONDS old.
            for attempt in (0, 1):
                urls, reused = self._resolve(start, stop, scope)
                uris_by_url = self._uris_by_url(start, urls)
                lf = self._scan_group(urls, uris_by_url)
                if remaining is not None:
                    lf = lf.head(remaining)
                if predicate is not None:
                    lf = lf.filter(predicate)
                if with_columns is not None:
                    lf = lf.select(with_columns)

                yielded = False
                failure = None
                try:
                    batches = lf.collect_batches(
                        chunk_size=batch_size, engine="streaming"
                    )
                    for batch in batches:
                        if remaining is not None:
                            remaining -= batch.height
                        yielded = True
                        yield batch
                    break
                except Exception as error:
                    self._forget_urls(start, stop)
                    if reused and not yielded and attempt == 0:
                        continue
                    failure = _without_signed_urls(error, uris_by_url)
                    if failure is None:
                        raise
                # Raised outside of the except block: no __context__.
                raise failure
            if remaining is not None and remaining <= 0:
                return


@dataclass
class _Found:
    """The files that a URI names, and what the requests for them need."""

    endpoint: str
    headers: dict[str, str]
    bucket_id: str
    paths: list[str]
    budget: _Budget
    # The signed URL of a single-file URI: the request that tells a file
    # from a directory resolves it.
    first_url: str | None = None


def _find_files(uri: str, token: str | None) -> _Found:
    """Parse ``uri`` and find its files: a single file, else a listing."""
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
            return _Found(endpoint, headers, bp.bucket_id, [bp.path], budget, urls[0])

    files = _list_files(endpoint, headers, bp, uri, budget)
    paths = [file.path for file in files]
    return _Found(endpoint, headers, bp.bucket_id, paths, budget)


def _all_signed_urls(found: _Found) -> list[str]:
    """The signed URL of every file of ``found``, within the budget of the call."""
    if found.first_url is not None:
        return [found.first_url]
    return _signed_urls(found.endpoint, found.headers, found.paths, found.budget)


def _resolve_mode(resolve: str | None) -> str:
    """The mode of a ``scan_bucket`` call; raises for one that is not allowed."""
    source = "resolve"
    if resolve is None:
        resolve = os.environ.get(RESOLVE_ENV_VAR) or "collect"
        source = f"the environment variable {RESOLVE_ENV_VAR}"
    if resolve not in _RESOLVE_MODES:
        raise ValueError(f"{source} must be one of {_RESOLVE_MODES}, not {resolve!r}")
    if resolve == "now" and os.environ.get(ALLOW_SIGNED_URLS_ENV_VAR) != "1":
        raise ValueError(
            'resolve="now" puts presigned URLs into the query plan: each one '
            "gives read access to one file for about 60 minutes to anyone who "
            "sees it in explain(), serialize(), an error message or a log. To "
            f"accept that, set the environment variable {ALLOW_SIGNED_URLS_ENV_VAR}=1; "
            'else use resolve="collect" (the default).'
        )
    return resolve


def _collect_time_scan(source: _BucketSource, uri: str) -> pl.LazyFrame:
    """The LazyFrame of a collect-time scan: an IO-plugin node over ``source``."""
    options = {}
    parameters = inspect.signature(register_io_source).parameters
    if "explain_name" in parameters:
        # Polars 2 prints these in explain() instead of "PYTHON SCAN".
        options["explain_name"] = "HF BUCKET SCAN"
        options["explain_detail"] = f"{uri} ({len(source.paths)} files)"
    return register_io_source(source, schema=source.schema, **options)


def scan_bucket(
    uri: str,
    *,
    token: str | None = None,
    resolve: str | None = None,
    **scan_kwargs: object,
) -> pl.LazyFrame:
    """Lazily scan parquet file(s) from a Hugging Face bucket.

    The files are read by native :func:`polars.scan_parquet` scans over
    presigned URLs, so only the column chunks a query needs are transferred.
    ``scan_bucket`` itself finds the files and reads no file data.

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
    resolve
        When the presigned URLs of the files are resolved:

        * ``"collect"``: when the query runs. The LazyFrame is a polars
          IO-plugin source that holds the bucket paths. The plan
          (``explain()``, ``serialize()``) holds no signed URL, and a URL is
          resolved shortly before it is used, so the plan does not expire.
          A row count reads one whole column and ``tail()`` scans all files
          (see Notes); :func:`count_rows` counts from the footers.
        * ``"now"``: in ``scan_bucket``. The LazyFrame is the native
          ``scan_parquet`` node over the signed URLs. It is the faster path
          for metadata-heavy work (row counts, ``tail()``), and it puts a
          read-only URL of every file, valid for about 60 minutes, into
          ``explain()``, ``serialize()``, error messages and logs. It is
          refused with a ``ValueError`` unless the environment variable
          ``POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN`` is ``1``.
        * ``None`` (default): the value of the environment variable
          ``POLARS_HF_RESOLVE`` if it is set, else ``"collect"``.

        See Notes for what differs between the two.
    **scan_kwargs
        Forwarded to :func:`polars.scan_parquet` — e.g.
        ``storage_options={"max_retries": 5}`` for flaky connections,
        ``missing_columns="insert"`` / ``extra_columns="ignore"`` for
        heterogeneous schemas across globbed files, ``schema=``, or
        ``cast_options=``. ``include_file_paths="col"`` gives the
        ``hf://buckets/...`` URI of the file of every row with
        ``resolve="collect"``, and the signed URL with ``resolve="now"``.
        (Two files with the same content that get the same URL and are in
        the same group show one URI.)
        ``hive_partitioning=`` sees the presigned URLs, not the bucket paths,
        so it finds no partition columns.

    Returns
    -------
    LazyFrame

    Raises
    ------
    ValueError
        The URI is not a valid bucket URI, a glob ends with ``/`` or uses
        ``**`` inside a path segment, ``resolve`` (or ``POLARS_HF_RESOLVE``)
        is not ``"collect"`` or ``"now"``, ``resolve="now"`` is used without
        ``POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1``, or a matched file is empty
        (0 bytes).
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
    **Hub requests of the call.** ``scan_bucket`` finds the files:

    * a file: one ``resolve`` request (HEAD);
    * a directory: one ``resolve`` request that the Hub answers "not found",
      then one listing request per page of results. With a trailing ``/``
      (and for the whole bucket) the first request is not made;
    * a glob: one listing request per page. A glob whose only glob segment is
      the last one (``data/*.parquet``) lists that directory only; other
      globs list the subtree below the text before their first glob
      character.

    With ``resolve="now"`` it then sends one ``resolve`` request for each of
    the N files.

    **Hub requests of a query** (``resolve="collect"``). The schema is read
    when polars first needs it (``collect()``, ``collect_schema()``,
    ``explain()``), once per LazyFrame: one ``resolve`` request for the first
    file and the footer of that file. A query then scans the files in path
    order in groups of 64. The URLs of a group are resolved (one ``resolve``
    request per file) when the group before it is exhausted, and the group is
    one native multi-file scan with the projection, the predicate and the row
    limit of the query. A query with a row limit (``head(n)``) uses groups of
    1, 4, 16, then 64 files and stops when it has its rows.

    The LazyFrame keeps the URLs it resolved, at most one per file. A group
    uses the URL of a file again if it is less than 5 minutes old when the
    scan of the group starts; an older URL is resolved again. So a full scan
    of N files sends N ``resolve`` requests with the one of the schema, a
    second query within 5 minutes sends none, and a single-file scan sends
    none after the one of the call. If the CDN refuses a URL that was used
    again, the group is resolved and scanned once more, provided that it has
    not returned rows yet.

    ``resolve`` requests count in the Hub's "resolvers" rate limit.

    The requests go through the shared HTTP session of ``huggingface_hub``, so
    ``HF_HUB_OFFLINE=1`` and a custom client factory
    (``huggingface_hub.set_client_factory``) apply to them.

    **Retries.** A 408, 429 or 5xx answer to any of these requests (every
    listing page, every ``resolve``) is retried up to 5 times. The wait is
    the one the Hub asks for (rate-limit reset, ``Retry-After``), else 1 s
    doubling up to 8 s; a wait of more than 5 s that the Hub asks for is
    announced with one warning (logged instead if warnings are turned into
    errors). A wait is made only if it ends within 10 minutes; else the error
    is raised. The 10 minutes start with the ``scan_bucket`` call for its own
    requests, and with the first request of the schema read and of every
    group for the requests of a query. Timeouts and connection errors are not
    retried.

    **Errors of a query** (``resolve="collect"``). An error of a ``resolve``
    request of a query (a file that was deleted after the listing, a 401 or
    403, a rate limit that the retries did not clear) is raised by
    ``collect()``, not by ``scan_bucket``. Polars 2 raises the exception of
    the list above. Polars 1.x wraps an exception of an IO source: it raises
    ``polars.exceptions.ComputeError`` whose message holds the type name and
    the message of that exception. All polars versions wrap an error of the
    schema read in a ``ComputeError`` ("schema callable failed"). A read
    error of polars names the ``hf://`` URI of the file, not its signed URL,
    and has no cause or context that holds the URL. With ``POLARS_VERBOSE=1``
    polars itself prints the URLs it scans to stderr, in both modes.

    **What the IO-plugin node changes for a query** (``resolve="collect"``):

    * polars pushes the projection, the predicate and ``head(n)`` into the
      source. It does not push ``tail()`` or a slice with an offset: these
      scan all files. ``resolve="now"`` reads only the last files for them;
    * ``select(pl.len())`` reads one whole column of every file, because
      polars asks an IO source for one column to count the rows. Polars
      chooses the column; it can be the largest one. :func:`count_rows`
      reads the footers only;
    * on files with different columns or types, a row count or a slice
      outside of the rows raises a schema error; ``resolve="now"`` answers
      them from the footers;
    * a consumer that stops reading a ``collect_batches()`` iterator does
      not stop the query, so the following groups are still resolved;
    * a group must be scanned within the ~1 hour that its URLs are valid;
    * with ``row_index_name=`` or ``n_rows=``, and for a row limit before a
      predicate (``head(n).filter(...)``), all files are one group: all
      their URLs are resolved at the start of the query;
    * every group is scanned with the schema of the first file of the scan
      (as ``schema=``), unless ``schema=`` is given;
    * ``LazyFrame.serialize()`` needs the ``cloudpickle`` package. The
      serialized plan holds the bucket paths and the scan options without
      ``storage_options`` and ``credential_provider``, and not the token:
      the query that is deserialized runs without these two options and
      uses the token of its own environment.

    Examples
    --------
    >>> import polars_hf as plhf
    >>> lf = plhf.scan_bucket("hf://buckets/me/data/*.parquet")  # doctest: +SKIP
    >>> lf.filter(pl.col("label") == 1).head(5).collect()  # doctest: +SKIP
    """
    mode = _resolve_mode(resolve)
    found = _find_files(uri, token)
    if mode == "now":
        return pl.scan_parquet(_all_signed_urls(found), **scan_kwargs)

    source = _BucketSource(
        endpoint=found.endpoint,
        headers=found.headers,
        bucket_id=found.bucket_id,
        paths=found.paths,
        scan_kwargs=scan_kwargs,
        first_url=found.first_url,
    )
    return _collect_time_scan(source, uri)


def count_rows(uri: str, *, token: str | None = None) -> int:
    """Count the rows of parquet file(s) in a Hugging Face bucket.

    The count is read from the parquet footers: no column data is
    transferred. Use it instead of ``scan_bucket(uri).select(pl.len())``,
    which reads one column of every file (see the Notes of
    :func:`scan_bucket`).

    Parameters
    ----------
    uri
        An ``hf://buckets/{namespace}/{name}/{path}`` URI, read as by
        :func:`scan_bucket`: a glob, a single file, or a directory.
    token
        Hugging Face token. If ``None``, resolved by ``huggingface_hub``.

    Returns
    -------
    int
        The total number of rows of the files.

    Raises
    ------
    ValueError, FileNotFoundError, PermissionError, RuntimeError, TimeoutError, \
huggingface_hub.errors.HfHubHTTPError
        As :func:`scan_bucket`, all from this call.
    polars.exceptions.PolarsError, OSError
        A file cannot be read as parquet. The message names the ``hf://`` URI
        of the file, not its presigned URL.

    Notes
    -----
    The call finds the files like ``scan_bucket``, sends one ``resolve``
    request per file and reads the footer of every file with a native
    ``scan_parquet(url).select(pl.len())``, in one query. It keeps nothing:
    no URL, no LazyFrame. The schemas of the files are not compared: files
    with different columns or types are counted.

    Examples
    --------
    >>> import polars_hf as plhf
    >>> plhf.count_rows("hf://buckets/me/data/*.parquet")  # doctest: +SKIP
    1000000
    """
    found = _find_files(uri, token)
    urls = _all_signed_urls(found)
    uris_by_url = {}
    for path, url in zip(found.paths, urls, strict=True):
        uris_by_url[url] = _file_uri(found.bucket_id, path)

    # One scan per file: a scan of all files compares their schemas.
    counts = []
    for url in urls:
        counts.append(pl.scan_parquet(url).select(pl.len()))
    total = pl.concat(counts).select(pl.col("len").cast(pl.UInt64).sum())

    failure = None
    try:
        return total.collect().item()
    except Exception as error:
        failure = _without_signed_urls(error, uris_by_url)
        if failure is None:
            raise
    # Raised outside of the except block: no __context__.
    raise failure
