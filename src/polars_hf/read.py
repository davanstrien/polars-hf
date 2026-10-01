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

All Hub requests go through ``huggingface_hub``: the listing through
``HfApi.list_bucket_tree`` and the ``resolve`` request through the shared Hub
session (``huggingface_hub.utils.get_session``: proxies, offline mode, custom
client factory). Retries of 408, 429 and 5xx answers are done here, with a
limit on every wait and on the total time of one ``scan_bucket`` call.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urljoin, urlparse

import polars as pl
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import (
    build_hf_headers,
    get_session,
    hf_raise_for_status,
    parse_ratelimit_headers,
)

from polars_hf._glob import glob_to_regex, has_glob, literal_prefix
from polars_hf._uri import BucketPath, parse_bucket_uri

# The signed URL carries its own expiry (an ``Expires`` query parameter, about
# one hour after the resolve request), so URLs are resolved at scan time.
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_MAX_RESOLVE_WORKERS = 16
_MAX_REDIRECT_HOPS = 5

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
# The server can ask for a longer wait (rate-limit reset, Retry-After). No
# single wait is longer than this, and no wait ends after the deadline of the
# scan_bucket call; the scan fails instead of sleeping for minutes or hours.
_MAX_WAIT_PER_RETRY = 60.0
_SCAN_DEADLINE = 600.0


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
    deadline applies to the call as a whole.
    """

    def __init__(self, bucket_id: str) -> None:
        self.bucket_id = bucket_id
        self.deadline = time.monotonic() + _SCAN_DEADLINE
        self.files_total = 0
        self._files_resolved = 0
        self._lock = threading.Lock()

    def file_resolved(self) -> None:
        with self._lock:
            self._files_resolved += 1

    def progress(self) -> str:
        """``"; 3 of 20 files were resolved"``, or ``""`` before the resolves."""
        if self.files_total == 0:
            return ""
        with self._lock:
            resolved = self._files_resolved
        return f"; {resolved} of {self.files_total} files were resolved"


def _server_wait_hint(response) -> float | None:
    """Seconds the server asks to wait before the next attempt, if it says so."""
    if response.status_code == 429:
        info = parse_ratelimit_headers(response.headers)
        if info is not None and info.remaining == 0:
            # One more second: the reset time is rounded down.
            return float(info.reset_in_seconds) + 1
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            # An HTTP date: not used by the Hub; fall back to the backoff.
            return None
    return None


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

    wait = _server_wait_hint(response)
    if wait is None:
        wait = min(_RETRY_MAX_BACKOFF, _RETRY_BASE_WAIT * 2**attempt)
    if wait > _MAX_WAIT_PER_RETRY:
        reason = (
            f"the Hub asks to wait {wait:.0f} s, more than the limit of "
            f"{_MAX_WAIT_PER_RETRY:.0f} s per retry; try again later"
        )
        raise _retry_error(response, budget, what, reason)
    if time.monotonic() + wait > budget.deadline:
        reason = (
            f"the next retry would pass the limit of {_SCAN_DEADLINE:.0f} s "
            "for one scan_bucket call; try again later"
        )
        raise _retry_error(response, budget, what, reason)
    time.sleep(wait)


# ---- listing ---------------------------------------------------------------


def _list_tree(
    api: HfApi,
    bucket_id: str,
    prefix: str,
    *,
    recursive: bool,
    uri: str,
    budget: _Budget,
) -> list:
    """All entries of one ``list_bucket_tree`` call (paginated by the client).

    A 408 / 429 / 5xx answer restarts the whole listing, within the limits of
    ``budget``. (Inside one listing, ``huggingface_hub`` itself retries the
    requests for the pages after the first one.)
    """
    attempt = 0
    while True:
        try:
            return list(
                api.list_bucket_tree(
                    bucket_id, prefix=prefix or None, recursive=recursive
                )
            )
        except HfHubHTTPError as error:
            status = _status_code(error)
            if status in (401, 403):
                raise _no_access_error(bucket_id, uri, status) from error
            if status == 404:
                raise FileNotFoundError(
                    f"bucket {bucket_id!r} not found (or the token has no access "
                    f"to it): {uri!r}"
                ) from error
            if status not in _RETRY_STATUS_CODES:
                raise
            _wait_before_retry(error.response, attempt, budget, "listing")
            attempt += 1


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


def _parquet_files_below(files: list, directory: str) -> list:
    """The parquet files below ``directory`` (``""`` is the whole bucket).

    The Hub matches a listing prefix as a plain string, so ``files`` can hold
    siblings such as ``data.parquet`` and ``data2/x`` for the prefix ``data``.
    """
    directory_prefix = f"{directory}/" if directory else ""
    selected = []
    for file in files:
        if file.path.startswith(directory_prefix) and _is_parquet_name(file.path):
            selected.append(file)
    return selected


def _glob_matches(files: list, pattern: str) -> list:
    regex = glob_to_regex(pattern)
    selected = []
    for file in files:
        if regex.fullmatch(file.path):
            selected.append(file)
    return selected


def _glob_is_in_one_directory(pattern: str) -> bool:
    """Whether only the last segment of ``pattern`` is a glob, without ``**``."""
    segments = pattern.split("/")
    for segment in segments[:-1]:
        if has_glob(segment):
            return False
    return "**" not in segments[-1]


def _select_recursive(
    api: HfApi, bp: BucketPath, path: str, uri: str, budget: _Budget
) -> list:
    """One recursive listing; ``path`` as a file, a directory, then a glob."""
    # Every candidate (the literal file, the files below the literal
    # directory, the glob matches) starts with the text before the first glob
    # character, so one listing covers all three readings of the path.
    prefix = literal_prefix(path) if bp.is_glob else path
    entries = _list_tree(
        api, bp.bucket_id, prefix, recursive=True, uri=uri, budget=budget
    )
    files = _files(entries)

    if not bp.path.endswith("/"):
        selected = _exact_file(files, path)
        if selected:
            return selected
    selected = _parquet_files_below(files, path)
    if selected or not bp.is_glob:
        return selected
    return _glob_matches(files, path)


def _select_in_one_directory(
    api: HfApi, bp: BucketPath, path: str, uri: str, budget: _Budget
) -> list:
    """A glob in the last segment only: list its directory, not the subtree."""
    parent = path.rpartition("/")[0]
    entries = _list_tree(
        api, bp.bucket_id, parent, recursive=False, uri=uri, budget=budget
    )
    files = _files(entries)

    selected = _exact_file(files, path)
    if selected:
        return selected
    # A directory whose name has glob characters ('run[1]'): read it as a
    # directory, which needs the listing of its subtree.
    for entry in entries:
        if entry.type == "directory" and entry.path == path:
            below = _list_tree(
                api, bp.bucket_id, path, recursive=True, uri=uri, budget=budget
            )
            selected = _parquet_files_below(_files(below), path)
            if selected:
                return selected
    return _glob_matches(files, path)


def _list_files(api: HfApi, bp: BucketPath, uri: str, budget: _Budget) -> list[str]:
    """List the bucket and return the sorted paths of the files ``bp`` names.

    The path is tried, in order, as: the exact name of a file; a directory
    (all parquet files below it); a glob. A name that contains glob characters
    (``data[1].parquet``) is therefore read literally when such a file or
    directory exists.
    """
    path = bp.path.rstrip("/")
    if bp.is_glob and bp.path.endswith("/"):
        raise ValueError(
            f"a glob cannot end with '/': {uri!r} (a glob selects files; use "
            f"'{path}/*.parquet' for the files of the matching directories or "
            f"'{path}/**/*.parquet' for all files below them)"
        )

    if bp.is_glob and _glob_is_in_one_directory(path):
        selected = _select_in_one_directory(api, bp, path, uri, budget)
    else:
        selected = _select_recursive(api, bp, path, uri, budget)
    if not selected:
        raise FileNotFoundError(f"no parquet files matched: {uri!r}")

    for file in selected:
        if file.size == 0:
            raise _empty_file_error(_file_uri(bp.bucket_id, file.path))
    return sorted(file.path for file in selected)


# ---- resolve ---------------------------------------------------------------


def _resolve_url(endpoint: str, bucket_id: str, path: str) -> str:
    """The Hub ``resolve`` URL of a bucket file (the one ``HfApi`` requests)."""
    return f"{endpoint}/buckets/{bucket_id}/resolve/{quote(path, safe='')}"


def _head(url: str, headers: dict[str, str], budget: _Budget):
    """One HEAD request with auth that does not follow redirects.

    The request is sent with the shared ``huggingface_hub`` session. A 408 /
    429 / 5xx answer is retried within the limits of ``budget``; when they are
    reached, ``HfHubHTTPError`` is raised. Any other answer is returned.

    Timeouts and connection errors are not retried and keep their type.
    """
    attempt = 0
    while True:
        response = get_session().request(
            "HEAD",
            url,
            headers=headers,
            follow_redirects=False,
            timeout=_REQUEST_TIMEOUT,
        )
        if response.status_code not in _RETRY_STATUS_CODES:
            return response
        _wait_before_retry(response, attempt, budget, "resolve")
        attempt += 1


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
    hub = urlparse(resolve_url)
    hub_origin = (hub.scheme, hub.hostname, hub.port)
    url = resolve_url
    for _ in range(_MAX_REDIRECT_HOPS):
        response = _head(url, headers, budget)
        if response.status_code in _REDIRECT_CODES and "location" in response.headers:
            # urljoin resolves relative *and* protocol-relative (//host/..)
            # locations; compare origins rather than sniffing the scheme prefix.
            # (.hostname is lower-cased by urlparse, so host case is ignored.)
            location = urljoin(url, response.headers["location"])
            target = urlparse(location)
            if (target.scheme, target.hostname, target.port) != hub_origin:
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
        An ``hf://buckets/{namespace}/{name}/{path}`` URI. ``path`` is read as,
        in this order:

        * a single file, whatever its extension, if a file with exactly this
          name exists (also when the name contains glob characters, such as
          ``data[1].parquet``);
        * a directory or the whole bucket: every ``.parquet`` / ``.pq`` file
          below it, at any depth, extension matched case-insensitively. A
          trailing ``/`` forces this reading;
        * a glob (e.g. ``data/*.parquet``): every *file* that matches. ``*``,
          ``?`` and ``[...]`` match inside one path segment; ``**`` must be a
          whole segment and matches any number of directories. Braces
          (``{a,b}``) are not expanded. Directories are never passed to the
          scan.
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
        URL (a file that is not Xet-backed).
    huggingface_hub.errors.HfHubHTTPError
        Any other HTTP error of the Hub, and a rate-limit (429), timeout
        (408) or server (5xx) answer that the retries did not clear. The
        message of a rate-limit error names the bucket, the quota when the
        Hub sends it, and the number of files already resolved.

    Notes
    -----
    ``scan_bucket`` makes these Hub requests and reads no file data:

    * a path that ends in ``.parquet`` / ``.pq`` and has no glob character:
      one ``resolve`` request (HEAD). If the Hub answers "not found", the path
      is then handled as a directory;
    * any other path: one listing request per page of results, then one
      ``resolve`` request per file. A glob whose only glob segment is the last
      one (``data/*.parquet``) lists that directory only; a directory scan and
      a glob with ``**`` list the whole subtree. ``resolve`` requests count in
      the Hub's "resolvers" rate limit.

    The requests go through the shared HTTP session of ``huggingface_hub``, so
    ``HF_HUB_OFFLINE=1`` and a custom client factory
    (``huggingface_hub.set_client_factory``) apply to them.

    A 408, 429 or 5xx answer to a listing or a ``resolve`` request is retried
    up to 5 times. The wait is the one the Hub asks for (rate-limit reset,
    ``Retry-After``), else 1 s doubling up to 8 s. ``scan_bucket`` raises
    instead of waiting when one wait would be longer than 60 s or would end
    more than 10 minutes after the call started. Timeouts and connection
    errors are not retried.

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
    api = HfApi(token=token)
    headers = build_hf_headers(token=token)
    budget = _Budget(bp.bucket_id)

    # The common case, one parquet file, needs no listing: ask for its signed
    # URL directly. The suffix only chooses which request is tried first; a
    # "not found" answer falls through to the listing, which decides.
    if not bp.is_glob and _is_parquet_name(bp.path):
        try:
            urls = _signed_urls(api.endpoint, headers, [bp.path], budget)
        except FileNotFoundError:
            pass
        else:
            return pl.scan_parquet(urls, **scan_kwargs)

    paths = _list_files(api, bp, uri, budget)
    urls = _signed_urls(api.endpoint, headers, paths, budget)
    return pl.scan_parquet(urls, **scan_kwargs)
