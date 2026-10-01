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
``HfApi.list_bucket_tree`` and the ``resolve`` request through
``huggingface_hub.utils.http_backoff``, which uses the shared Hub session
(proxies, custom client factory) and retries 429 and 5xx answers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urljoin, urlparse

import polars as pl
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status, http_backoff

from polars_hf._glob import glob_to_regex, literal_prefix
from polars_hf._uri import BucketPath, parse_bucket_uri

# The signed URL carries its own expiry (an ``Expires`` query parameter, about
# one hour after the resolve request), so URLs are resolved at scan time.
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_MAX_RESOLVE_WORKERS = 16
_MAX_REDIRECT_HOPS = 5

# Keep in sync with the parquet extensions accepted by sink_bucket
# (write._EXT_FORMAT, which matches case-insensitively).
_PARQUET_SUFFIXES = (".parquet", ".pq")

# Retry policy of one resolve request (passed to http_backoff): the wait
# doubles from the base up to the maximum; on a 429 with rate-limit headers
# http_backoff waits for the announced reset instead.
_RESOLVE_TIMEOUT = 30
_RESOLVE_MAX_RETRIES = 5
_RESOLVE_BASE_WAIT = 1.0
_RESOLVE_MAX_WAIT = 8.0


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


def _list_bucket_files(api: HfApi, bucket_id: str, prefix: str, uri: str) -> list:
    """List the files of a bucket whose path starts with ``prefix``.

    One recursive ``list_bucket_tree`` call (paginated by the client). The Hub
    matches ``prefix`` as a plain string, so the result can hold siblings such
    as ``data.parquet`` and ``data2/x`` for the prefix ``data``; the caller
    filters. Only entries of type ``file`` are returned.
    """
    try:
        entries = list(
            api.list_bucket_tree(bucket_id, prefix=prefix or None, recursive=True)
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
        raise

    files = []
    for entry in entries:
        if entry.type == "file":
            files.append(entry)
    return files


def _select_files(files: list, bp: BucketPath) -> list:
    """Pick the listed files that the path of ``bp`` names.

    The path is tried, in order, as: the exact name of a file; a directory
    (all parquet files below it); a glob. A name that contains glob characters
    (``data[1].parquet``) is therefore read literally when such a file or
    directory exists.
    """
    path = bp.path.rstrip("/")
    names_directory = bp.path.endswith("/")

    if not names_directory:
        for file in files:
            if file.path == path:
                return [file]

    directory_prefix = f"{path}/" if path else ""
    selected = []
    for file in files:
        if file.path.startswith(directory_prefix) and _is_parquet_name(file.path):
            selected.append(file)
    if selected or not bp.is_glob:
        return selected

    pattern = glob_to_regex(path)
    for file in files:
        if pattern.fullmatch(file.path):
            selected.append(file)
    return selected


def _list_files(api: HfApi, bp: BucketPath, uri: str) -> list[str]:
    """List the bucket and return the sorted paths of the files ``bp`` names."""
    path = bp.path.rstrip("/")
    # Every candidate (the literal file, the files below the literal
    # directory, the glob matches) starts with the text before the first glob
    # character, so one listing covers all three readings of the path.
    prefix = literal_prefix(path) if bp.is_glob else path

    files = _list_bucket_files(api, bp.bucket_id, prefix, uri)
    selected = _select_files(files, bp)
    if not selected:
        raise FileNotFoundError(f"no parquet files matched: {uri!r}")

    for file in selected:
        if file.size == 0:
            raise _empty_file_error(_file_uri(bp.bucket_id, file.path))
    return sorted(file.path for file in selected)


def _resolve_url(endpoint: str, bucket_id: str, path: str) -> str:
    """The Hub ``resolve`` URL of a bucket file (the one ``HfApi`` requests)."""
    return f"{endpoint}/buckets/{bucket_id}/resolve/{quote(path, safe='')}"


def _head(url: str, headers: dict[str, str]):
    """One HEAD request with auth that does not follow redirects.

    ``http_backoff`` retries 429 and 5xx answers; it raises ``HfHubHTTPError``
    when the last attempt still fails with one of those status codes.

    Timeouts and network errors are not retried and keep their type. On a
    connection error ``http_backoff`` closes the session that all resolve
    threads share, and huggingface_hub 1.x then retries on the closed client
    (``RuntimeError``).
    """
    return http_backoff(
        "HEAD",
        url,
        headers=headers,
        follow_redirects=False,
        timeout=_RESOLVE_TIMEOUT,
        max_retries=_RESOLVE_MAX_RETRIES,
        base_wait_time=_RESOLVE_BASE_WAIT,
        max_wait_time=_RESOLVE_MAX_WAIT,
        retry_on_exceptions=(),
    )


def _signed_url(
    resolve_url: str, headers: dict[str, str], *, bucket_id: str, uri: str
) -> str:
    """Follow the authenticated resolve redirect to a range-readable signed URL.

    Uses HEAD — the same way ``HfApi.get_bucket_file_metadata`` probes this
    endpoint — so no file bytes are transferred. Redirects that stay on the
    Hub origin (relative, or absolute with the same scheme, host and port) are
    followed with auth; the first ``location`` on another origin is the
    presigned CDN URL, readable without auth. The auth header is never sent to
    another origin — including a scheme downgrade on the same host.

    ``bucket_id`` and ``uri`` are used in error messages only.

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
        Any other HTTP error, after the retries for 429 and 5xx.
    """
    hub = urlparse(resolve_url)
    hub_origin = (hub.scheme, hub.hostname, hub.port)
    url = resolve_url
    for _ in range(_MAX_REDIRECT_HOPS):
        response = _head(url, headers)
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
                raise _no_access_error(bucket_id, uri, status) from error
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
    endpoint: str, headers: dict[str, str], bucket_id: str, paths: list[str]
) -> list[str]:
    """Resolve every path to its signed URL, one HEAD request per file."""

    def resolve(path: str) -> str:
        return _signed_url(
            _resolve_url(endpoint, bucket_id, path),
            headers,
            bucket_id=bucket_id,
            uri=_file_uri(bucket_id, path),
        )

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
          ``?`` and ``[...]`` match inside one path segment, ``**`` matches
          any number of directories. Directories are never passed to the scan.
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
        The URI is not a valid bucket URI, or a matched file is empty
        (0 bytes).
    FileNotFoundError
        The bucket does not exist, or no file matches ``uri``.
    PermissionError
        The Hub answers 401 or 403: the token is not valid or lacks access to
        the bucket. The original ``HfHubHTTPError`` is the ``__cause__``.
    huggingface_hub.errors.HfHubHTTPError
        Any other HTTP error of the Hub. Rate-limit (429) and server (5xx)
        answers are retried first, with the backoff of ``huggingface_hub``.

    Notes
    -----
    ``scan_bucket`` makes these Hub requests and reads no file data:

    * a path that ends in ``.parquet`` / ``.pq`` and has no glob character:
      one ``resolve`` request (HEAD). If the Hub answers "not found", the path
      is then handled as a directory;
    * any other path: one listing request per page of results, then one
      ``resolve`` request per file. ``resolve`` requests count in the Hub's
      "resolvers" rate limit.

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

    # The common case, one parquet file, needs no listing: ask for its signed
    # URL directly. The suffix only chooses which request is tried first; a
    # "not found" answer falls through to the listing, which decides.
    if not bp.is_glob and _is_parquet_name(bp.path):
        try:
            urls = _signed_urls(api.endpoint, headers, bp.bucket_id, [bp.path])
        except FileNotFoundError:
            pass
        else:
            return pl.scan_parquet(urls, **scan_kwargs)

    paths = _list_files(api, bp, uri)
    urls = _signed_urls(api.endpoint, headers, bp.bucket_id, paths)
    return pl.scan_parquet(urls, **scan_kwargs)
