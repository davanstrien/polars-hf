"""Scan parquet from Hugging Face buckets as a native polars ``LazyFrame``.

Bucket files are XET-backed: the Hub ``resolve`` URL 302-redirects (when
requested with auth) to a presigned ``cas-bridge.xethub.hf.co`` URL that needs
no auth and supports HTTP range requests. We follow that redirect in Python and
hand the **signed URLs** to native :func:`polars.scan_parquet`, so polars' Rust
object store does async, concurrent, range-read scans with full projection /
predicate / slice pushdown — the same mechanism the upstream ``hf://`` reader
uses, but reachable from stock polars.

Stock polars cannot authenticate a generic ``https://`` URL itself (bearer-token
injection is gated behind the ``hf://`` scheme), which is why we resolve the
signed URL here rather than passing the ``resolve`` URL directly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

import httpx
import polars as pl
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import build_hf_headers

from polars_hf._uri import parse_bucket_uri

# Signed URLs are valid for ~1 hour (X-Amz-Expires=3600); resolve at scan time.
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_MAX_RESOLVE_WORKERS = 16
_MAX_REDIRECT_HOPS = 5


def _list_files(fs: HfFileSystem, uri: str) -> list[str]:
    """Resolve a bucket URI to a sorted list of parquet file paths."""
    bp = parse_bucket_uri(uri)

    # Keep in sync with the parquet extensions accepted by sink_bucket
    # (write._EXT_FORMAT, which matches case-insensitively): a file written
    # as .pq / .PARQUET must scan back as a single file.
    if not bp.is_glob and bp.path.lower().endswith((".parquet", ".pq")):
        return [bp.fs_path]

    pattern = bp.fs_path if bp.is_glob else f"{bp.fs_path.rstrip('/')}/**/*.parquet"
    # Avoid a stale dircache when scanning right after an in-process write.
    fs.invalidate_cache()
    files = fs.glob(pattern)

    if not files:
        raise FileNotFoundError(f"no parquet files matched: {uri!r}")
    return sorted(files)


def _signed_url(client: httpx.Client, headers: dict, resolve_url: str) -> str:
    """Follow the authenticated resolve redirect to a range-readable signed URL.

    Uses HEAD — the same way ``HfApi.get_bucket_file_metadata`` probes this
    endpoint — so no file bytes are transferred (a plain GET would download the
    whole file when the server serves it directly with a 200). Redirects that
    stay on the Hub origin (relative, or absolute with the same scheme, host
    and port) are followed with auth; the first ``location`` on another origin
    is the presigned CDN URL, readable without auth. The auth header is never
    sent to another origin — including a scheme downgrade on the same host.
    """
    hub = urlparse(resolve_url)
    hub_origin = (hub.scheme, hub.hostname, hub.port)
    url = resolve_url
    for _ in range(_MAX_REDIRECT_HOPS):
        r = client.head(url, headers=headers)
        if r.status_code in _REDIRECT_CODES and "location" in r.headers:
            # urljoin resolves relative *and* protocol-relative (//host/..)
            # locations; compare origins rather than sniffing the scheme prefix.
            # (.hostname is lower-cased by urlparse, so host case is ignored.)
            location = urljoin(url, r.headers["location"])
            target = urlparse(location)
            if (target.scheme, target.hostname, target.port) != hub_origin:
                return location
            url = location
            continue
        if r.status_code == 200:
            # Served directly (e.g. a public file): the URL is itself
            # readable without auth.
            return url
        r.raise_for_status()
        return url  # pragma: no cover - raise_for_status covers error codes
    raise RuntimeError(f"too many redirects while resolving {resolve_url!r}")


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
        An ``hf://buckets/{namespace}/{name}/{path}`` URI. ``path`` may be a
        single parquet file (``.parquet`` / ``.pq``), a glob (e.g.
        ``data/*.parquet``), or a directory / the whole bucket (expanded to
        ``**/*.parquet``).
    token
        Hugging Face token. If ``None``, resolved by ``huggingface_hub`` (the
        ``HF_TOKEN`` env var or cached login).
    **scan_kwargs
        Forwarded to :func:`polars.scan_parquet` — e.g. ``retries=`` for flaky
        connections, ``missing_columns="insert"`` / ``extra_columns="ignore"``
        for heterogeneous schemas across globbed files, ``schema=``, or
        ``cast_options=``. Options that derive meaning from the file *path*
        (``hive_partitioning=``, ``include_file_paths=``) see the presigned
        CDN URLs, not the bucket paths, so they are not useful here.

    Returns
    -------
    LazyFrame

    Notes
    -----
    Signed URLs are resolved when ``scan_bucket`` is called and are valid for
    ~1 hour. Collect within that window; for long-lived plans, call
    ``scan_bucket`` again to refresh.

    Examples
    --------
    >>> import polars_hf as plhf
    >>> lf = plhf.scan_bucket("hf://buckets/me/data/*.parquet")  # doctest: +SKIP
    >>> lf.filter(pl.col("label") == 1).head(5).collect()  # doctest: +SKIP
    """
    fs = HfFileSystem(token=token)
    files = _list_files(fs, uri)
    headers = build_hf_headers(token=token)

    with httpx.Client(follow_redirects=False, timeout=30) as client:
        if len(files) == 1:
            urls = [_signed_url(client, headers, fs.url(files[0]))]
        else:
            with ThreadPoolExecutor(
                max_workers=min(_MAX_RESOLVE_WORKERS, len(files))
            ) as pool:
                urls = list(
                    pool.map(
                        lambda p: _signed_url(client, headers, fs.url(p)),
                        files,
                    )
                )

    return pl.scan_parquet(urls, **scan_kwargs)
