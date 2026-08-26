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

import httpx
import polars as pl
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import build_hf_headers

from polars_hf._uri import parse_bucket_uri

# Signed URLs are valid for ~1 hour (X-Amz-Expires=3600); resolve at scan time.
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_MAX_RESOLVE_WORKERS = 16


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


def _signed_url(
    client: httpx.Client, fs: HfFileSystem, headers: dict, fs_path: str
) -> str:
    """Follow the authenticated resolve redirect to a range-readable signed URL."""
    resolve_url = fs.url(fs_path)
    r = client.get(resolve_url, headers=headers)
    if r.status_code in _REDIRECT_CODES and "location" in r.headers:
        return r.headers["location"]
    if r.status_code == 200:
        # Served directly (e.g. a public file): the resolve URL is itself
        # readable without auth.
        return resolve_url
    r.raise_for_status()
    return resolve_url  # pragma: no cover - raise_for_status covers error codes


def scan_bucket(uri: str, *, token: str | None = None) -> pl.LazyFrame:
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
            urls = [_signed_url(client, fs, headers, files[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=min(_MAX_RESOLVE_WORKERS, len(files))
            ) as pool:
                urls = list(
                    pool.map(
                        lambda p: _signed_url(client, fs, headers, p),
                        files,
                    )
                )

    return pl.scan_parquet(urls)
