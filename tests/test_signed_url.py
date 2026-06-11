"""Unit tests for ``_signed_url`` (no network — uses ``httpx.MockTransport``)."""

from __future__ import annotations

import httpx
import pytest

from polars_hf.read import _signed_url

RESOLVE = "https://huggingface.co/buckets/ns/name/resolve/data.parquet"
SIGNED = "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc?X-Amz-Signature=sig"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_absolute_redirect_returns_signed_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"  # must never GET file bytes
        return httpx.Response(302, headers={"location": SIGNED})

    with _client(handler) as client:
        assert _signed_url(client, {}, RESOLVE) == SIGNED


def test_relative_redirect_followed_with_auth() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(
                307, headers={"location": "/buckets/ns/name/resolve2/data.parquet"}
            )
        return httpx.Response(302, headers={"location": SIGNED})

    headers = {"authorization": "Bearer hf_test"}
    with _client(handler) as client:
        assert _signed_url(client, headers, RESOLVE) == SIGNED

    # The relative hop stays on the Hub and keeps the auth header.
    assert (
        str(seen[1].url)
        == "https://huggingface.co/buckets/ns/name/resolve2/data.parquet"
    )
    assert seen[1].headers["authorization"] == "Bearer hf_test"


def test_direct_200_returns_resolve_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    with _client(handler) as client:
        assert _signed_url(client, {}, RESOLVE) == RESOLVE


def test_error_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            _signed_url(client, {}, RESOLVE)


def test_redirect_loop_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/loop"})

    with _client(handler) as client:
        with pytest.raises(RuntimeError, match="too many redirects"):
            _signed_url(client, {}, RESOLVE)
