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


def test_same_host_absolute_redirect_followed_with_auth() -> None:
    # An absolute redirect that stays on the Hub host is not the CDN URL: keep
    # following it (with auth) rather than handing polars an auth-only URL.
    seen: list[httpx.Request] = []
    moved = "https://huggingface.co/buckets/ns/renamed/resolve/data.parquet"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(301, headers={"location": moved})
        return httpx.Response(302, headers={"location": SIGNED})

    headers = {"authorization": "Bearer hf_test"}
    with _client(handler) as client:
        assert _signed_url(client, headers, RESOLVE) == SIGNED
    assert str(seen[1].url) == moved
    assert seen[1].headers["authorization"] == "Bearer hf_test"


def test_same_host_scheme_downgrade_is_terminal() -> None:
    # Same host but http:// is a different origin: never re-send the Bearer
    # header in cleartext. Stop, as the old scheme-prefix rule did.
    seen: list[httpx.Request] = []
    downgraded = "http://huggingface.co/buckets/ns/name/resolve/data.parquet"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(302, headers={"location": downgraded})

    with _client(handler) as client:
        got = _signed_url(client, {"authorization": "Bearer hf_test"}, RESOLVE)
    assert got == downgraded
    assert len(seen) == 1


def test_same_host_case_insensitive() -> None:
    # Host labels are case-insensitive: an absolute redirect to HuggingFace.co
    # is still the Hub origin and is followed with auth.
    seen: list[httpx.Request] = []
    mixed = "https://HuggingFace.co/buckets/ns/name/resolve2/data.parquet"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": mixed})
        return httpx.Response(302, headers={"location": SIGNED})

    with _client(handler) as client:
        assert _signed_url(client, {"authorization": "Bearer x"}, RESOLVE) == SIGNED
    assert seen[1].headers["authorization"] == "Bearer x"


def test_protocol_relative_redirect_is_terminal() -> None:
    # "//host/path" is off-host: return it resolved, and never send the auth
    # header to that host.
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            302, headers={"location": "//cas-bridge.xethub.hf.co/xet-bridge-us/abc"}
        )

    with _client(handler) as client:
        got = _signed_url(client, {"authorization": "Bearer hf_test"}, RESOLVE)
    assert got == "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc"
    assert len(seen) == 1


def test_multiple_relative_hops() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) < 3:
            return httpx.Response(307, headers={"location": f"/hop{len(seen)}"})
        return httpx.Response(302, headers={"location": SIGNED})

    with _client(handler) as client:
        assert _signed_url(client, {}, RESOLVE) == SIGNED
    assert [str(r.url) for r in seen[1:]] == [
        "https://huggingface.co/hop1",
        "https://huggingface.co/hop2",
    ]


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
