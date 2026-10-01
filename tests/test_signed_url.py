"""Unit tests for ``_signed_url`` (no network).

``_signed_url`` sends its requests through the shared ``huggingface_hub``
session; ``conftest.hub_session`` answers them with a handler function.
"""

from __future__ import annotations

import pytest
from conftest import fast_resolve_retries, hub_httpx, hub_session
from huggingface_hub.errors import HfHubHTTPError

from polars_hf import read
from polars_hf.read import _Budget, _signed_url

RESOLVE = "https://huggingface.co/buckets/ns/name/resolve/data.parquet"
SIGNED = "https://us.aws.cdn.hf.co/xet-bridge-us/abc?Expires=1&Signature=sig"
URI = "hf://buckets/ns/name/data.parquet"

Response = hub_httpx.Response


def _resolve(headers: dict[str, str] | None = None) -> str:
    return _signed_url(RESOLVE, headers or {}, uri=URI, budget=_Budget("ns/name"))


@pytest.fixture(autouse=True)
def _no_backoff_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    fast_resolve_retries(monkeypatch)


def test_absolute_redirect_returns_signed_url() -> None:
    def handler(request):
        assert request.method == "HEAD"  # must never GET file bytes
        return Response(302, headers={"location": SIGNED})

    with hub_session(handler):
        assert _resolve() == SIGNED


def test_relative_redirect_followed_with_auth() -> None:
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return Response(
                307, headers={"location": "/buckets/ns/name/resolve2/data.parquet"}
            )
        return Response(302, headers={"location": SIGNED})

    with hub_session(handler):
        assert _resolve({"authorization": "Bearer hf_test"}) == SIGNED

    # The relative hop stays on the Hub and keeps the auth header.
    assert (
        str(seen[1].url)
        == "https://huggingface.co/buckets/ns/name/resolve2/data.parquet"
    )
    assert seen[1].headers["authorization"] == "Bearer hf_test"


def test_same_host_absolute_redirect_followed_with_auth() -> None:
    # An absolute redirect that stays on the Hub host is not the CDN URL: keep
    # following it (with auth) rather than handing polars an auth-only URL.
    seen = []
    moved = "https://huggingface.co/buckets/ns/renamed/resolve/data.parquet"

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return Response(301, headers={"location": moved})
        return Response(302, headers={"location": SIGNED})

    with hub_session(handler):
        assert _resolve({"authorization": "Bearer hf_test"}) == SIGNED
    assert str(seen[1].url) == moved
    assert seen[1].headers["authorization"] == "Bearer hf_test"


def test_same_host_scheme_downgrade_is_terminal() -> None:
    # Same host but http:// is a different origin: never re-send the Bearer
    # header in cleartext. Stop, as the old scheme-prefix rule did.
    seen = []
    downgraded = "http://huggingface.co/buckets/ns/name/resolve/data.parquet"

    def handler(request):
        seen.append(request)
        return Response(302, headers={"location": downgraded})

    with hub_session(handler):
        got = _resolve({"authorization": "Bearer hf_test"})
    assert got == downgraded
    assert len(seen) == 1


def test_same_host_case_insensitive() -> None:
    # Host labels are case-insensitive: an absolute redirect to HuggingFace.co
    # is still the Hub origin and is followed with auth.
    seen = []
    mixed = "https://HuggingFace.co/buckets/ns/name/resolve2/data.parquet"

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return Response(302, headers={"location": mixed})
        return Response(302, headers={"location": SIGNED})

    with hub_session(handler):
        assert _resolve({"authorization": "Bearer x"}) == SIGNED
    assert seen[1].headers["authorization"] == "Bearer x"


def test_protocol_relative_redirect_is_terminal() -> None:
    # "//host/path" is off-host: return it resolved, and never send the auth
    # header to that host.
    seen = []

    def handler(request):
        seen.append(request)
        return Response(
            302, headers={"location": "//us.aws.cdn.hf.co/xet-bridge-us/abc"}
        )

    with hub_session(handler):
        got = _resolve({"authorization": "Bearer hf_test"})
    assert got == "https://us.aws.cdn.hf.co/xet-bridge-us/abc"
    assert len(seen) == 1


def test_multiple_relative_hops() -> None:
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) < 3:
            return Response(307, headers={"location": f"/hop{len(seen)}"})
        return Response(302, headers={"location": SIGNED})

    with hub_session(handler):
        assert _resolve() == SIGNED
    assert [str(r.url) for r in seen[1:]] == [
        "https://huggingface.co/hop1",
        "https://huggingface.co/hop2",
    ]


def test_redirect_loop_raises() -> None:
    def handler(request):
        return Response(302, headers={"location": "/loop"})

    with hub_session(handler):
        with pytest.raises(RuntimeError, match="too many redirects"):
            _resolve()


# ---- answers without a redirect --------------------------------------------


def test_direct_200_is_not_handed_to_polars() -> None:
    # The Hub serves a file that is not Xet-backed itself. That URL needs the
    # token, which polars cannot send: raise instead of returning it.
    def handler(request):
        return Response(200, headers={"content-length": "512"})

    with hub_session(handler):
        with pytest.raises(RuntimeError, match="did not redirect") as error:
            _resolve()
    assert URI in str(error.value)


@pytest.mark.parametrize("size_header", ["content-length", "x-linked-size"])
def test_direct_200_of_an_empty_file_names_the_file(size_header: str) -> None:
    def handler(request):
        return Response(200, headers={size_header: "0"})

    with hub_session(handler):
        with pytest.raises(ValueError, match="is empty") as error:
            _resolve()
    assert URI in str(error.value)


def test_redirect_of_an_empty_file_names_the_file() -> None:
    # The Hub CI instance redirects an empty file like any other one.
    def handler(request):
        return Response(302, headers={"location": SIGNED, "x-linked-size": "0"})

    with hub_session(handler):
        with pytest.raises(ValueError, match="is empty") as error:
            _resolve()
    assert URI in str(error.value)


# ---- errors ----------------------------------------------------------------


def test_404_raises_file_not_found() -> None:
    def handler(request):
        return Response(404, headers={"x-error-code": "EntryNotFound"})

    with hub_session(handler):
        with pytest.raises(FileNotFoundError) as error:
            _resolve()
    assert URI in str(error.value)
    assert isinstance(error.value.__cause__, HfHubHTTPError)


@pytest.mark.parametrize("status", [401, 403])
def test_refused_token_raises_permission_error(status: int) -> None:
    seen = []

    def handler(request):
        seen.append(request)
        return Response(status)

    with hub_session(handler):
        with pytest.raises(PermissionError) as error:
            _resolve({"authorization": "Bearer hf_test"})

    message = str(error.value)
    assert "'ns/name'" in message and "lacks access" in message
    assert URI in message
    assert "hf_test" not in message
    cause = error.value.__cause__
    assert isinstance(cause, HfHubHTTPError)
    assert cause.response.status_code == status
    assert len(seen) == 1  # not retried


@pytest.mark.parametrize("status", [400, 410])
def test_other_client_error_keeps_its_type(status: int) -> None:
    seen = []

    def handler(request):
        seen.append(request)
        return Response(status)

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError) as error:
            _resolve()
    assert error.value.response.status_code == status
    assert len(seen) == 1  # not retried


# ---- retries ---------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_is_retried(status: int) -> None:
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) <= 2:
            return Response(status)
        return Response(302, headers={"location": SIGNED})

    with hub_session(handler):
        assert _resolve({"authorization": "Bearer hf_test"}) == SIGNED
    assert len(seen) == 3
    assert all(r.headers["authorization"] == "Bearer hf_test" for r in seen)


def test_retries_are_bounded() -> None:
    seen = []

    def handler(request):
        seen.append(request)
        return Response(503)

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError) as error:
            _resolve()
    assert error.value.response.status_code == 503
    assert len(seen) == read._MAX_RETRIES + 1


def test_network_error_is_not_retried_and_keeps_its_type() -> None:
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            raise hub_httpx.ConnectError("scripted connection failure")
        return Response(302, headers={"location": SIGNED})

    with hub_session(handler):
        with pytest.raises(hub_httpx.ConnectError):
            _resolve()
        assert len(seen) == 1
        # The shared session was not closed: the next request goes through.
        assert _resolve() == SIGNED
