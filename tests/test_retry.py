"""Retry policy of the Hub requests of ``scan_bucket``: bounded waits (no network).

The clock of ``polars_hf.read`` is replaced: ``sleep`` records the wait and
moves the clock, so the tests check the waits without waiting.
"""

from __future__ import annotations

import pytest
from conftest import RETRY_BACKOFF_CONSTANTS, hub_httpx, hub_session
from fakehub import HUB, FakeHub
from huggingface_hub.errors import HfHubHTTPError

import polars_hf as plhf
from polars_hf import read
from polars_hf.read import _Budget, _signed_url

RESOLVE = "https://huggingface.co/buckets/ns/name/resolve/data.parquet"
SIGNED = "https://us.aws.cdn.hf.co/xet-bridge-us/abc?Expires=1&Signature=sig"
URI = "hf://buckets/ns/name/data.parquet"

Response = hub_httpx.Response


def _rate_limited(seconds: int) -> dict[str, str]:
    """The rate-limit headers of the Hub (IETF draft): reset in ``seconds``."""
    return {
        "ratelimit": f'"resolvers";r=0;t={seconds}',
        "ratelimit-policy": '"fixed window";"resolvers";q=5000;w=300',
    }


class FakeClock:
    """Stands in for the ``time`` module inside ``polars_hf.read``."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(read, "time", fake)
    return fake


def _resolve() -> str:
    return _signed_url(RESOLVE, {}, uri=URI, budget=_Budget("ns/name"))


def _answers(*responses):
    """A handler that gives ``responses`` in order, then the signed redirect."""
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) <= len(responses):
            return responses[len(seen) - 1]
        return Response(302, headers={"location": SIGNED})

    return handler, seen


def test_constants_patched_by_the_test_fixtures_exist() -> None:
    # conftest patches these names with raising=False.
    for name in (*RETRY_BACKOFF_CONSTANTS, "_MAX_WAIT_PER_RETRY", "_SCAN_DEADLINE"):
        assert isinstance(getattr(read, name), float), name
    assert read._RETRY_MAX_BACKOFF <= read._MAX_WAIT_PER_RETRY < read._SCAN_DEADLINE


def test_backoff_without_server_hint_doubles_to_the_maximum(clock: FakeClock) -> None:
    handler, seen = _answers(*[Response(503)] * 5)

    with hub_session(handler):
        assert _resolve() == SIGNED

    assert clock.waits == [1.0, 2.0, 4.0, 8.0, 8.0]
    assert len(seen) == 6


@pytest.mark.parametrize("status", [408, 429, 503])
def test_retry_after_is_honoured(clock: FakeClock, status: int) -> None:
    handler, seen = _answers(Response(status, headers={"retry-after": "3"}))

    with hub_session(handler):
        assert _resolve() == SIGNED

    assert clock.waits == [3.0]
    assert len(seen) == 2


def test_rate_limit_reset_is_honoured(clock: FakeClock) -> None:
    handler, seen = _answers(Response(429, headers=_rate_limited(7)))

    with hub_session(handler):
        assert _resolve() == SIGNED

    # One second more than announced: the reset time is rounded down.
    assert clock.waits == [8.0]
    assert len(seen) == 2


def test_retry_after_http_date_falls_back_to_the_backoff(clock: FakeClock) -> None:
    date = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    handler, _ = _answers(Response(503, headers=date))

    with hub_session(handler):
        assert _resolve() == SIGNED

    assert clock.waits == [1.0]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "soon", ""])
def test_non_finite_retry_after_falls_back_to_the_backoff(
    clock: FakeClock, value: str
) -> None:
    handler, seen = _answers(Response(503, headers={"retry-after": value}))

    with hub_session(handler):
        assert _resolve() == SIGNED

    assert clock.waits == [1.0]
    assert len(seen) == 2


def test_negative_retry_after_is_no_wait(clock: FakeClock) -> None:
    handler, _ = _answers(Response(503, headers={"retry-after": "-5"}))

    with hub_session(handler):
        assert _resolve() == SIGNED

    assert clock.waits == [0.0]


def test_rate_limit_reset_above_the_cap_raises_without_waiting(
    clock: FakeClock,
) -> None:
    handler, seen = _answers(*[Response(429, headers=_rate_limited(300))] * 10)

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError) as error:
            _resolve()

    assert clock.waits == []
    assert len(seen) == 1
    message = str(error.value)
    assert "rate limit for resolve requests was reached" in message
    assert "5000 requests per 300 s" in message
    assert "'ns/name'" in message
    assert "wait 301 s" in message and "60 s per retry" in message
    assert error.value.response.status_code == 429


def test_retry_after_above_the_cap_raises_without_waiting(clock: FakeClock) -> None:
    # huggingface_hub 2.0 sleeps for the full Retry-After of any status.
    handler, seen = _answers(Response(503, headers={"retry-after": "86400"}))

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError) as error:
            _resolve()

    assert clock.waits == []
    assert len(seen) == 1
    assert "HTTP 503 to a resolve request" in str(error.value)
    assert "'ns/name'" in str(error.value)
    assert error.value.response.status_code == 503


def test_wait_at_the_cap_is_allowed(clock: FakeClock) -> None:
    handler, _ = _answers(Response(429, headers=_rate_limited(59)))

    with hub_session(handler):
        assert _resolve() == SIGNED

    assert clock.waits == [60.0]


def test_waits_stop_at_the_deadline_of_the_call(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(read, "_SCAN_DEADLINE", 100.0)
    handler, seen = _answers(*[Response(429, headers=_rate_limited(39))] * 10)

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError) as error:
            _resolve()

    # Two waits of 40 s fit in 100 s; a third one does not.
    assert clock.waits == [40.0, 40.0]
    assert len(seen) == 3
    assert "limit of 100 s for one scan_bucket call" in str(error.value)
    assert "rate limit for resolve requests was reached" in str(error.value)


def test_total_wait_of_one_call_is_bounded(clock: FakeClock) -> None:
    # Every answer asks for the longest wait that is allowed.
    handler, seen = _answers(*[Response(429, headers=_rate_limited(59))] * 100)

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError, match="no success after 5 retries"):
            _resolve()

    assert clock.waits == [60.0] * read._MAX_RETRIES
    assert len(seen) == read._MAX_RETRIES + 1


def test_deadline_is_shared_by_the_requests_of_one_call(clock: FakeClock) -> None:
    budget = _Budget("ns/name")
    handler, _ = _answers(Response(503, headers={"retry-after": "50"}))
    with hub_session(handler):
        _signed_url(RESOLVE, {}, uri=URI, budget=budget)
    # Time passes in the same call (other files, the listing).
    clock.now += read._SCAN_DEADLINE - 60

    handler, seen = _answers(Response(503, headers={"retry-after": "50"}))
    with hub_session(handler):
        with pytest.raises(HfHubHTTPError, match="for one scan_bucket call"):
            _signed_url(RESOLVE, {}, uri=URI, budget=budget)
    assert len(seen) == 1


# ---- through scan_bucket, on the fake Hub ----------------------------------


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _put(fake_hub: FakeHub, bucket_id: str, path: str) -> None:
    import polars as pl

    fake_hub.put_parquet(bucket_id, path, pl.DataFrame({"a": [1]}))


def test_rate_limited_scan_reports_the_files_resolved(
    fake_hub: FakeHub, fake_bucket: str, clock: FakeClock
) -> None:
    n_files = 4
    for i in range(n_files):
        _put(fake_hub, fake_bucket, f"data/p{i}.parquet")
    fake_hub.add_fault(
        HUB, "HEAD", r"/resolve/data/p3\.parquet$", 429, headers=_rate_limited(300)
    )

    with pytest.raises(HfHubHTTPError) as error:
        plhf.scan_bucket(_uri(fake_bucket, "data"))

    message = str(error.value)
    assert "rate limit for resolve requests was reached" in message
    assert "5000 requests per 300 s" in message
    assert f"'{fake_bucket}'" in message
    assert f"of {n_files} files were resolved" in message
    assert clock.waits == []


@pytest.mark.parametrize("status", [429, 500, 503])
def test_listing_is_retried(
    fake_hub: FakeHub, fake_bucket: str, clock: FakeClock, status: int
) -> None:
    _put(fake_hub, fake_bucket, "data/a.parquet")
    fake_hub.add_fault(HUB, "GET", r"/tree/data$", status, times=2)

    plhf.scan_bucket(_uri(fake_bucket, "data"))

    listings = fake_hub.matching(origin=HUB, method="GET")
    assert [request.status for request in listings] == [status, status, 200]
    # The fake_hub fixture shortens the backoff; two waits were made.
    assert len(clock.waits) == 2


def test_rate_limited_listing_raises_without_waiting(
    fake_hub: FakeHub, fake_bucket: str, clock: FakeClock
) -> None:
    _put(fake_hub, fake_bucket, "data/a.parquet")
    fake_hub.add_fault(HUB, "GET", r"/tree/data$", 429, headers=_rate_limited(300))

    with pytest.raises(HfHubHTTPError) as error:
        plhf.scan_bucket(_uri(fake_bucket, "data"))

    assert "rate limit for listing requests was reached" in str(error.value)
    assert f"'{fake_bucket}'" in str(error.value)
    assert clock.waits == []
    assert len(fake_hub.matching(origin=HUB)) == 1


def test_listing_retries_are_bounded(
    fake_hub: FakeHub, fake_bucket: str, clock: FakeClock
) -> None:
    _put(fake_hub, fake_bucket, "data/a.parquet")
    fake_hub.add_fault(HUB, "GET", r"/tree/data$", 503, times=100)

    with pytest.raises(HfHubHTTPError) as error:
        plhf.scan_bucket(_uri(fake_bucket, "data"))

    assert error.value.response.status_code == 503
    assert len(fake_hub.matching(origin=HUB)) == read._MAX_RETRIES + 1
