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
    for name in (*RETRY_BACKOFF_CONSTANTS, "_SCAN_DEADLINE"):
        assert isinstance(getattr(read, name), float), name
    assert read._RETRY_MAX_BACKOFF < read._SCAN_DEADLINE


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


@pytest.mark.filterwarnings("ignore:scan_bucket. Hub rate limit:UserWarning")
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


def test_long_rate_limit_reset_is_waited_for_with_a_warning(clock: FakeClock) -> None:
    # "Reset in 300 s" fits in the 600 s of one call: wait, and say so.
    handler, seen = _answers(Response(429, headers=_rate_limited(300)))

    with hub_session(handler):
        with pytest.warns(UserWarning, match="waiting 301 s") as caught:
            assert _resolve() == SIGNED

    assert clock.waits == [301.0]
    assert len(seen) == 2
    assert "rate limit" in str(caught[0].message)
    assert "'ns/name'" in str(caught[0].message)


def test_short_wait_gives_no_warning(clock: FakeClock) -> None:
    import warnings

    handler, _ = _answers(Response(429, headers=_rate_limited(3)))

    with hub_session(handler):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _resolve() == SIGNED

    assert clock.waits == [4.0]


def test_second_long_reset_passes_the_deadline_and_raises(clock: FakeClock) -> None:
    handler, seen = _answers(*[Response(429, headers=_rate_limited(300))] * 10)

    with hub_session(handler):
        with pytest.warns(UserWarning, match="waiting 301 s"):
            with pytest.raises(HfHubHTTPError) as error:
                _resolve()

    # One wait of 301 s fits in 600 s; the second one does not.
    assert clock.waits == [301.0]
    assert len(seen) == 2
    message = str(error.value)
    assert "rate limit for resolve requests was reached" in message
    assert "5000 requests per 300 s" in message
    assert "'ns/name'" in message
    assert "the Hub asks to wait 301 s" in message
    assert "only 299 s are left of the 600 s" in message
    assert error.value.response.status_code == 429


def test_wait_longer_than_the_deadline_raises_without_waiting(
    clock: FakeClock,
) -> None:
    # huggingface_hub 2.0 sleeps for the full Retry-After of any status.
    handler, seen = _answers(Response(503, headers={"retry-after": "86400"}))

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError) as error:
            _resolve()

    assert clock.waits == []
    assert len(seen) == 1
    message = str(error.value)
    assert "HTTP 503 to a resolve request" in message
    assert "'ns/name'" in message
    assert "the Hub asks to wait 86400 s" in message
    assert "600 s allowed for one scan_bucket call" in message
    assert error.value.response.status_code == 503


def test_waits_stop_at_the_deadline_of_the_call(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(read, "_SCAN_DEADLINE", 100.0)
    handler, seen = _answers(*[Response(429, headers=_rate_limited(39))] * 10)

    with hub_session(handler):
        with pytest.warns(UserWarning, match="waiting 40 s"):
            with pytest.raises(HfHubHTTPError) as error:
                _resolve()

    # Two waits of 40 s fit in 100 s; a third one does not.
    assert clock.waits == [40.0, 40.0]
    assert len(seen) == 3
    assert "only 20 s are left of the 100 s" in str(error.value)
    assert "rate limit for resolve requests was reached" in str(error.value)


def test_backoff_stops_at_the_deadline_of_the_call(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(read, "_SCAN_DEADLINE", 5.0)
    handler, seen = _answers(*[Response(503)] * 10)

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError) as error:
            _resolve()

    assert clock.waits == [1.0, 2.0]
    assert "the next retry is in 4 s, but only 2 s are left" in str(error.value)


@pytest.mark.filterwarnings("ignore:scan_bucket. Hub rate limit:UserWarning")
def test_total_wait_of_one_call_is_bounded(clock: FakeClock) -> None:
    handler, seen = _answers(*[Response(429, headers=_rate_limited(59))] * 100)

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError, match="no success after 5 retries"):
            _resolve()

    assert clock.waits == [60.0] * read._MAX_RETRIES
    assert len(seen) == read._MAX_RETRIES + 1


def test_deadline_is_shared_by_the_requests_of_one_call(clock: FakeClock) -> None:
    budget = _Budget("ns/name")
    handler, _ = _answers(Response(503, headers={"retry-after": "5"}))
    with hub_session(handler):
        _signed_url(RESOLVE, {}, uri=URI, budget=budget)
    # Time passes in the same call (other files, the listing).
    clock.now += read._SCAN_DEADLINE - 60

    handler, seen = _answers(Response(503, headers={"retry-after": "58"}))
    with hub_session(handler):
        with pytest.raises(HfHubHTTPError, match="only 55 s are left"):
            _signed_url(RESOLVE, {}, uri=URI, budget=budget)
    assert len(seen) == 1


# ---- listing pages ---------------------------------------------------------

ENDPOINT = "https://huggingface.co"
TREE = f"{ENDPOINT}/api/buckets/ns/name/tree/data%2F"


def _page(paths: list[str], next_url: str | None = None):
    items = [
        {"type": "file", "path": path, "size": 10, "xetHash": "a" * 64}
        for path in paths
    ]
    headers = {}
    if next_url is not None:
        headers["link"] = f'<{next_url}>; rel="next"'
    return Response(200, json=items, headers=headers)


def _list(recursive: bool = True) -> list:
    return read._list_tree(
        ENDPOINT,
        {"authorization": "Bearer token"},
        "ns/name",
        "data/",
        recursive=recursive,
        uri="hf://buckets/ns/name/data/",
        budget=_Budget("ns/name"),
    )


def test_listing_request_and_pages(clock: FakeClock) -> None:
    seen = []
    second = f"{TREE}?recursive=true&cursor=abc"

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return _page(["data/a.parquet", "data/b.parquet"], second)
        return _page(["data/c.parquet"])

    with hub_session(handler):
        entries = _list()

    assert [entry.path for entry in entries] == [
        "data/a.parquet",
        "data/b.parquet",
        "data/c.parquet",
    ]
    assert entries[0] == read._Entry("file", "data/a.parquet", 10, "a" * 64)
    # The prefix is one percent-encoded segment; the next link is used as is.
    assert [str(request.url) for request in seen] == [
        f"{TREE}?recursive=true",
        second,
    ]
    assert all(r.method == "GET" for r in seen)
    assert all(r.headers["authorization"] == "Bearer token" for r in seen)
    assert clock.waits == []


def test_non_recursive_listing_parameter(clock: FakeClock) -> None:
    seen = []

    def handler(request):
        seen.append(request)
        return _page([])

    with hub_session(handler):
        assert _list(recursive=False) == []

    assert str(seen[0].url) == f"{TREE}?recursive=false"


@pytest.mark.parametrize("status", [429, 503])
def test_every_listing_page_is_retried(clock: FakeClock, status: int) -> None:
    seen = []
    second = f"{TREE}?recursive=true&cursor=abc"

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return _page(["data/a.parquet"], second)
        if len(seen) in (2, 3):
            return Response(status)
        return _page(["data/b.parquet"])

    with hub_session(handler):
        entries = _list()

    assert [entry.path for entry in entries] == ["data/a.parquet", "data/b.parquet"]
    # Only the failed page is requested again.
    assert [str(r.url) for r in seen] == [f"{TREE}?recursive=true"] + [second] * 3
    assert clock.waits == [1.0, 2.0]


def test_later_listing_page_wait_is_bounded(clock: FakeClock) -> None:
    # huggingface_hub's own paginate sleeps for the full announced time here.
    seen = []
    second = f"{TREE}?recursive=true&cursor=abc"

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return _page(["data/a.parquet"], second)
        return Response(429, headers=_rate_limited(900))

    with hub_session(handler):
        with pytest.raises(HfHubHTTPError) as error:
            _list()

    assert clock.waits == []
    assert len(seen) == 2
    assert "rate limit for listing requests was reached" in str(error.value)
    assert "the Hub asks to wait 901 s" in str(error.value)


def test_next_page_on_another_origin_is_not_requested(clock: FakeClock) -> None:
    seen = []

    def handler(request):
        seen.append(request)
        return _page(["data/a.parquet"], "https://evil.example/api/next?cursor=1")

    with hub_session(handler):
        with pytest.raises(RuntimeError, match="links to another origin"):
            _list()

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
        HUB, "HEAD", r"/resolve/data/p3\.parquet$", 429, headers=_rate_limited(900)
    )

    with pytest.raises(HfHubHTTPError) as error:
        plhf.scan_bucket(_uri(fake_bucket, "data/"))

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
    fake_hub.add_fault(HUB, "GET", r"/tree/data/$", status, times=2)

    plhf.scan_bucket(_uri(fake_bucket, "data/"))

    listings = fake_hub.matching(origin=HUB, method="GET")
    assert [request.status for request in listings] == [status, status, 200]
    # The fake_hub fixture shortens the backoff; two waits were made.
    assert len(clock.waits) == 2


def test_rate_limited_listing_raises_without_waiting(
    fake_hub: FakeHub, fake_bucket: str, clock: FakeClock
) -> None:
    _put(fake_hub, fake_bucket, "data/a.parquet")
    fake_hub.add_fault(HUB, "GET", r"/tree/data/$", 429, headers=_rate_limited(900))

    with pytest.raises(HfHubHTTPError) as error:
        plhf.scan_bucket(_uri(fake_bucket, "data/"))

    assert "rate limit for listing requests was reached" in str(error.value)
    assert f"'{fake_bucket}'" in str(error.value)
    assert clock.waits == []
    assert len(fake_hub.matching(origin=HUB)) == 1


def test_listing_retries_are_bounded(
    fake_hub: FakeHub, fake_bucket: str, clock: FakeClock
) -> None:
    _put(fake_hub, fake_bucket, "data/a.parquet")
    fake_hub.add_fault(HUB, "GET", r"/tree/data/$", 503, times=100)

    with pytest.raises(HfHubHTTPError) as error:
        plhf.scan_bucket(_uri(fake_bucket, "data/"))

    assert error.value.response.status_code == 503
    assert len(fake_hub.matching(origin=HUB)) == read._MAX_RETRIES + 1
