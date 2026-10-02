"""``scan_bucket`` with ``resolve="redirect"`` (the default) and its local server.

Offline, against the fake Hub. In this mode the LazyFrame is the native
``scan_parquet`` node over ``http://127.0.0.1:{port}/{id}/...`` URLs, and the
server of ``polars_hf._redirect`` answers each request with a redirect to the
presigned URL. ``resolve="now"`` (the native node over the presigned URLs
themselves) is the reference for the rows.

The first part tests ``scan_bucket``; the second part sends requests to the
server directly, with resolver functions of the tests.
"""

from __future__ import annotations

import asyncio
import io
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path
from urllib.parse import quote, urlsplit

import polars as pl
import pytest
from conftest import assert_no_signed_url, exception_chain
from fakehub import CDN, HUB, SIGNATURE, FakeHub
from huggingface_hub.errors import HfHubHTTPError
from polars.testing import assert_frame_equal

import polars_hf as plhf
from polars_hf import _redirect, read

ROWS = 10
TESTS_DIRECTORY = str(Path(__file__).parent)
READ_ERRORS = (pl.exceptions.PolarsError, OSError)


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _numbered_frame(start: int, rows: int) -> pl.DataFrame:
    ids = pl.int_range(start, start + rows, eager=True)
    return pl.DataFrame({"id": ids, "label": (ids % 3).cast(pl.String)})


def _put_numbered(
    fake_hub: FakeHub, bucket_id: str, n_files: int, rows: int = ROWS
) -> pl.DataFrame:
    """Store ``data/p000.parquet``, ... and return their rows in file order."""
    frames = []
    for i in range(n_files):
        frame = _numbered_frame(i * rows, rows)
        fake_hub.put_parquet(bucket_id, f"data/p{i:03d}.parquet", frame)
        frames.append(frame)
    return pl.concat(frames)


def _wide_frame(start: int, rows: int) -> pl.DataFrame:
    """A frame whose ``payload`` column dominates the (uncompressed) file size."""
    ids = pl.int_range(start, start + rows, eager=True)
    payload = (ids * 2654435761 % 1000003).cast(pl.String) + "-" + ids.cast(pl.String)
    return pl.DataFrame({"id": ids, "payload": payload, "value": ids.cast(pl.Float64)})


def _resolved(fake_hub: FakeHub) -> list[str]:
    """The file names of the successful ``resolve`` requests, in arrival order."""
    names = []
    for request in fake_hub.matching(origin=HUB, method="HEAD"):
        if request.status == 302:
            names.append(request.path.rpartition("/")[2])
    return names


def _local_prefix() -> str:
    return f"http://127.0.0.1:{_redirect.get_server().port}/"


# ---- the modes -------------------------------------------------------------


def test_redirect_is_the_default_mode(fake_hub: FakeHub, fake_bucket: str) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 3)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))

    # The native scan node, over URLs of the local server.
    plan = lf.explain()
    assert "PYTHON" not in plan
    assert _local_prefix() in plan
    assert f"valid-only-while-pid-{os.getpid()}-runs" in plan
    assert_frame_equal(lf.collect(), expected)
    explicit = plhf.scan_bucket(_uri(fake_bucket, "data/"), resolve="redirect")
    assert_frame_equal(explicit.collect(), expected)


def test_now_mode_needs_the_acknowledgement_variable(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 2)
    uri = _uri(fake_bucket, "data/")

    for value in (None, "0", "true", ""):
        if value is None:
            monkeypatch.delenv("POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN")
        else:
            monkeypatch.setenv("POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN", value)
        with pytest.raises(ValueError) as error:
            plhf.scan_bucket(uri, resolve="now")
        message = str(error.value)
        assert "about 60 minutes" in message
        assert "explain(), serialize(), an error message or a log" in message
        assert "POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1" in message
    # Refused before any request.
    assert fake_hub.requests == []

    monkeypatch.setenv("POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN", "1")
    lf = plhf.scan_bucket(uri, resolve="now")
    assert SIGNATURE in lf.explain()
    assert SIGNATURE.encode() in lf.serialize(format="binary")
    assert_frame_equal(lf.collect(), expected)


def test_environment_variable_selects_the_mode(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put_numbered(fake_hub, fake_bucket, 2)
    uri = _uri(fake_bucket, "data/")

    monkeypatch.setenv("POLARS_HF_RESOLVE", "collect")
    assert "PYTHON" in plhf.scan_bucket(uri).explain()
    # resolve= has priority over the variable.
    assert _local_prefix() in plhf.scan_bucket(uri, resolve="redirect").explain()

    monkeypatch.setenv("POLARS_HF_RESOLVE", "now")
    assert SIGNATURE in plhf.scan_bucket(uri).explain()
    monkeypatch.delenv("POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN")
    with pytest.raises(ValueError, match="POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1"):
        plhf.scan_bucket(uri)

    monkeypatch.setenv("POLARS_HF_RESOLVE", "")
    assert _local_prefix() in plhf.scan_bucket(uri).explain()

    fake_hub.reset_log()
    monkeypatch.setenv("POLARS_HF_RESOLVE", "later")
    with pytest.raises(ValueError, match="POLARS_HF_RESOLVE must be one of"):
        plhf.scan_bucket(uri)
    with pytest.raises(ValueError, match="resolve must be one of"):
        plhf.scan_bucket(uri, resolve="later")
    assert fake_hub.requests == []


# ---- the plan --------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*json.*:UserWarning")
def test_plan_holds_no_signed_url_and_no_token(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    token = "explicit-read-token"
    fake_hub.accept_token(token)
    expected = _put_numbered(fake_hub, fake_bucket, 2)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"), token=token)
    # After a query the server has resolved the URLs.
    assert_frame_equal(lf.collect(), expected)

    for text in (
        lf.explain(),
        lf.explain(optimized=False),
        lf.serialize(format="binary"),
        lf.serialize(format="json"),
    ):
        assert_no_signed_url(text, fake_hub)
        encoded = text if isinstance(text, bytes) else text.encode()
        assert token.encode() not in encoded


def test_scan_bucket_resolves_nothing_and_a_query_resolves_each_file_once(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    n_files = 6
    expected = _put_numbered(fake_hub, fake_bucket, n_files)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    assert [(r.method, r.status) for r in fake_hub.requests] == [("GET", 200)]

    assert_frame_equal(lf.collect(), expected)
    assert lf.select(pl.len()).collect().item() == n_files * ROWS
    assert lf.head(3).collect().height == 3

    # Three queries, one resolve request per file: the server keeps the URLs.
    assert sorted(_resolved(fake_hub)) == [f"p{i:03d}.parquet" for i in range(n_files)]
    hub_requests = fake_hub.matching(origin=HUB)
    assert all(request.has_authorization for request in hub_requests)
    assert not any(r.has_authorization for r in fake_hub.matching(origin=CDN))


def test_single_file_is_resolved_once_by_the_call(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 1)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/p000.parquet"))
    assert_frame_equal(lf.collect(), expected)

    assert _resolved(fake_hub) == ["p000.parquet"]


def test_new_scan_reads_a_file_that_was_replaced(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # Every scan_bucket call has its own URLs: no URL of an earlier call.
    uri = _uri(fake_bucket, "data/p000.parquet")
    fake_hub.put_parquet(fake_bucket, "data/p000.parquet", _numbered_frame(0, 5))
    assert plhf.scan_bucket(uri).collect().height == 5

    fake_hub.put_parquet(fake_bucket, "data/p000.parquet", _numbered_frame(0, 7))

    assert plhf.scan_bucket(uri).collect().height == 7


# ---- the same rows as the native scan over presigned URLs ------------------

_QUERIES = {
    "full": lambda lf: lf,
    "projection": lambda lf: lf.select("label", "id"),
    "pushed predicate": lambda lf: lf.filter(pl.col("id").is_between(25, 34)),
    "predicate that is not pushed": lambda lf: lf.filter(
        pl.col("id").cum_sum() % 7 == 0
    ),
    "head": lambda lf: lf.head(35),
    "tail": lambda lf: lf.tail(7),
    "slice": lambda lf: lf.slice(27, 30),
    "slice outside": lambda lf: lf.slice(500, 5),
    "row count": lambda lf: lf.select(pl.len()),
    "row index": lambda lf: lf.with_row_index().filter(pl.col("index") > 40),
    "head then filter": lambda lf: lf.head(35).filter(pl.col("label") == "1"),
    "aggregate": lambda lf: lf.group_by("label").agg(pl.col("id").sum()).sort("label"),
}


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("name", list(_QUERIES))
def test_queries_equal_the_native_scan(
    fake_hub: FakeHub, fake_bucket: str, name: str, engine: str
) -> None:
    _put_numbered(fake_hub, fake_bucket, 5)
    uri = _uri(fake_bucket, "data/")
    query = _QUERIES[name]

    lf = query(plhf.scan_bucket(uri))
    native = query(plhf.scan_bucket(uri, resolve="now"))

    assert lf.collect_schema() == native.collect_schema()
    assert_frame_equal(lf.collect(engine=engine), native.collect(engine=engine))


@pytest.mark.parametrize("n_files", [1, 63, 64, 65, 129])
def test_file_counts_equal_the_native_scan(
    fake_hub: FakeHub, fake_bucket: str, n_files: int
) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, n_files, rows=2)
    uri = _uri(fake_bucket, "data/")

    got = plhf.scan_bucket(uri).collect()

    # The rows of the files, in file order; one resolve request per file.
    assert_frame_equal(got, expected)
    assert len(_resolved(fake_hub)) == n_files
    if n_files == 65:
        assert_frame_equal(got, plhf.scan_bucket(uri, resolve="now").collect())


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_concat_and_join_of_two_scans(
    fake_hub: FakeHub, fake_bucket: str, engine: str
) -> None:
    _put_numbered(fake_hub, fake_bucket, 3)
    fake_hub.put_parquet(
        fake_bucket,
        "names/n.parquet",
        pl.DataFrame({"id": [1, 12, 23], "name": ["a", "b", "c"]}),
    )

    def query(mode: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        data = plhf.scan_bucket(_uri(fake_bucket, "data/"), resolve=mode)
        names = plhf.scan_bucket(_uri(fake_bucket, "names/"), resolve=mode)
        joined = data.join(names, on="id").sort("id").collect(engine=engine)
        both = pl.concat([data, data.head(4)]).collect(engine=engine)
        return joined, both

    joined, both = query("redirect")
    native_joined, native_both = query("now")

    assert joined["name"].to_list() == ["a", "b", "c"]
    assert_frame_equal(joined, native_joined)
    assert both.height == 3 * ROWS + 4
    assert_frame_equal(both, native_both)


def _put_mixed(fake_hub: FakeHub, bucket_id: str) -> str:
    """Three files: ``id``; ``id`` and ``note``; ``id`` again."""
    fake_hub.put_parquet(bucket_id, "mixed/a.parquet", pl.DataFrame({"id": [1, 2]}))
    fake_hub.put_parquet(
        bucket_id, "mixed/b.parquet", pl.DataFrame({"id": [3], "note": ["x"]})
    )
    fake_hub.put_parquet(bucket_id, "mixed/c.parquet", pl.DataFrame({"id": [4, 5]}))
    return _uri(bucket_id, "mixed/")


@pytest.mark.parametrize(
    "scan_kwargs",
    [
        {"missing_columns": "insert", "extra_columns": "ignore"},
        {"extra_columns": "ignore"},
        {"schema": {"id": pl.Int64, "note": pl.String}, "missing_columns": "insert"},
        {"row_index_name": "row", "row_index_offset": 10, "extra_columns": "ignore"},
        {"n_rows": 4, "extra_columns": "ignore"},
    ],
)
def test_scan_options_equal_the_native_scan(
    fake_hub: FakeHub, fake_bucket: str, scan_kwargs: dict
) -> None:
    uri = _put_mixed(fake_hub, fake_bucket)

    lf = plhf.scan_bucket(uri, **scan_kwargs)
    native = plhf.scan_bucket(uri, resolve="now", **scan_kwargs)

    assert lf.collect_schema() == native.collect_schema()
    assert_frame_equal(lf.collect(), native.collect())


def test_mixed_schemas_raise_like_the_native_scan(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    uri = _put_mixed(fake_hub, fake_bucket)

    with pytest.raises(pl.exceptions.PolarsError):
        plhf.scan_bucket(uri, resolve="now").collect()
    with pytest.raises(pl.exceptions.PolarsError) as error:
        plhf.scan_bucket(uri).collect()

    assert_no_signed_url(str(error.value), fake_hub)

    # A row count: the same result as the native node, a count or an error
    # (it depends on the polars version).
    def count(mode: str) -> object:
        try:
            return plhf.scan_bucket(uri, resolve=mode).select(pl.len()).collect().item()
        except pl.exceptions.PolarsError as error:
            return type(error)

    assert count("redirect") == count("now")


def _bytes_served(fake_hub: FakeHub, uri: str, mode: str, query) -> int:
    fake_hub.reset_log()
    query(plhf.scan_bucket(uri, resolve=mode)).collect()
    gets = fake_hub.matching(origin=CDN, method="GET")
    # The Range header of polars reaches the cdn through the redirect.
    assert gets and all(request.range is not None for request in gets)
    return fake_hub.cdn_bytes_served


@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.select(pl.len()),
        lambda lf: lf.select("id"),
        lambda lf: lf.filter(pl.col("id") < 10),
        lambda lf: lf.head(5),
        lambda lf: lf.tail(5),
    ],
    ids=["row count", "one column", "predicate", "head", "tail"],
)
def test_query_reads_what_the_native_scan_reads(
    fake_hub: FakeHub, fake_bucket: str, query
) -> None:
    rows = 150_000
    total = 0
    for i in range(3):
        frame = _wide_frame(i * rows, rows)
        total += fake_hub.put_parquet(
            fake_bucket, f"wide/w{i}.parquet", frame, compression="uncompressed"
        )
    uri = _uri(fake_bucket, "wide/")

    redirect_bytes = _bytes_served(fake_hub, uri, "redirect", query)
    now_bytes = _bytes_served(fake_hub, uri, "now", query)

    # Footers and the pages a query needs: not the files, and not more than
    # the native scan over the presigned URLs.
    assert 0 < redirect_bytes < total / 2
    assert redirect_bytes <= now_bytes * 1.1


def test_row_count_reads_the_footers_only(fake_hub: FakeHub, fake_bucket: str) -> None:
    rows = 150_000
    total = 0
    for i in range(3):
        frame = _wide_frame(i * rows, rows)
        total += fake_hub.put_parquet(
            fake_bucket, f"wide/w{i}.parquet", frame, compression="uncompressed"
        )

    lf = plhf.scan_bucket(_uri(fake_bucket, "wide/"))
    count = lf.select(pl.len()).collect().item()

    assert count == 3 * rows
    gets = fake_hub.matching(origin=CDN, method="GET")
    assert all(request.range is not None for request in gets)
    # No column is read: far less than the smallest column of the files.
    assert fake_hub.cdn_bytes_served < total / 4
    assert count == plhf.count_rows(_uri(fake_bucket, "wide/"))


# ---- file paths ------------------------------------------------------------

_ODD_NAMES = [
    "odd/with space/a b.parquet",
    "odd/ünï/日本.parquet",
    "odd/100%/50%25.parquet",
    "odd/#hash/q?.parquet",
    "odd/plus+and&/x=1;y.parquet",
    "odd/a%2Fb.parquet",
]


def test_odd_file_names(fake_hub: FakeHub, fake_bucket: str) -> None:
    for name in _ODD_NAMES:
        fake_hub.put_parquet(fake_bucket, name, pl.DataFrame({"name": [name]}))
    uri = _uri(fake_bucket, "odd/")

    lf = plhf.scan_bucket(uri, include_file_paths="file")
    got = lf.collect().sort("name")

    assert got["name"].to_list() == sorted(_ODD_NAMES)
    # The hf:// URI of the file of every row, not the local URL.
    assert got["file"].to_list() == [
        _uri(fake_bucket, name) for name in sorted(_ODD_NAMES)
    ]
    for name in _ODD_NAMES:
        single = plhf.scan_bucket(_uri(fake_bucket, name)).collect()
        assert single["name"].to_list() == [name]


def test_include_file_paths_gives_bucket_uris(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    _put_numbered(fake_hub, fake_bucket, 3)
    uris = [_uri(fake_bucket, f"data/p{i:03d}.parquet") for i in range(3)]
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"), include_file_paths="file")

    got = lf.collect()

    assert lf.collect_schema().names() == ["id", "label", "file"]
    expected = []
    for uri in uris:
        expected.extend([uri] * ROWS)
    assert got["file"].to_list() == expected
    one_file = lf.filter(pl.col("file") == uris[2]).collect()
    assert one_file["id"].to_list() == list(range(2 * ROWS, 3 * ROWS))
    assert lf.select(pl.len()).collect().item() == 3 * ROWS


def test_hive_partitioning_reads_the_keys_of_the_bucket_path(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The path of a local URL is the bucket path, so polars finds the
    # key=value directories. The presigned URLs do not have them.
    for year, start in ((2024, 0), (2025, 10)):
        frame = pl.DataFrame({"id": list(range(start, start + 3))})
        fake_hub.put_parquet(
            fake_bucket, f"hive/year={year}/kind=a b/part.parquet", frame
        )
    uri = _uri(fake_bucket, "hive/")

    got = plhf.scan_bucket(uri, hive_partitioning=True).collect()

    assert got.columns == ["id", "year", "kind"]
    assert got["year"].to_list() == [2024] * 3 + [2025] * 3
    assert got["kind"].to_list() == ["a b"] * 6
    only_2025 = plhf.scan_bucket(uri, hive_partitioning=True).filter(
        pl.col("year") == 2025
    )
    assert only_2025.collect()["id"].to_list() == [10, 11, 12]
    assert plhf.scan_bucket(uri).collect().columns == ["id"]


# ---- failures of a query ---------------------------------------------------


def _assert_names_only_the_local_url(error: BaseException, fake_hub: FakeHub) -> None:
    """The canary: polars must name the local URL, never the redirect target."""
    assert _local_prefix() in str(error)
    for linked in exception_chain(error):
        assert_no_signed_url(str(linked), fake_hub)
        assert_no_signed_url(repr(linked), fake_hub)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    "fault",
    [
        {"status": 403, "body": b"expired"},
        {"status": 416},
        {"status": 302, "headers": {"Location": "/xet-bridge-us/loop"}},
    ],
    ids=["403", "416", "redirect loop"],
)
def test_cdn_refusal_names_only_the_local_url(
    fake_hub: FakeHub, fake_bucket: str, fault: dict, engine: str
) -> None:
    _put_numbered(fake_hub, fake_bucket, 2)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    for method in ("HEAD", "GET"):
        fake_hub.add_fault(CDN, method, r"^/xet-bridge-us/", times=1000, **fault)

    with pytest.raises(READ_ERRORS) as error:
        lf.collect(engine=engine)

    _assert_names_only_the_local_url(error.value, fake_hub)
    assert len(fake_hub.matching(origin=CDN)) > 0


@pytest.mark.parametrize(
    "fault",
    [
        {"status": 500},
        {"status": 0, "action": "reset"},
        {"status": 0, "action": "truncate"},
    ],
    ids=["500", "connection reset", "truncated body"],
)
def test_cdn_failure_after_the_retries_of_polars_names_only_the_local_url(
    fake_hub: FakeHub, fake_bucket: str, fault: dict
) -> None:
    # Polars retries these for several seconds (the retries cannot be turned
    # off for http URLs), then raises.
    _put_numbered(fake_hub, fake_bucket, 1)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    for method in ("HEAD", "GET"):
        fake_hub.add_fault(CDN, method, r"^/xet-bridge-us/", times=100_000, **fault)

    with pytest.raises(READ_ERRORS) as error:
        lf.collect()

    _assert_names_only_the_local_url(error.value, fake_hub)
    # The retries of polars use the URL that the server keeps.
    assert _resolved(fake_hub) == ["p000.parquet"]


def test_file_deleted_after_the_listing(
    fake_hub: FakeHub, fake_bucket: str, caplog: pytest.LogCaptureFixture
) -> None:
    _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    fake_hub.delete(fake_bucket, "data/p001.parquet")
    started = time.monotonic()

    with caplog.at_level("WARNING", logger="polars_hf._redirect"):
        with pytest.raises(READ_ERRORS) as error:
            lf.collect()

    # A 404 of the local server: polars does not retry it.
    assert time.monotonic() - started < 5
    assert "404" in str(error.value)
    assert "p001.parquet" in str(error.value)
    _assert_names_only_the_local_url(error.value, fake_hub)
    missing = fake_hub.matching(origin=HUB, path_contains="p001.parquet")
    assert [request.status for request in missing] == [404]
    # The reason is logged: polars shows no answer body of a HEAD request.
    reason = f"no such file: '{_uri(fake_bucket, 'data/p001.parquet')}'"
    assert any(reason in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("status", [401, 403])
def test_refused_resolve_of_a_query(
    fake_hub: FakeHub, fake_bucket: str, caplog: pytest.LogCaptureFixture, status: int
) -> None:
    _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    fake_hub.add_fault(HUB, "HEAD", r"/resolve/data/p001", status, times=1000)

    with caplog.at_level("WARNING", logger="polars_hf._redirect"):
        with pytest.raises(READ_ERRORS) as error:
            lf.collect()

    assert "403" in str(error.value)
    _assert_names_only_the_local_url(error.value, fake_hub)
    assert len(fake_hub.matching(origin=HUB, path_contains="p001.parquet")) == 1
    assert any("lacks access" in record.getMessage() for record in caplog.records)


def test_rate_limit_of_a_query(
    fake_hub: FakeHub, fake_bucket: str, caplog: pytest.LogCaptureFixture
) -> None:
    _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    headers = {
        "ratelimit": '"resolvers";r=0;t=900',
        "ratelimit-policy": '"fixed window";"resolvers";q=5000;w=300',
    }
    fake_hub.add_fault(
        HUB, "HEAD", r"/resolve/data/p002", 429, times=1000, headers=headers
    )
    started = time.monotonic()

    with caplog.at_level("WARNING", logger="polars_hf._redirect"):
        with pytest.raises(READ_ERRORS) as error:
            lf.collect()

    # Answered with 424, which polars does not retry (it retries a 429): no
    # wait of 900 s, no further resolve request for the file.
    assert time.monotonic() - started < 5
    assert "424" in str(error.value)
    _assert_names_only_the_local_url(error.value, fake_hub)
    assert len(fake_hub.matching(origin=HUB, path_contains="p002.parquet")) == 1
    logged = [record.getMessage() for record in caplog.records]
    assert any("rate limit for resolve requests was reached" in m for m in logged)
    assert any("5000 requests per 300 s" in message for message in logged)
    assert any("only 20 s are left of the 20 s allowed" in m for m in logged)


def test_resolver_describes_errors_for_the_server() -> None:
    from conftest import hub_httpx

    resolver = read._RedirectResolver("https://huggingface.co", {}, "ns/name")

    def http_error(status: int) -> HfHubHTTPError:
        request = hub_httpx.Request("HEAD", "https://huggingface.co/x")
        response = hub_httpx.Response(status, request=request)
        return HfHubHTTPError(f"HTTP {status}", response=response)

    assert resolver.describe(FileNotFoundError("gone")) == (404, "gone")
    assert resolver.describe(PermissionError("refused")) == (403, "refused")
    # Polars retries 429, 408 and 5xx: only what can clear in seconds is 503.
    assert resolver.describe(http_error(429))[0] == 424
    assert resolver.describe(http_error(503))[0] == 503
    assert resolver.describe(http_error(408))[0] == 503
    assert resolver.describe(http_error(400))[0] == 424
    assert resolver.describe(ValueError("empty")) == (424, "empty")
    assert resolver.describe(RuntimeError("no redirect")) == (424, "no redirect")
    assert resolver.describe(TimeoutError("slow")) == (503, "TimeoutError: slow")
    # No signature in the text of an answer.
    status, message = resolver.describe(
        RuntimeError("bad URL https://cdn.example/x?X-Amz-Signature=abc&a=1")
    )
    assert status == 424 and "abc" not in message and "X-Amz" not in message


# ---- expiry ----------------------------------------------------------------


def test_url_older_than_the_margin_is_resolved_again(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = types.SimpleNamespace(now=5000.0)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(_redirect, "time", clock)
    expected = _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    lf.collect()
    assert len(_resolved(fake_hub)) == 3

    # At the margin the URLs are served again.
    clock.now += _redirect._URL_MAX_AGE
    fake_hub.reset_log()
    lf.collect()
    assert _resolved(fake_hub) == []

    # One second later they are not: the cdn refuses the old ones from now
    # on, so the query works only if no old URL is served.
    clock.now += 1
    fake_hub.expire_signed_urls()
    assert_frame_equal(lf.collect(), expected)
    assert sorted(_resolved(fake_hub)) == [f"p{i:03d}.parquet" for i in range(3)]


def test_cdn_refusal_of_a_kept_url(fake_hub: FakeHub, fake_bucket: str) -> None:
    # The server does not see the answers of the cdn: polars talks to the
    # cdn itself. A URL that the cdn refuses before the margin is served
    # until the margin; a new scan_bucket call resolves new URLs.
    expected = _put_numbered(fake_hub, fake_bucket, 2)
    uri = _uri(fake_bucket, "data/")
    lf = plhf.scan_bucket(uri)
    lf.collect()

    fake_hub.expire_signed_urls()
    with pytest.raises(READ_ERRORS) as error:
        lf.collect()
    _assert_names_only_the_local_url(error.value, fake_hub)

    assert_frame_equal(plhf.scan_bucket(uri).collect(), expected)


# ---- proxy variables -------------------------------------------------------


@pytest.mark.parametrize("name", ["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"])
def test_proxy_without_loopback_exemption_is_refused(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    _put_numbered(fake_hub, fake_bucket, 1)
    monkeypatch.setenv(name, "http://proxy.example:3128")
    for no_proxy in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(no_proxy, raising=False)
    fake_hub.reset_log()

    with pytest.raises(RuntimeError) as error:
        plhf.scan_bucket(_uri(fake_bucket, "data/"))

    message = str(error.value)
    assert name in message
    assert "NO_PROXY=127.0.0.1,localhost" in message
    assert 'resolve="collect"' in message
    assert fake_hub.requests == []
    # The environment is not changed.
    assert os.environ[name] == "http://proxy.example:3128"
    assert "NO_PROXY" not in os.environ and "no_proxy" not in os.environ


@pytest.mark.parametrize("name", ["NO_PROXY", "no_proxy"])
def test_proxy_with_loopback_exemption_works(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 2)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    for no_proxy in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(no_proxy, raising=False)
    monkeypatch.setenv(name, "127.0.0.1,localhost")

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))

    assert_frame_equal(lf.collect(), expected)


@pytest.mark.parametrize(
    ("value", "covered"),
    [
        ("127.0.0.1", True),
        ("localhost, 127.0.0.1 ,.internal", True),
        ("*", True),
        ("127.0.0.0/8", True),
        ("", False),
        ("localhost", False),
        ("10.0.0.0/8,.example.com", False),
        ("127.0.0.10", False),
    ],
)
def test_no_proxy_values(value: str, covered: bool) -> None:
    assert read._no_proxy_covers_loopback(value) is covered


def test_proxy_check_is_for_the_redirect_mode_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    for no_proxy in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(no_proxy, raising=False)

    # Refused for another reason, before a request: not for the proxy.
    with pytest.raises(ValueError, match="a glob cannot end with"):
        plhf.scan_bucket("hf://buckets/ns/name/data/*/", resolve="collect")
    with pytest.raises(RuntimeError, match="HTTP_PROXY"):
        plhf.scan_bucket("hf://buckets/ns/name/data/*/")


# ---- the server: requests sent directly ------------------------------------


def _url_of(path: str) -> str:
    quoted = quote(path, safe="/")
    return f"https://cdn.example/files/{quoted}?Signature=sig-of-{len(path)}"


def _describe(error: Exception) -> tuple[int, str]:
    if isinstance(error, FileNotFoundError):
        return 404, str(error)
    if isinstance(error, TimeoutError):
        return 503, str(error)
    return 424, str(error)


def _register(paths: list[str], resolve=_url_of, describe=_describe) -> list[str]:
    return _redirect.register(paths, resolve, describe)


def _target(url: str) -> str:
    """The request target (path) of a local URL."""
    return urlsplit(url).path


def _request_bytes(
    target: str, *, method: str = "GET", host: str | None = None
) -> bytes:
    if host is None:
        host = f"127.0.0.1:{_redirect.get_server().port}"
    return f"{method} {target} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode()


def _read_response(sock: socket.socket) -> tuple[int, dict[str, str], bytes]:
    """One HTTP response from ``sock``: status, headers (lower case) and body."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError(f"closed without a response: {data!r}")
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ")[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    length = int(headers["content-length"])
    while len(body) < length:
        chunk = sock.recv(65536)
        if not chunk:
            break
        body += chunk
    return status, headers, body


def _send(data: bytes) -> tuple[int, dict[str, str], bytes]:
    """Send ``data`` on a new connection; the response must come within 2 s."""
    port = _redirect.get_server().port
    with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
        sock.sendall(data)
        return _read_response(sock)


def test_registered_file_is_redirected_for_get_and_head() -> None:
    urls = _register(["data/a.parquet", "data/b c.parquet"])

    for method in ("GET", "HEAD"):
        status, headers, body = _send(_request_bytes(_target(urls[1]), method=method))
        assert status == 302
        assert headers["location"] == _url_of("data/b c.parquet")
        assert headers["content-length"] == "0"
        assert body == b""
    # "localhost" names the same server.
    port = _redirect.get_server().port
    status, _, _ = _send(_request_bytes(_target(urls[0]), host=f"LocalHost:{port}"))
    assert status == 302


def test_connection_stays_open_after_an_answer() -> None:
    urls = _register(["data/a.parquet"])
    port = _redirect.get_server().port

    with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
        for target, expected in (
            (_target(urls[0]), 302),
            (_target(urls[0]) + "x", 404),
            ("/unknown/x", 404),
            (_target(urls[0]), 302),
        ):
            sock.sendall(_request_bytes(target))
            assert _read_response(sock)[0] == expected


def _bad_requests() -> list[tuple[str, bytes, int]]:
    """Requests that must be refused: ``(name, bytes, status)``."""
    urls = _register(["data/a.parquet", "data/sub/b.parquet"])
    target = _target(urls[0])
    base = target[: -len("data/a.parquet")]
    host = f"127.0.0.1:{_redirect.get_server().port}"
    return [
        ("long request line", _request_bytes("/" + "a" * 40_000), 400),
        (
            "long headers",
            f"GET {target} HTTP/1.1\r\nX: {'a' * 40_000}\r\n\r\n".encode(),
            400,
        ),
        ("garbage", b"\x00\x01\x02 not http\r\n\r\n", 400),
        ("no version", f"GET {target}\r\nHost: {host}\r\n\r\n".encode(), 400),
        ("not http", f"GET {target} SPDY/3\r\nHost: {host}\r\n\r\n".encode(), 400),
        ("absolute target", _request_bytes("http://evil.example/x"), 400),
        ("post", _request_bytes(target, method="POST"), 405),
        ("delete", _request_bytes(target, method="DELETE"), 405),
        ("unknown id", _request_bytes("/not-a-scan-id/x/data/a.parquet"), 404),
        ("no path", _request_bytes("/"), 404),
        ("id only", _request_bytes(base.rstrip("/")), 404),
        ("wrong marker", _request_bytes(target.replace("valid-only", "valid")), 404),
        ("prefix of a file", _request_bytes(target[:-3]), 404),
        ("file as a directory", _request_bytes(target + "/x"), 404),
        ("directory", _request_bytes(base + "data/"), 404),
        ("traversal", _request_bytes(base + "data/../data/a.parquet"), 404),
        ("encoded traversal", _request_bytes(base + "%2e%2e%2fdata%2fa.parquet"), 404),
        ("double slash", _request_bytes(base + "data//a.parquet"), 404),
        ("encoded line break", _request_bytes(target + "%0d%0aX-Header:%20y"), 400),
        ("encoded nul", _request_bytes(target + "%00"), 400),
        ("encoded delete", _request_bytes(target + "%7f"), 400),
        ("not utf-8", _request_bytes(base + "data/%ff.parquet"), 400),
        ("other host", _request_bytes(target, host="evil.example"), 403),
        ("other port", _request_bytes(target, host="127.0.0.1:1"), 403),
        ("no host", f"GET {target} HTTP/1.1\r\n\r\n".encode(), 403),
    ]


def test_bad_requests_get_a_4xx_answer(capfd: pytest.CaptureFixture) -> None:
    calls = []

    def resolve(path: str) -> str:
        calls.append(path)
        return _url_of(path)

    _register(["other.parquet"], resolve)
    capfd.readouterr()
    for name, data, expected in _bad_requests():
        started = time.monotonic()
        status, headers, body = _send(data)
        assert status == expected, name
        assert time.monotonic() - started < 2, name
        assert "location" not in headers, name
        # The answer does not repeat the path, and nothing was resolved.
        assert b"a.parquet" not in body and b"evil" not in body, name
    assert calls == []
    assert capfd.readouterr().err == ""


def test_answer_for_an_unknown_scan_says_what_to_do() -> None:
    status, _, body = _send(_request_bytes("/not-a-scan-id/x/data/a.parquet"))

    assert status == 404
    assert b"valid only in the process that made it" in body
    assert b'resolve="collect"' in body
    # A HEAD answer has no body.
    head = _request_bytes("/not-a-scan-id/x/data/a.parquet", method="HEAD")
    assert _send(head) == (404, {"content-length": "0"}, b"")


@pytest.mark.parametrize(
    ("error", "status", "text"),
    [
        (
            FileNotFoundError("no such file: 'hf://buckets/ns/n/a.parquet'"),
            404,
            b"no such file",
        ),
        (RuntimeError("the Hub did not redirect"), 424, b"did not redirect"),
        (TimeoutError("the Hub did not answer"), 503, b"did not answer"),
    ],
)
def test_resolver_error_becomes_an_answer(
    capfd: pytest.CaptureFixture,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    status: int,
    text: bytes,
) -> None:
    calls = []

    def resolve(path: str) -> str:
        calls.append(path)
        raise error

    urls = _register(["a.parquet"], resolve)
    capfd.readouterr()

    with caplog.at_level("WARNING", logger="polars_hf._redirect"):
        first = _send(_request_bytes(_target(urls[0])))
        second = _send(_request_bytes(_target(urls[0])))

    assert first[0] == status and text in first[2]
    assert first[2].startswith(b"polars-hf: ")
    # The answer is repeated for a few seconds: no second call (polars
    # retries a 503 ten times).
    assert second[0] == status
    assert calls == ["a.parquet"]
    assert capfd.readouterr().err == ""
    assert len(caplog.records) == 1


def test_resolver_is_called_again_after_the_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = types.SimpleNamespace(now=100.0)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(_redirect, "time", clock)
    answers = [TimeoutError("slow"), None]

    def resolve(path: str) -> str:
        answer = answers.pop(0)
        if answer is not None:
            raise answer
        return _url_of(path)

    urls = _register(["a.parquet"], resolve)

    assert _send(_request_bytes(_target(urls[0])))[0] == 503
    clock.now += _redirect._FAILURE_HOLD + 0.1
    assert _send(_request_bytes(_target(urls[0])))[0] == 302


def test_failing_describe_and_unsafe_urls_are_refused(
    capfd: pytest.CaptureFixture,
) -> None:
    def broken(path: str) -> str:
        raise KeyError(path)

    def bad_describe(error: Exception) -> tuple[int, str]:
        raise ZeroDivisionError

    def odd_status(error: Exception) -> tuple[int, str]:
        return 302, "moved"

    unsafe = [
        "https://cdn.example/x\r\nSet-Cookie: a=b",
        "https://cdn.example/x y",
        "https://cdn.example/é",
        "ftp://cdn.example/x",
        "",
    ]
    urls = _register(["a.parquet"], broken, bad_describe)
    urls += _register(["b.parquet"], broken, odd_status)
    for location in unsafe:
        urls += _register(["c.parquet"], lambda path, location=location: location)
    capfd.readouterr()

    for url in urls:
        status, headers, body = _send(_request_bytes(_target(url)))
        assert status == 424
        assert "location" not in headers and "set-cookie" not in headers
        assert b"cdn.example" not in body

    assert capfd.readouterr().err == ""


def test_token_is_not_in_an_answer(fake_hub: FakeHub, fake_bucket: str) -> None:
    from conftest import STAGING_TOKEN

    _put_numbered(fake_hub, fake_bucket, 1)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    local_url = lf.explain().partition("[")[2].partition("]")[0].split(",")[0]
    assert local_url.startswith(_local_prefix())
    port = _redirect.get_server().port

    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(_request_bytes(_target(local_url)))
        raw = sock.recv(65536)

    assert raw.startswith(b"HTTP/1.1 302 Found\r\n")
    assert fake_hub.cdn_endpoint.encode() in raw
    assert STAGING_TOKEN.encode() not in raw
    assert STAGING_TOKEN not in local_url


# ---- the server: concurrency -----------------------------------------------


async def _keep_alive_requests(port: int, targets: list[str]) -> list[int]:
    """Send one GET per target on one connection; the statuses."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    statuses = []
    try:
        for target in targets:
            writer.write(_request_bytes(target))
            await writer.drain()
            head = await reader.readuntil(b"\r\n\r\n")
            statuses.append(int(head.split(b" ", 2)[1]))
    finally:
        writer.close()
    return statuses


def test_ten_thousand_requests_over_a_thousand_connections() -> None:
    n_connections = 1000
    per_connection = 10
    paths = [f"data/f{i:03d}.parquet" for i in range(50)]
    targets = [_target(url) for url in _register(paths)]
    port = _redirect.get_server().port

    async def run() -> list[int]:
        statuses: list[int] = []
        # 100 connections at a time: the test process has a limit of open files.
        for batch in range(0, n_connections, 100):
            jobs = []
            for connection in range(batch, batch + 100):
                chosen = []
                for k in range(per_connection):
                    chosen.append(targets[(connection + k) % len(targets)])
                jobs.append(_keep_alive_requests(port, chosen))
            for result in await asyncio.gather(*jobs):
                statuses.extend(result)
        return statuses

    statuses = asyncio.run(run())

    assert len(statuses) == n_connections * per_connection
    assert set(statuses) == {302}


def test_burst_of_first_requests_resolves_every_file_once() -> None:
    n_files = 500
    lock = threading.Lock()
    calls: dict[str, int] = {}

    def counting_resolve(path: str) -> str:
        with lock:
            calls[path] = calls.get(path, 0) + 1
        # Long enough for the other requests for the file to arrive.
        time.sleep(0.05)
        return _url_of(path)

    paths = [f"burst/f{i:03d}.parquet" for i in range(n_files)]
    targets = [_target(url) for url in _register(paths, counting_resolve)]
    port = _redirect.get_server().port

    async def run() -> list[int]:
        # 4 requests for every file at the same time, on 100 connections.
        jobs = []
        for start in range(0, n_files, 20):
            for _ in range(4):
                jobs.append(_keep_alive_requests(port, targets[start : start + 20]))
        statuses: list[int] = []
        for result in await asyncio.gather(*jobs):
            statuses.extend(result)
        return statuses

    statuses = asyncio.run(run())

    assert len(statuses) == 4 * n_files
    assert set(statuses) == {302}
    assert len(calls) == n_files
    assert set(calls.values()) == {1}


def test_client_that_goes_away_does_not_break_the_resolve() -> None:
    release = threading.Event()
    calls = []

    def slow_resolve(path: str) -> str:
        calls.append(path)
        release.wait(timeout=5)
        return _url_of(path)

    urls = _register(["a.parquet"], slow_resolve)
    port = _redirect.get_server().port

    # The first client leaves before the answer.
    with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
        sock.sendall(_request_bytes(_target(urls[0])))
        time.sleep(0.1)
    release.set()

    assert _send(_request_bytes(_target(urls[0])))[0] == 302
    assert calls == ["a.parquet"]


# ---- the server: lifecycle -------------------------------------------------


def test_second_scan_uses_the_same_server(fake_hub: FakeHub, fake_bucket: str) -> None:
    _put_numbered(fake_hub, fake_bucket, 1)
    server = _redirect.get_server()

    first = plhf.scan_bucket(_uri(fake_bucket, "data/")).explain()
    second = plhf.scan_bucket(_uri(fake_bucket, "data/")).explain()

    assert _redirect.get_server() is server
    assert first.count(_local_prefix()) == 1 and second.count(_local_prefix()) == 1
    # Each call has its own id.
    assert first != second
    names = [thread.name for thread in threading.enumerate()]
    assert names.count("polars-hf-redirect") == 1
    assert server.pid == os.getpid()


def test_registrations_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_redirect, "_MAX_SCANS", 3)
    server = _redirect.get_server()

    first = _register(["a.parquet"])
    second = _register(["a.parquet"])
    # A request makes a scan the one used last.
    assert _send(_request_bytes(_target(first[0])))[0] == 302
    for _ in range(2):
        _register(["a.parquet"])

    assert server.registered_scans() == 3
    assert _send(_request_bytes(_target(first[0])))[0] == 302
    status, _, body = _send(_request_bytes(_target(second[0])))
    assert status == 404
    assert b"Call scan_bucket again" in body


_SCRIPT_PRELUDE = """
import os, sys
sys.path.insert(0, {tests!r})
import polars as pl
from fakehub import CDN, FakeHub
from huggingface_hub import constants
import polars_hf as plhf
from polars_hf import _redirect

def start_hub():
    hub = FakeHub(token="offline-token").start()
    constants.ENDPOINT = hub.endpoint
    for i in range(3):
        hub.put_parquet("u/b", f"d/p{{i}}.parquet", pl.DataFrame({{"id": [i] * 100}}))
    return hub
"""


def _run_script(body: str, **environment: str) -> subprocess.CompletedProcess:
    """Run ``body`` after the prelude in a new interpreter."""
    code = _SCRIPT_PRELUDE.format(tests=TESTS_DIRECTORY) + textwrap.dedent(body)
    env = dict(os.environ)
    env.update(
        HF_TOKEN="offline-token",
        HF_HUB_OFFLINE="0",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    env.pop("POLARS_VERBOSE", None)
    env.update(environment)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_interpreter_exit_is_silent() -> None:
    done = _run_script(
        """
        import socket
        hub = start_hub()
        lf = plhf.scan_bucket("hf://buckets/u/b/d/")
        assert lf.collect().height == 300
        # A connection that is still open when the interpreter exits.
        idle = socket.create_connection(("127.0.0.1", _redirect.get_server().port))
        idle.sendall(b"GET /x HTTP/1.1\\r\\nHost: x\\r\\n")
        print("done")
        """
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "done"
    assert done.stderr == ""


def test_verbose_polars_log_has_no_signed_url() -> None:
    done = _run_script(
        """
        hub = start_hub()
        print("CDN", hub.cdn_endpoint.partition("://")[2])
        lf = plhf.scan_bucket("hf://buckets/u/b/d/")
        print("ROWS", lf.collect().height, lf.select(pl.len()).collect().item())
        print("HEAD", lf.head(5).collect(engine="streaming").height)
        for method in ("HEAD", "GET"):
            hub.add_fault(CDN, method, "^/xet-bridge-us/", 403, times=1000)
        try:
            lf.collect()
        except Exception as error:
            print("FAILED", type(error).__name__)
        """,
        POLARS_VERBOSE="1",
    )

    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert "ROWS 300 300" in lines and "HEAD 5" in lines and "FAILED OSError" in lines
    cdn_host = lines[0].split(" ")[1]
    # The verbose log of polars is there, and it names the local URLs only.
    assert "127.0.0.1" in done.stderr
    assert cdn_host not in done.stderr
    lowered = done.stderr.lower()
    assert "signature" not in lowered and "x-amz" not in lowered
    assert "offline-token" not in done.stderr


def test_plan_fails_with_a_readable_error_when_its_process_is_gone() -> None:
    # A serialized plan of the redirect mode, used after the process that
    # made it has exited. The package gets no control here: polars cannot
    # connect, retries for some seconds and names the local URL.
    done = _run_script(
        """
        hub = start_hub()
        lf = plhf.scan_bucket("hf://buckets/u/b/d/")
        assert lf.collect().height == 300
        print(os.getpid(), lf.serialize(format="binary").hex())
        """
    )
    assert done.returncode == 0, done.stderr
    pid, plan = done.stdout.split()
    lf = pl.LazyFrame.deserialize(io.BytesIO(bytes.fromhex(plan)), format="binary")
    started = time.monotonic()

    with pytest.raises(READ_ERRORS) as error:
        lf.collect()

    assert time.monotonic() - started < 60
    assert f"valid-only-while-pid-{pid}-runs" in str(error.value)
    assert "http://127.0.0.1:" in str(error.value)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_starts_its_own_server() -> None:
    done = _run_script(
        """
        import socket
        server = _redirect.get_server()
        port = server.port
        child = os.fork()
        if child == 0:
            # No server thread here: the inherited socket is closed, and the
            # next registration starts a server of this process.
            ok = _redirect._server is None and server._listening == []
            new = _redirect.get_server()
            ok = ok and new is not server and new.pid == os.getpid()
            ok = ok and new.port != port
            os._exit(0 if ok else 1)
        _, status = os.waitpid(child, 0)
        # The server of the parent still answers.
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            host = f"Host: 127.0.0.1:{port}"
            sock.sendall(f"GET /x/y HTTP/1.1\\r\\n{host}\\r\\n\\r\\n".encode())
            answer = sock.recv(100)
        print(os.WEXITSTATUS(status), answer.split(b"\\r\\n")[0].decode())
        """
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "0 HTTP/1.1 404 Not Found"
    assert done.stderr == ""
