"""``scan_bucket`` with ``resolve="collect"`` (the default) and ``resolve="now"``.

Offline, against the fake Hub. With ``resolve="collect"`` the LazyFrame is an
IO-plugin source: the signed URLs are resolved when the query runs, for one
group of files at a time. With ``resolve="now"`` they are resolved by
``scan_bucket`` and the LazyFrame is the native scan node. The tests pin what
each mode requests and when, that both return the same rows, and what differs.
"""

from __future__ import annotations

import io
import pickle
import types
from urllib.parse import quote

import polars as pl
import pytest
from conftest import assert_no_signed_url, exception_chain, raises_at_collect
from fakehub import CDN, HUB, SIGNATURE, FakeHub
from huggingface_hub.errors import HfHubHTTPError
from polars.testing import assert_frame_equal

import polars_hf as plhf
from polars_hf import read

ROWS = 10


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _numbered_frame(start: int, rows: int) -> pl.DataFrame:
    ids = pl.int_range(start, start + rows, eager=True)
    return pl.DataFrame({"id": ids, "label": (ids % 3).cast(pl.String)})


def _put_numbered(fake_hub: FakeHub, bucket_id: str, n_files: int) -> pl.DataFrame:
    """Store ``data/p00.parquet``, ... and return their rows in file order."""
    frames = []
    for i in range(n_files):
        frame = _numbered_frame(i * ROWS, ROWS)
        fake_hub.put_parquet(bucket_id, f"data/p{i:02d}.parquet", frame)
        frames.append(frame)
    return pl.concat(frames)


def _wide_frame(start: int, rows: int) -> pl.DataFrame:
    """A frame whose ``payload`` column dominates the (uncompressed) file size."""
    ids = pl.int_range(start, start + rows, eager=True)
    payload = (ids * 2654435761 % 1000003).cast(pl.String) + "-" + ids.cast(pl.String)
    return pl.DataFrame({"id": ids, "payload": payload, "value": ids.cast(pl.Float64)})


def _put_wide(fake_hub: FakeHub, bucket_id: str, n_files: int, rows: int) -> int:
    """Store ``wide/w0.parquet``, ... uncompressed and return their total size."""
    total = 0
    for i in range(n_files):
        frame = _wide_frame(i * rows, rows)
        total += fake_hub.put_parquet(
            bucket_id, f"wide/w{i}.parquet", frame, compression="uncompressed"
        )
    return total


def _resolved(fake_hub: FakeHub) -> list[str]:
    """The file names of the successful ``resolve`` requests, in arrival order."""
    names = []
    for request in fake_hub.matching(origin=HUB, method="HEAD"):
        if request.status == 302:
            names.append(request.path.rpartition("/")[2])
    return names


def _request_runs(fake_hub: FakeHub) -> list[tuple[str, int]]:
    """Runs of consecutive requests of one kind: ``("resolve", 3), ("cdn", 7), ...``.

    The listing is left out. The order shows whether a group of files was
    resolved before or after the data of the group before it was read.
    """
    runs: list[tuple[str, int]] = []
    for request in list(fake_hub.requests):
        if request.origin == CDN:
            kind = "cdn"
        elif "/resolve/" in request.path:
            kind = "resolve"
        else:
            continue
        if runs and runs[-1][0] == kind:
            runs[-1] = (kind, runs[-1][1] + 1)
        else:
            runs.append((kind, 1))
    return runs


def _source(bucket_id: str, n_files: int, **scan_kwargs: object) -> read._BucketSource:
    """The IO source of ``data/p00.parquet``, ..., to call it like polars does."""
    from huggingface_hub import constants
    from huggingface_hub.utils import build_hf_headers

    return read._BucketSource(
        endpoint=constants.ENDPOINT,
        headers=build_hf_headers(),
        bucket_id=bucket_id,
        paths=[f"data/p{i:02d}.parquet" for i in range(n_files)],
        scan_kwargs=scan_kwargs,
    )


@pytest.fixture(autouse=True)
def _collect_is_the_mode_of_this_module(
    monkeypatch: pytest.MonkeyPatch, _allow_signed_urls_in_plan: None
) -> None:
    """A ``scan_bucket`` call without ``resolve=`` uses ``resolve="collect"`` here."""
    monkeypatch.setenv("POLARS_HF_RESOLVE", "collect")


@pytest.fixture
def small_groups(monkeypatch: pytest.MonkeyPatch) -> int:
    """Groups of 3 files, so a few small files make several groups."""
    monkeypatch.setattr(read, "_GROUP_FILES", 3)
    return 3


# ---- the keyword -----------------------------------------------------------


def test_unknown_resolve_mode_is_rejected_without_a_request(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 5))

    with pytest.raises(ValueError, match="resolve must be one of"):
        plhf.scan_bucket(_uri(fake_bucket, "one.parquet"), resolve="later")

    assert fake_hub.requests == []


@pytest.mark.parametrize("path", ["data/p00.parquet", "data", "data/*.parquet"])
def test_both_modes_return_the_same_rows(
    fake_hub: FakeHub, fake_bucket: str, path: str
) -> None:
    _put_numbered(fake_hub, fake_bucket, 4)
    uri = _uri(fake_bucket, path)

    default = plhf.scan_bucket(uri).collect()
    collect = plhf.scan_bucket(uri, resolve="collect").collect()
    now = plhf.scan_bucket(uri, resolve="now").collect()

    assert_frame_equal(default, now)
    assert_frame_equal(collect, now)


# ---- the plan --------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*json.*:UserWarning")
def test_plan_holds_no_signed_url_and_no_token(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    token = "explicit-read-token"
    fake_hub.accept_token(token)
    expected = _put_numbered(fake_hub, fake_bucket, 2)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"), token=token)
    # After a query the source has resolved URLs and the schema.
    assert_frame_equal(lf.collect(), expected)

    binary = lf.serialize(format="binary")
    for text in (lf.explain(), lf.explain(optimized=False)):
        assert_no_signed_url(text, fake_hub)
        assert token not in text
    assert_no_signed_url(binary, fake_hub)
    assert token.encode() not in binary
    assert_no_signed_url(lf.serialize(format="json"), fake_hub)
    # What the serialized plan holds: the bucket paths.
    assert b"data/p01.parquet" in binary


def test_now_mode_plan_holds_the_signed_url(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The difference to the default mode, pinned.
    _put_numbered(fake_hub, fake_bucket, 2)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"), resolve="now")

    assert SIGNATURE in lf.explain()
    assert SIGNATURE.encode() in lf.serialize(format="binary")


def test_deserialized_plan_scans_with_the_token_of_the_environment(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    from conftest import STAGING_TOKEN

    token = "explicit-read-token"
    fake_hub.accept_token(token)
    expected = _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"), token=token)
    binary = lf.select("id").serialize(format="binary")
    fake_hub.reset_log()

    restored = pl.LazyFrame.deserialize(io.BytesIO(binary), format="binary")

    assert_frame_equal(restored.collect(), expected.select("id"))
    sent = {request.authorization for request in fake_hub.matching(origin=HUB)}
    assert sent == {f"Bearer {STAGING_TOKEN}"}


# ---- what is requested, and when -------------------------------------------


def test_scan_bucket_of_a_directory_only_lists(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    n_files = 5
    _put_numbered(fake_hub, fake_bucket, n_files)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))

    assert [(r.method, r.status) for r in fake_hub.requests] == [("GET", 200)]

    # The schema: one resolve and the footer of the first file, read once.
    schema = lf.collect_schema()
    assert schema == pl.Schema({"id": pl.Int64, "label": pl.String})
    assert _resolved(fake_hub) == ["p00.parquet"]
    footer_bytes = fake_hub.cdn_bytes_served
    assert 0 < footer_bytes < 2000
    lf.collect_schema()
    lf.explain()
    assert _resolved(fake_hub) == ["p00.parquet"]
    assert fake_hub.cdn_bytes_served == footer_bytes

    # The query: the other files; the URL of the first one is used again.
    assert lf.collect().height == n_files * ROWS
    assert sorted(_resolved(fake_hub)) == [f"p{i:02d}.parquet" for i in range(n_files)]


def test_single_file_is_resolved_once(fake_hub: FakeHub, fake_bucket: str) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 1)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/p00.parquet"))
    assert fake_hub.matching(origin=CDN) == []
    got = lf.collect()

    # The request that tells a file from a directory gives the URL.
    assert_frame_equal(got, expected)
    assert _resolved(fake_hub) == ["p00.parquet"]
    assert fake_hub.matching(origin=HUB, method="GET") == []


def test_second_query_within_the_window_resolves_nothing(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int
) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 8)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    assert_frame_equal(lf.collect(), expected)
    assert len(_resolved(fake_hub)) == 8
    fake_hub.reset_log()

    # The source keeps the URLs of all files, not of the first one only.
    assert_frame_equal(lf.collect(), expected)
    assert lf.select("id").head(25).collect().height == 25
    assert lf.filter(pl.col("id") > 70).collect().height == 9

    assert _resolved(fake_hub) == []
    assert len(fake_hub.matching(origin=CDN)) > 0


def test_query_after_the_window_resolves_again(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _moving_clock(monkeypatch)
    expected = _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    lf.collect()

    # At the limit a URL is used again; one second later it is not.
    clock.now += read._URL_REUSE_SECONDS
    fake_hub.reset_log()
    lf.collect()
    assert _resolved(fake_hub) == []

    clock.now += 1
    # The old URLs must not be used: the cdn refuses them from now on.
    fake_hub.expire_signed_urls()
    assert_frame_equal(lf.collect(), expected)
    assert sorted(_resolved(fake_hub)) == [f"p{i:02d}.parquet" for i in range(3)]


def test_kept_urls_are_not_more_than_the_files(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int
) -> None:
    _put_numbered(fake_hub, fake_bucket, 8)
    source = _source(fake_bucket, 8)

    for _ in range(3):
        assert sum(frame.height for frame in source(None, None, None, None)) == 80

    assert sorted(source._urls) == list(range(8))
    restored = pickle.loads(pickle.dumps(source))
    assert restored._urls == {}
    assert restored.paths == source.paths


@pytest.mark.parametrize("schema_first", [False, True])
def test_refused_url_is_resolved_again_once(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int, schema_first: bool
) -> None:
    # The cdn refuses the kept URLs before they are 5 minutes old: a group
    # that fails before its first batch is resolved again, once.
    expected = _put_numbered(fake_hub, fake_bucket, 8)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    if schema_first:
        lf.collect_schema()
    lf.collect()
    fake_hub.reset_log()

    fake_hub.expire_signed_urls()
    got = lf.collect()

    assert_frame_equal(got, expected)
    assert sorted(_resolved(fake_hub)) == [f"p{i:02d}.parquet" for i in range(8)]


def test_group_bounds() -> None:
    def bounds(n_files: int, **kwargs: bool) -> list[tuple[int, int]]:
        return list(read._group_bounds(n_files, **kwargs))

    assert bounds(130, limited=False, one_group=False) == [
        (0, 64),
        (64, 128),
        (128, 130),
    ]
    assert bounds(3, limited=False, one_group=False) == [(0, 3)]
    # A row limit: 1, 4, 16, then 64 files.
    assert bounds(100, limited=True, one_group=False) == [
        (0, 1),
        (1, 5),
        (5, 21),
        (21, 85),
        (85, 100),
    ]
    assert bounds(130, limited=True, one_group=True) == [(0, 130)]


def test_files_are_resolved_and_scanned_group_by_group(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int
) -> None:
    n_files = 8
    expected = _put_numbered(fake_hub, fake_bucket, n_files)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    fake_hub.reset_log()

    got = lf.collect()
    runs = _request_runs(fake_hub)

    # The rows are in file order, as from one native scan.
    assert_frame_equal(got, expected)
    assert_frame_equal(
        got, plhf.scan_bucket(_uri(fake_bucket, "data/"), resolve="now").collect()
    )
    # Schema (1 file), then groups of 3, 3 and 2 files. Each group is resolved
    # after the data of the group before it was read; the first file is not
    # resolved twice.
    assert [kind for kind, _ in runs] == ["resolve", "cdn"] * 4
    assert [count for kind, count in runs if kind == "resolve"] == [1, 2, 3, 2]


def test_head_stops_after_the_first_group(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int
) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 8)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))

    got = lf.head(5).collect()

    assert_frame_equal(got, expected.head(5))
    assert _resolved(fake_hub) == ["p00.parquet"]
    # Only the first file was read.
    assert len({request.path for request in fake_hub.matching(origin=CDN)}) == 1


def test_head_over_several_groups(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int
) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 8)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))

    got = lf.head(25).collect()

    # Groups of 1 and 3 files hold 40 rows: the third group is not resolved.
    assert_frame_equal(got, expected.head(25))
    assert sorted(_resolved(fake_hub)) == [f"p{i:02d}.parquet" for i in range(4)]


def test_now_mode_resolves_every_file_for_head(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    _put_numbered(fake_hub, fake_bucket, 8)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"), resolve="now")

    assert len(_resolved(fake_hub)) == 8
    assert fake_hub.matching(origin=CDN) == []
    assert lf.head(5).collect().height == 5
    assert len(_resolved(fake_hub)) == 8


_QUERIES = {
    "full": lambda lf: lf,
    "projection": lambda lf: lf.select("label", "id"),
    "filter": lambda lf: lf.filter(pl.col("id") % 7 == 0),
    "filter on two files": lambda lf: lf.filter(pl.col("id").is_between(25, 44)),
    "head": lambda lf: lf.head(35),
    "filter then head": lambda lf: lf.filter(pl.col("label") == "1").head(12),
    "head then filter": lambda lf: lf.head(35).filter(pl.col("label") == "1"),
    "tail": lambda lf: lf.tail(7),
    "slice": lambda lf: lf.slice(27, 30),
    "row count": lambda lf: lf.select(pl.len()),
    "column count": lambda lf: lf.select(pl.col("label").count()),
    "aggregate": lambda lf: lf.group_by("label").agg(pl.col("id").sum()).sort("label"),
    "row index": lambda lf: lf.with_row_index().filter(pl.col("index") > 40),
    "self join": lambda lf: lf.join(lf.select("id"), on="id").sort("id"),
    "nothing matches": lambda lf: lf.filter(pl.col("id") < 0),
}


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("name", list(_QUERIES))
def test_queries_equal_the_native_scan(
    fake_hub: FakeHub,
    fake_bucket: str,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    engine: str,
) -> None:
    # 50 rows in 5 files: groups of 2, 2 and 1 files.
    monkeypatch.setattr(read, "_GROUP_FILES", 2)
    _put_numbered(fake_hub, fake_bucket, 5)
    uri = _uri(fake_bucket, "data/")
    query = _QUERIES[name]

    got = query(plhf.scan_bucket(uri)).collect(engine=engine)
    native = query(plhf.scan_bucket(uri, resolve="now")).collect(engine=engine)

    assert_frame_equal(got, native)


# ---- pushdown into the native scan -----------------------------------------

WIDE_FILES = 3
WIDE_ROWS = 150_000


def _bytes_of_both_modes(fake_hub: FakeHub, uri: str, query) -> tuple[int, int]:
    """Run ``query`` in both modes; the cdn bytes of ``collect`` and of ``now``."""
    served = {}
    frames = {}
    for mode in ("collect", "now"):
        fake_hub.reset_log()
        frames[mode] = query(plhf.scan_bucket(uri, resolve=mode)).collect()
        served[mode] = fake_hub.cdn_bytes_served
        gets = fake_hub.matching(origin=CDN, method="GET")
        assert all(request.range is not None for request in gets)
    assert_frame_equal(frames["collect"], frames["now"])
    return served["collect"], served["now"]


@pytest.mark.parametrize(
    ("query", "rows"),
    [
        (lambda lf: lf.select("id"), WIDE_FILES * WIDE_ROWS),
        # The statistics of the files exclude all rows but those of the first.
        (lambda lf: lf.filter(pl.col("id") < 10), 10),
        (lambda lf: lf.head(5), 5),
    ],
    ids=["one column", "predicate", "head"],
)
def test_query_reads_a_part_of_the_files(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch, query, rows
) -> None:
    # The projection, the predicate and the row limit reach the native scan
    # of every group: the bytes read are those of the native scan node.
    monkeypatch.setattr(read, "_GROUP_FILES", 2)
    total = _put_wide(fake_hub, fake_bucket, WIDE_FILES, WIDE_ROWS)
    uri = _uri(fake_bucket, "wide/")
    assert query(plhf.scan_bucket(uri)).collect().height == rows

    collect_bytes, now_bytes = _bytes_of_both_modes(fake_hub, uri, query)

    assert 0 < collect_bytes < total / 2
    assert collect_bytes <= now_bytes * 1.1


def test_row_count_reads_one_column(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Polars asks an IO source for one column to count the rows, so the count
    # is not answered from the footers alone. It does not read the files.
    monkeypatch.setattr(read, "_GROUP_FILES", 2)
    total = _put_wide(fake_hub, fake_bucket, WIDE_FILES, WIDE_ROWS)
    uri = _uri(fake_bucket, "wide/")

    def count(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select(pl.len())

    assert count(plhf.scan_bucket(uri)).collect().item() == WIDE_FILES * WIDE_ROWS
    collect_bytes, now_bytes = _bytes_of_both_modes(fake_hub, uri, count)

    assert collect_bytes < total / 2
    # The native scan node answers the count without a column.
    assert now_bytes < collect_bytes


def test_head_reads_the_first_file_only(fake_hub: FakeHub, fake_bucket: str) -> None:
    _put_wide(fake_hub, fake_bucket, WIDE_FILES, 1000)

    got = plhf.scan_bucket(_uri(fake_bucket, "wide/")).head(5).collect()

    assert got["id"].to_list() == [0, 1, 2, 3, 4]
    assert _resolved(fake_hub) == ["w0.parquet"]
    assert len({request.path for request in fake_hub.matching(origin=CDN)}) == 1


# ---- URLs that expire ------------------------------------------------------


def test_later_groups_get_new_urls(
    fake_hub: FakeHub,
    fake_bucket: str,
    small_groups: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A scan that takes longer than the validity of a signed URL: every URL
    # made before a group starts is expired when the group is resolved.
    expected = _put_numbered(fake_hub, fake_bucket, 8)
    resolve_group = read._BucketSource._resolve

    def expire_then_resolve(self, start: int, stop: int, scope: str) -> list[str]:
        if start > 0:
            fake_hub.expire_signed_urls()
        return resolve_group(self, start, stop, scope)

    monkeypatch.setattr(read._BucketSource, "_resolve", expire_then_resolve)

    got = plhf.scan_bucket(_uri(fake_bucket, "data/")).collect()

    assert_frame_equal(got, expected)


def test_plan_does_not_expire(fake_hub: FakeHub, fake_bucket: str) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 3)
    uri = _uri(fake_bucket, "data/")
    default = plhf.scan_bucket(uri)
    now = plhf.scan_bucket(uri, resolve="now")

    fake_hub.expire_signed_urls()

    assert_frame_equal(default.collect(), expected)
    with pytest.raises((pl.exceptions.PolarsError, OSError)):
        now.collect()


def _moving_clock(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Replace the clock of ``polars_hf.read``; ``clock.now`` can be moved."""
    clock = types.SimpleNamespace(now=1000.0)
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda seconds: None
    monkeypatch.setattr(read, "time", clock)
    return clock


def test_url_of_the_first_file_is_resolved_again_when_it_is_old(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _moving_clock(monkeypatch)
    expected = _put_numbered(fake_hub, fake_bucket, 1)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/p00.parquet"))

    clock.now += read._URL_REUSE_SECONDS + 1
    fake_hub.expire_signed_urls()

    assert_frame_equal(lf.collect(), expected)
    assert _resolved(fake_hub) == ["p00.parquet", "p00.parquet"]


def _refuse_every_cdn_request(fake_hub: FakeHub) -> None:
    for method in ("HEAD", "GET"):
        fake_hub.add_fault(
            CDN, method, r"^/xet-bridge-us/", 403, times=1000, body=b"refused"
        )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("schema_first", [False, True])
def test_read_error_names_the_bucket_file_in_the_whole_chain(
    fake_hub: FakeHub, fake_bucket: str, schema_first: bool, engine: str
) -> None:
    # The cdn refuses every request: the error of polars is raised without
    # the signed URL, and no exception linked to it (cause, context) has it.
    _put_numbered(fake_hub, fake_bucket, 1)
    uri = _uri(fake_bucket, "data/p00.parquet")
    lf = plhf.scan_bucket(uri)
    if schema_first:
        lf.collect_schema()

    _refuse_every_cdn_request(fake_hub)
    with pytest.raises((pl.exceptions.PolarsError, OSError)) as error:
        lf.collect(engine=engine)

    assert uri in str(error.value)
    for linked in exception_chain(error.value):
        assert_no_signed_url(str(linked), fake_hub)
        assert_no_signed_url(repr(linked), fake_hub)
    # The URL was tried, dropped, resolved again and tried once more.
    assert _resolved(fake_hub) == ["p00.parquet", "p00.parquet"]


def test_clean_error_is_raised_without_context(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # Called as polars calls it: the exception that leaves the source has
    # no cause and no context, so no chain walker finds the original error.
    _put_numbered(fake_hub, fake_bucket, 2)
    source = _source(fake_bucket, 2)
    source.schema()
    _refuse_every_cdn_request(fake_hub)

    with pytest.raises((pl.exceptions.PolarsError, OSError)) as error:
        list(source(None, None, None, None))

    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert _uri(fake_bucket, "data/p00.parquet") in str(error.value)
    assert_no_signed_url(str(error.value), fake_hub)


SIGNED = (
    "https://us.aws.cdn.hf.co/xet-bridge-us/abc/def"
    "?Expires=1&Policy=pol~icy&Signature=si-g_n~&Key-Pair-Id=K123"
)
SIGNED_URI = "hf://buckets/ns/name/data/a.parquet"
_SIGNED_NAMES = ("signature", "policy", "key-pair-id", "x-amz-", "x-xet-")


@pytest.mark.parametrize(
    "text",
    [
        f"cannot read {SIGNED}: 403",
        f"cannot read {SIGNED[:60]}... (truncated)",
        f"cannot read '{quote(SIGNED, safe='')}'",
        f"cannot read {SIGNED.replace('&', '&amp;')}",
        f"cannot read {SIGNED.upper()}",
        f'path: "{SIGNED.replace("https://", "")}"',
        "request to another.host/x?X-Amz-Signature=abc&X-Amz-Credential=c&a=1 failed",
        "x-amz-security-token=tok x-xet-signed-range=0-5 key-pair-id=K123 policy=p",
        "query X-Amz-Signature%3Dabc%26X-Amz-Date%3D2026 was refused",
    ],
)
def test_scrub_removes_every_form_of_a_signed_url(text: str) -> None:
    clean = read._scrub_signed_urls(text, {SIGNED: SIGNED_URI})

    lowered = clean.lower()
    assert "cdn.hf.co" not in lowered
    for name in _SIGNED_NAMES:
        assert name not in lowered
    for value in ("si-g_n~", "pol~icy", "k123", "abc", "tok"):
        assert value not in lowered


def test_scrub_names_the_bucket_file_and_keeps_other_text() -> None:
    text = f"error reading {SIGNED} at offset 4, see https://example.com/a?b=1"

    clean = read._scrub_signed_urls(text, {SIGNED: SIGNED_URI})

    assert clean == (
        f"error reading {SIGNED_URI} at offset 4, see https://example.com/a?b=1"
    )
    assert read._without_signed_urls(ValueError("no url here"), {SIGNED: "x"}) is None


def test_scrub_finds_a_signed_url_in_a_linked_exception() -> None:
    inner = OSError(f"GET {SIGNED} failed")
    outer = RuntimeError("the scan failed")
    outer.__cause__ = inner

    clean = read._without_signed_urls(outer, {SIGNED: SIGNED_URI})

    assert type(clean) is RuntimeError
    assert str(clean) == "the scan failed"
    assert clean.__cause__ is None and clean.__context__ is None


# ---- errors of a query -----------------------------------------------------


def test_file_deleted_after_the_listing(fake_hub: FakeHub, fake_bucket: str) -> None:
    _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    fake_hub.delete(fake_bucket, "data/p01.parquet")
    missing = _uri(fake_bucket, "data/p01.parquet")

    with raises_at_collect(FileNotFoundError, "no such file") as error:
        lf.collect()

    message = str(error.value)
    assert missing in message
    assert_no_signed_url(message, fake_hub)


def test_first_file_deleted_after_the_listing(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The first file is resolved for the schema: polars wraps that error.
    _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    fake_hub.delete(fake_bucket, "data/p00.parquet")

    with raises_at_collect(FileNotFoundError, "no such file", schema_step=True):
        lf.collect()
    with raises_at_collect(FileNotFoundError, "no such file", schema_step=True):
        lf.collect_schema()


@pytest.mark.parametrize("status", [401, 403])
def test_refused_resolve_of_a_query(
    fake_hub: FakeHub, fake_bucket: str, status: int
) -> None:
    _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    fake_hub.add_fault(HUB, "HEAD", r"/resolve/data/p01\.parquet$", status)

    with raises_at_collect(PermissionError, "lacks access") as error:
        lf.collect()

    assert f"'{fake_bucket}'" in str(error.value)
    assert f"HTTP {status}" in str(error.value)


def test_rate_limit_of_a_query(fake_hub: FakeHub, fake_bucket: str) -> None:
    _put_numbered(fake_hub, fake_bucket, 3)
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"))
    headers = {
        "ratelimit": '"resolvers";r=0;t=900',
        "ratelimit-policy": '"fixed window";"resolvers";q=5000;w=300',
    }
    fake_hub.add_fault(
        HUB, "HEAD", r"/resolve/data/p02\.parquet$", 429, headers=headers
    )

    with raises_at_collect(HfHubHTTPError, "rate limit for resolve requests") as error:
        lf.collect()

    message = str(error.value)
    assert "5000 requests per 300 s" in message
    assert f"'{fake_bucket}'" in message
    assert "of 3 files were resolved" in message
    assert "allowed for one group of files of a scan_bucket query" in message


def test_non_parquet_file_error_names_the_bucket_file(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 5))
    fake_hub.put(fake_bucket, "data/notes.txt", b"these bytes are not parquet")

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/*"))

    with pytest.raises(pl.exceptions.PolarsError) as error:
        lf.collect()
    assert_no_signed_url(str(error.value), fake_hub)


# ---- schemas ---------------------------------------------------------------


def _put_mixed(fake_hub: FakeHub, bucket_id: str) -> str:
    """Three files: ``id``; ``id`` and ``note``; ``id`` again."""
    fake_hub.put_parquet(bucket_id, "mixed/a.parquet", pl.DataFrame({"id": [1, 2]}))
    fake_hub.put_parquet(
        bucket_id, "mixed/b.parquet", pl.DataFrame({"id": [3], "note": ["x"]})
    )
    fake_hub.put_parquet(bucket_id, "mixed/c.parquet", pl.DataFrame({"id": [4, 5]}))
    return _uri(bucket_id, "mixed/")


def _put_mixed_wide_first(fake_hub: FakeHub, bucket_id: str) -> str:
    """Two files: ``id`` and ``note``; then ``id`` only."""
    fake_hub.put_parquet(
        bucket_id, "wide_first/a.parquet", pl.DataFrame({"id": [1], "note": ["x"]})
    )
    fake_hub.put_parquet(
        bucket_id, "wide_first/b.parquet", pl.DataFrame({"id": [2, 3]})
    )
    return _uri(bucket_id, "wide_first/")


@pytest.mark.parametrize("group_files", [1, 2, 64])
@pytest.mark.parametrize(
    ("put", "scan_kwargs"),
    [
        (_put_mixed, {"missing_columns": "insert", "extra_columns": "ignore"}),
        (_put_mixed, {"extra_columns": "ignore"}),
        (_put_mixed_wide_first, {"missing_columns": "insert"}),
        (
            _put_mixed,
            {
                "schema": {"id": pl.Int64, "note": pl.String},
                "missing_columns": "insert",
            },
        ),
        (
            _put_mixed_wide_first,
            {"schema": {"id": pl.Int64}, "extra_columns": "ignore"},
        ),
    ],
)
def test_mixed_schemas_equal_the_native_scan(
    fake_hub: FakeHub,
    fake_bucket: str,
    monkeypatch: pytest.MonkeyPatch,
    put,
    scan_kwargs: dict,
    group_files: int,
) -> None:
    # With groups of 1 or 2 files, a file with other columns starts a group.
    monkeypatch.setattr(read, "_GROUP_FILES", group_files)
    uri = put(fake_hub, fake_bucket)

    lf = plhf.scan_bucket(uri, **scan_kwargs)
    native = plhf.scan_bucket(uri, resolve="now", **scan_kwargs)

    assert lf.collect_schema() == native.collect_schema()
    assert_frame_equal(lf.collect(), native.collect())


@pytest.mark.parametrize("group_files", [1, 64])
@pytest.mark.parametrize(
    ("put", "scan_kwargs"),
    [
        (_put_mixed, {}),
        (_put_mixed, {"missing_columns": "insert"}),
        (_put_mixed_wide_first, {}),
        (_put_mixed_wide_first, {"extra_columns": "ignore"}),
    ],
)
def test_mixed_schemas_raise_like_the_native_scan(
    fake_hub: FakeHub,
    fake_bucket: str,
    monkeypatch: pytest.MonkeyPatch,
    put,
    scan_kwargs: dict,
    group_files: int,
) -> None:
    monkeypatch.setattr(read, "_GROUP_FILES", group_files)
    uri = put(fake_hub, fake_bucket)

    with pytest.raises(pl.exceptions.PolarsError):
        plhf.scan_bucket(uri, resolve="now", **scan_kwargs).collect()
    with pytest.raises(pl.exceptions.PolarsError) as error:
        plhf.scan_bucket(uri, **scan_kwargs).collect()

    assert_no_signed_url(str(error.value), fake_hub)


@pytest.mark.parametrize("group_files", [1, 64])
def test_cast_options_equal_the_native_scan(
    fake_hub: FakeHub,
    fake_bucket: str,
    monkeypatch: pytest.MonkeyPatch,
    group_files: int,
) -> None:
    monkeypatch.setattr(read, "_GROUP_FILES", group_files)
    wide = pl.DataFrame({"n": [1, 2]}, schema={"n": pl.Int64})
    narrow = pl.DataFrame({"n": [3]}, schema={"n": pl.Int32})
    fake_hub.put_parquet(fake_bucket, "cast/a.parquet", wide)
    fake_hub.put_parquet(fake_bucket, "cast/b.parquet", narrow)
    uri = _uri(fake_bucket, "cast/")
    options = {"cast_options": pl.ScanCastOptions(integer_cast="upcast")}

    with pytest.raises(pl.exceptions.PolarsError):
        plhf.scan_bucket(uri).collect()
    got = plhf.scan_bucket(uri, **options).collect()
    native = plhf.scan_bucket(uri, resolve="now", **options).collect()

    assert got.schema == pl.Schema({"n": pl.Int64})
    assert_frame_equal(got, native)


def test_given_schema_is_the_schema_of_the_frame(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    uri = _put_mixed(fake_hub, fake_bucket)
    schema = {"id": pl.Int64, "note": pl.String}

    lf = plhf.scan_bucket(uri, schema=schema, missing_columns="insert")

    assert lf.collect_schema() == pl.Schema(schema)
    assert lf.collect()["note"].to_list() == [None, None, "x", None, None]


# ---- options that name files or count rows ---------------------------------


def test_include_file_paths_gives_bucket_uris(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(read, "_GROUP_FILES", 2)
    _put_numbered(fake_hub, fake_bucket, 3)
    uris = [_uri(fake_bucket, f"data/p{i:02d}.parquet") for i in range(3)]
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"), include_file_paths="file")

    got = lf.collect()

    assert lf.collect_schema().names() == ["id", "label", "file"]
    expected = []
    for uri in uris:
        expected.extend([uri] * ROWS)
    assert got["file"].to_list() == expected
    assert got["id"].to_list() == list(range(3 * ROWS))

    # The column can be filtered on and selected alone.
    one_file = lf.filter(pl.col("file") == uris[2]).collect()
    assert one_file["id"].to_list() == list(range(2 * ROWS, 3 * ROWS))
    only_paths = lf.select("file").unique().sort("file").collect()
    assert only_paths["file"].to_list() == uris


def test_include_file_paths_in_now_mode_gives_signed_urls(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The difference to the default mode, pinned.
    _put_numbered(fake_hub, fake_bucket, 2)
    uri = _uri(fake_bucket, "data/")

    got = plhf.scan_bucket(uri, resolve="now", include_file_paths="file").collect()

    assert all(SIGNATURE in value for value in got["file"].to_list())


@pytest.mark.parametrize("scan_kwargs", [{"row_index_name": "row"}, {"n_rows": 25}])
def test_options_over_all_rows_scan_one_group(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int, scan_kwargs: dict
) -> None:
    _put_numbered(fake_hub, fake_bucket, 8)
    uri = _uri(fake_bucket, "data/")
    lf = plhf.scan_bucket(uri, **scan_kwargs)
    native = plhf.scan_bucket(uri, resolve="now", **scan_kwargs).collect()
    fake_hub.reset_log()

    got = lf.collect()

    assert_frame_equal(got, native)
    # Schema (1 file), then all other files in one burst.
    resolves = [count for kind, count in _request_runs(fake_hub) if kind == "resolve"]
    assert resolves == [1, 7]


def test_row_index_with_offset_and_filter(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int
) -> None:
    _put_numbered(fake_hub, fake_bucket, 8)
    uri = _uri(fake_bucket, "data/")
    options = {"row_index_name": "row", "row_index_offset": 100}

    def query(lf: pl.LazyFrame) -> pl.DataFrame:
        return lf.filter(pl.col("id") >= 55).head(8).collect()

    got = query(plhf.scan_bucket(uri, **options))

    assert got["row"].to_list() == list(range(155, 163))
    assert_frame_equal(got, query(plhf.scan_bucket(uri, resolve="now", **options)))


# ---- what a pickled plan holds ---------------------------------------------


def _secret_provider():  # pragma: no cover - never called
    raise AssertionError("the credential provider of a test was called")


@pytest.mark.filterwarnings("ignore:.*json.*:UserWarning")
def test_serialized_plan_has_no_storage_options(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    expected = _put_numbered(fake_hub, fake_bucket, 2)
    secret = "storage-option-secret-value"
    options = {"max_retries": 2, "aws_secret_access_key": secret}
    lf = plhf.scan_bucket(_uri(fake_bucket, "data/"), storage_options=options)
    # The options reach the native scans.
    assert_frame_equal(lf.collect(), expected)

    binary = lf.serialize(format="binary")

    assert secret.encode() not in binary
    assert b"aws_secret_access_key" not in binary
    assert b"storage_options" not in binary
    assert secret not in lf.serialize(format="json")
    # The plan that is read back runs without them.
    restored = pl.LazyFrame.deserialize(io.BytesIO(binary), format="binary")
    assert_frame_equal(restored.collect(), expected)


def test_pickled_source_has_no_credentials(fake_bucket: str) -> None:
    secret = "storage-option-secret-value"
    source = _source(
        fake_bucket,
        2,
        storage_options={"aws_secret_access_key": secret},
        credential_provider=_secret_provider,
        missing_columns="insert",
    )

    pickled = pickle.dumps(source)
    restored = pickle.loads(pickled)

    assert secret.encode() not in pickled
    assert b"_secret_provider" not in pickled
    assert b"storage_options" not in pickled
    assert b"credential_provider" not in pickled
    assert restored.scan_kwargs == {"missing_columns": "insert"}
    # The source that was pickled keeps its options.
    assert sorted(source.scan_kwargs) == [
        "credential_provider",
        "missing_columns",
        "storage_options",
    ]


# ---- an empty projection ---------------------------------------------------


@pytest.mark.parametrize("n_rows", [None, 25])
def test_empty_projection_keeps_the_row_count(
    fake_hub: FakeHub, fake_bucket: str, small_groups: int, n_rows: int | None
) -> None:
    # A frame without columns has no rows. If polars asks for no column, the
    # source reads one, so that the heights of its frames are right.
    _put_numbered(fake_hub, fake_bucket, 8)
    source = _source(fake_bucket, 8)

    frames = list(source([], None, n_rows, None))

    assert sum(frame.height for frame in frames) == (n_rows or 8 * ROWS)
    assert {tuple(frame.columns) for frame in frames} == {("id",)}


def test_narrowest_column() -> None:
    schema = pl.Schema(
        {"text": pl.String, "big": pl.Int64, "flag": pl.Boolean, "small": pl.Int16}
    )

    assert read._narrowest_column(schema) == "flag"
    assert read._narrowest_column(pl.Schema({"a": pl.String, "b": pl.Int64})) == "b"
    assert read._narrowest_column(pl.Schema({"a": pl.String, "b": pl.Binary})) == "a"


# ---- count_rows ------------------------------------------------------------


@pytest.mark.parametrize("path", ["wide/", "wide/*.parquet", "wide/w1.parquet"])
def test_count_rows_reads_the_footers_only(
    fake_hub: FakeHub, fake_bucket: str, path: str
) -> None:
    total = _put_wide(fake_hub, fake_bucket, WIDE_FILES, WIDE_ROWS)
    n_files = 1 if path.endswith("w1.parquet") else WIDE_FILES

    count = plhf.count_rows(_uri(fake_bucket, path))

    assert count == n_files * WIDE_ROWS
    assert len(_resolved(fake_hub)) == n_files
    gets = fake_hub.matching(origin=CDN, method="GET")
    assert all(request.range is not None for request in gets)
    assert len({request.path for request in gets}) == n_files
    assert 0 < fake_hub.cdn_bytes_served < total / 4
    # The same count as a scan, which reads a column.
    assert count == plhf.scan_bucket(_uri(fake_bucket, path)).collect().height


def test_count_rows_counts_files_with_different_columns(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    uri = _put_mixed(fake_hub, fake_bucket)
    text = pl.DataFrame({"id": ["a", "b", "c"]})
    fake_hub.put_parquet(fake_bucket, "mixed/d.parquet", text)

    # A scan of these files raises without options; a count does not.
    assert plhf.count_rows(uri) == 8


def test_count_rows_has_the_path_rules_and_errors_of_scan_bucket(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    _put_numbered(fake_hub, fake_bucket, 2)
    fake_hub.put(fake_bucket, "empty/e.parquet", b"")
    fake_hub.put(fake_bucket, "text/notes.txt", b"these bytes are not parquet")

    assert plhf.count_rows(_uri(fake_bucket, "data")) == 2 * ROWS
    with pytest.raises(FileNotFoundError, match="no parquet files matched"):
        plhf.count_rows(_uri(fake_bucket, "nope"))
    with pytest.raises(FileNotFoundError, match="not found"):
        plhf.count_rows("hf://buckets/fake-user/no-such-bucket/data")
    with pytest.raises(ValueError, match="a glob cannot end with"):
        plhf.count_rows(_uri(fake_bucket, "data/*/"))
    with pytest.raises(ValueError, match="is empty"):
        plhf.count_rows(_uri(fake_bucket, "empty/"))
    with pytest.raises(PermissionError, match="lacks access"):
        plhf.count_rows(_uri(fake_bucket, "data/"), token="unknown-token")
    with pytest.raises((pl.exceptions.PolarsError, OSError)) as error:
        plhf.count_rows(_uri(fake_bucket, "text/*"))
    for linked in exception_chain(error.value):
        assert_no_signed_url(str(linked), fake_hub)
