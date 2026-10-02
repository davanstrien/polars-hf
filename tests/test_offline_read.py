"""Offline read tests: ``scan_bucket`` against the fake Hub (no network).

The tests of the requests of ``scan_bucket()`` run in both ``resolve`` modes:
``"now"`` resolves every file in the call, ``"collect"`` (the default) resolves
none of the listed files before the query runs. ``test_collect_time.py`` has
the tests of what a query requests.
"""

from __future__ import annotations

import polars as pl
import pytest
from fakehub import CDN, HUB, FakeHub
from huggingface_hub import HfFileSystem
from huggingface_hub.errors import HfHubHTTPError
from polars.testing import assert_frame_equal

import polars_hf as plhf
from polars_hf import read


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _numbered_frame(start: int, rows: int) -> pl.DataFrame:
    ids = pl.int_range(start, start + rows, eager=True)
    return pl.DataFrame({"id": ids, "label": (ids % 3).cast(pl.String)})


MODES = ("collect", "now")


def _resolves_in_the_call(mode: str, n_files: int) -> int:
    """The ``resolve`` requests that ``scan_bucket()`` sends for N listed files."""
    return n_files if mode == "now" else 0


def _wide_frame(rows: int) -> pl.DataFrame:
    """A frame whose ``payload`` column dominates the (uncompressed) file size."""
    ids = pl.int_range(0, rows, eager=True)
    payload = (ids * 2654435761 % 1000003).cast(pl.String) + "-" + ids.cast(pl.String)
    return pl.DataFrame({"id": ids, "payload": payload})


def test_single_file_scan(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = _numbered_frame(0, 500)
    fake_hub.put_parquet(fake_bucket, "data/one.parquet", df)

    got = plhf.scan_bucket(_uri(fake_bucket, "data/one.parquet")).collect()

    assert_frame_equal(got, df)


def test_glob_scan(fake_hub: FakeHub, fake_bucket: str) -> None:
    frames = []
    for i in range(3):
        frame = _numbered_frame(i * 100, 100)
        fake_hub.put_parquet(fake_bucket, f"data/run_{i}.parquet", frame)
        frames.append(frame)
    fake_hub.put_parquet(fake_bucket, "data/other.parquet", _numbered_frame(900, 5))

    got = plhf.scan_bucket(_uri(fake_bucket, "data/run_*.parquet")).collect()

    assert_frame_equal(got.sort("id"), pl.concat(frames))


def test_directory_scan_is_recursive(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 10))
    fake_hub.put_parquet(fake_bucket, "data/sub/b.parquet", _numbered_frame(10, 10))
    fake_hub.put(fake_bucket, "data/notes.txt", b"not parquet")

    got = plhf.scan_bucket(_uri(fake_bucket, "data")).collect()

    assert_frame_equal(got.sort("id"), _numbered_frame(0, 20))


@pytest.mark.usefixtures("allow_signed_urls_in_plan")
@pytest.mark.parametrize("mode", MODES)
def test_scan_is_lazy_and_resolves_one_head_per_file(
    fake_hub: FakeHub, fake_bucket: str, mode: str
) -> None:
    n_files = 4
    for i in range(n_files):
        frame = _numbered_frame(i * 100, 100)
        fake_hub.put_parquet(fake_bucket, f"data/run_{i}.parquet", frame)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/run_*.parquet"), resolve=mode)

    # Building the LazyFrame transfers no file data: nothing reaches the cdn
    # and the resolve endpoint only sees HEAD requests: one per file with
    # resolve="now", none with resolve="collect".
    assert fake_hub.matching(origin=CDN) == []
    assert fake_hub.matching(origin=HUB, method="GET", path_contains="/resolve/") == []
    heads = fake_hub.matching(origin=HUB, method="HEAD", path_contains="/resolve/")
    assert len(heads) == _resolves_in_the_call(mode, n_files)

    assert lf.collect().height == n_files * 100
    assert len(fake_hub.matching(origin=CDN, method="GET")) > 0
    # After the query, both modes have sent one HEAD request per file.
    assert fake_hub.matching(origin=HUB, method="GET", path_contains="/resolve/") == []
    heads = fake_hub.matching(origin=HUB, method="HEAD", path_contains="/resolve/")
    assert sorted(r.path for r in heads) == [
        f"/buckets/{fake_bucket}/resolve/data/run_{i}.parquet" for i in range(n_files)
    ]


def test_authorization_stays_on_the_hub_origin(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/one.parquet", _numbered_frame(0, 50))

    plhf.scan_bucket(_uri(fake_bucket, "data/one.parquet")).collect()

    hub_requests = fake_hub.matching(origin=HUB)
    cdn_requests = fake_hub.matching(origin=CDN)
    assert hub_requests and all(r.has_authorization for r in hub_requests)
    assert cdn_requests and not any(r.has_authorization for r in cdn_requests)


def test_projection_reads_fewer_bytes_than_the_file(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    df = _wide_frame(200_000)
    size = fake_hub.put_parquet(
        fake_bucket, "wide.parquet", df, compression="uncompressed"
    )
    lf = plhf.scan_bucket(_uri(fake_bucket, "wide.parquet"))

    got = lf.select("id").collect()

    assert_frame_equal(got, df.select("id"))
    served = fake_hub.cdn_bytes_served
    assert 0 < served < size / 2
    # Every data request is a range request.
    gets = fake_hub.matching(origin=CDN, method="GET")
    assert all(r.range is not None for r in gets)


def test_scan_kwargs_forwarded_mixed_schemas(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "mixed/a.parquet", pl.DataFrame({"id": [1, 2]}))
    fake_hub.put_parquet(
        fake_bucket, "mixed/b.parquet", pl.DataFrame({"id": [3], "note": ["x"]})
    )
    uri = _uri(fake_bucket, "mixed/*.parquet")

    with pytest.raises(pl.exceptions.PolarsError):
        plhf.scan_bucket(uri).collect()
    got = plhf.scan_bucket(
        uri, missing_columns="insert", extra_columns="ignore"
    ).collect()

    assert sorted(got["id"].to_list()) == [1, 2, 3]


def test_missing_bucket_raises_file_not_found(fake_hub: FakeHub) -> None:
    with pytest.raises(FileNotFoundError):
        plhf.scan_bucket("hf://buckets/fake-user/no-such-bucket/data")


def test_empty_directory_raises_file_not_found(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put(fake_bucket, "data/notes.txt", b"not parquet")

    with pytest.raises(FileNotFoundError, match="no parquet files matched"):
        plhf.scan_bucket(_uri(fake_bucket, "data"))


def test_default_token_is_the_staging_ci_token(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    from conftest import STAGING_TOKEN

    fake_hub.put_parquet(fake_bucket, "data/one.parquet", _numbered_frame(0, 5))

    plhf.scan_bucket(_uri(fake_bucket, "data/one.parquet")).collect()

    sent = {request.authorization for request in fake_hub.matching(origin=HUB)}
    assert sent == {f"Bearer {STAGING_TOKEN}"}


def test_explicit_token_reaches_the_hub(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.accept_token("hf_explicit_read_token")
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 5))
    fake_hub.put_parquet(fake_bucket, "data/b.parquet", _numbered_frame(5, 5))

    lf = plhf.scan_bucket(_uri(fake_bucket, "data"), token="hf_explicit_read_token")

    assert lf.collect().height == 10
    sent = {request.authorization for request in fake_hub.matching(origin=HUB)}
    assert sent == {"Bearer hf_explicit_read_token"}


def test_missing_bucket_message_names_the_bucket(fake_hub: FakeHub) -> None:
    uri = "hf://buckets/fake-user/no-such-bucket/data/one.parquet"

    with pytest.raises(FileNotFoundError) as error:
        plhf.scan_bucket(uri)

    assert "'fake-user/no-such-bucket' not found" in str(error.value)
    assert uri in str(error.value)


@pytest.mark.parametrize("path", ["nope.parquet", "nope", "nope/*.parquet", "nope/"])
def test_missing_path_message_has_the_uri(
    fake_hub: FakeHub, fake_bucket: str, path: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/one.parquet", _numbered_frame(0, 5))
    uri = _uri(fake_bucket, path)

    with pytest.raises(FileNotFoundError) as error:
        plhf.scan_bucket(uri)

    assert uri in str(error.value)


# ---- what a path names -----------------------------------------------------


@pytest.mark.parametrize("name", ["table", "table.bin", "table.parquet.bak"])
def test_single_file_without_parquet_extension(
    fake_hub: FakeHub, fake_bucket: str, name: str
) -> None:
    df = _numbered_frame(0, 20)
    fake_hub.put_parquet(fake_bucket, f"data/{name}", df)
    # A string-prefix sibling of the file name.
    fake_hub.put_parquet(fake_bucket, f"data/{name}2", _numbered_frame(50, 5))

    got = plhf.scan_bucket(_uri(fake_bucket, f"data/{name}")).collect()

    assert_frame_equal(got, df)


def test_trailing_slash_scans_the_directory(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 10))
    fake_hub.put_parquet(fake_bucket, "data/sub/b.pq", _numbered_frame(10, 10))

    got = plhf.scan_bucket(_uri(fake_bucket, "data/")).collect()

    assert_frame_equal(got.sort("id"), _numbered_frame(0, 20))


def test_whole_bucket_scan(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "a.parquet", _numbered_frame(0, 10))
    fake_hub.put_parquet(fake_bucket, "x/y/b.parquet", _numbered_frame(10, 10))
    fake_hub.put(fake_bucket, "README.md", b"not parquet")

    for uri in (f"hf://buckets/{fake_bucket}", f"hf://buckets/{fake_bucket}/"):
        got = plhf.scan_bucket(uri).collect()
        assert_frame_equal(got.sort("id"), _numbered_frame(0, 20))


def test_recursive_glob(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 10))
    fake_hub.put_parquet(fake_bucket, "data/x/y/b.parquet", _numbered_frame(10, 10))
    fake_hub.put(fake_bucket, "data/x/notes.txt", b"not parquet")
    fake_hub.put_parquet(fake_bucket, "data2/c.parquet", _numbered_frame(90, 5))

    got = plhf.scan_bucket(_uri(fake_bucket, "data/**/*.parquet")).collect()

    assert_frame_equal(got.sort("id"), _numbered_frame(0, 20))


def test_at_sign_in_path_reads(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = _numbered_frame(0, 5)
    fake_hub.put_parquet(fake_bucket, "exports@2026/user@example.com.parquet", df)

    single = _uri(fake_bucket, "exports@2026/user@example.com.parquet")
    directory = _uri(fake_bucket, "exports@2026")

    assert_frame_equal(plhf.scan_bucket(single).collect(), df)
    assert_frame_equal(plhf.scan_bucket(directory).collect(), df)


def test_max_retries_storage_option_is_forwarded(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The current Polars spelling of the deprecated retries= scan option.
    df = _numbered_frame(0, 10)
    fake_hub.put_parquet(fake_bucket, "one.parquet", df)
    uri = _uri(fake_bucket, "one.parquet")

    got = plhf.scan_bucket(uri, storage_options={"max_retries": 3}).collect()

    assert_frame_equal(got, df)


# ---- Hub requests made by scan_bucket() ------------------------------------


def _hub_calls(fake_hub: FakeHub) -> list[tuple[str, str, int]]:
    """``(method, kind, status)`` of every hub request; kind is tree or resolve."""
    calls = []
    for request in fake_hub.matching(origin=HUB):
        kind = "tree" if "/tree" in request.path else "resolve"
        assert f"/{kind}" in request.path
        calls.append((request.method, kind, request.status))
    return calls


@pytest.mark.usefixtures("allow_signed_urls_in_plan")
@pytest.mark.parametrize("mode", MODES)
def test_single_parquet_file_is_one_request(
    fake_hub: FakeHub, fake_bucket: str, mode: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/one.parquet", _numbered_frame(0, 5))

    plhf.scan_bucket(_uri(fake_bucket, "data/one.parquet"), resolve=mode)

    assert _hub_calls(fake_hub) == [("HEAD", "resolve", 302)]
    assert fake_hub.matching(origin=CDN) == []


@pytest.mark.usefixtures("allow_signed_urls_in_plan")
@pytest.mark.parametrize("mode", MODES)
def test_directory_of_n_files_is_one_listing_and_n_resolves(
    fake_hub: FakeHub, fake_bucket: str, mode: str
) -> None:
    n_files = 20
    for i in range(n_files):
        frame = _numbered_frame(i * 10, 10)
        fake_hub.put_parquet(fake_bucket, f"data/sub{i % 3}/part-{i}.parquet", frame)
    fake_hub.put(fake_bucket, "data/_SUCCESS", b"")

    lf = plhf.scan_bucket(_uri(fake_bucket, "data"), resolve=mode)

    # The path is tried as a file first; the listing then names the files.
    calls = _hub_calls(fake_hub)
    assert calls[:2] == [("HEAD", "resolve", 404), ("GET", "tree", 200)]
    in_the_call = _resolves_in_the_call(mode, n_files)
    assert calls[2:] == [("HEAD", "resolve", 302)] * in_the_call
    # No file data, and no request to the cdn, before collect().
    assert fake_hub.matching(origin=CDN) == []

    assert lf.collect().height == n_files * 10
    # N resolve requests in total, in the call or in the query.
    assert _hub_calls(fake_hub)[2:] == [("HEAD", "resolve", 302)] * n_files
    resolved = fake_hub.matching(origin=HUB, method="HEAD")
    assert len({request.path for request in resolved}) == n_files + 1


@pytest.mark.usefixtures("allow_signed_urls_in_plan")
@pytest.mark.parametrize("mode", MODES)
def test_glob_is_one_listing_and_one_resolve_per_match(
    fake_hub: FakeHub, fake_bucket: str, mode: str
) -> None:
    for i in range(3):
        frame = _numbered_frame(i * 10, 10)
        fake_hub.put_parquet(fake_bucket, f"data/run_{i}.parquet", frame)
    fake_hub.put_parquet(fake_bucket, "data/other.parquet", _numbered_frame(90, 5))

    plhf.scan_bucket(_uri(fake_bucket, "data/run_*.parquet"), resolve=mode)

    resolves = [("HEAD", "resolve", 302)] * _resolves_in_the_call(mode, 3)
    assert _hub_calls(fake_hub) == [("GET", "tree", 200)] + resolves
    # The glob is in the last segment: its directory is listed, not the subtree.
    listing = fake_hub.matching(origin=HUB, method="GET")[0]
    assert listing.path.endswith("/tree/data")
    assert listing.query == "recursive=false"


@pytest.mark.usefixtures("allow_signed_urls_in_plan")
@pytest.mark.parametrize("path", ["data", "data/**/*.parquet"])
def test_directory_and_recursive_glob_list_the_subtree(
    fake_hub: FakeHub, fake_bucket: str, path: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 5))
    fake_hub.put_parquet(fake_bucket, "data/x/y/b.parquet", _numbered_frame(5, 5))

    plhf.scan_bucket(_uri(fake_bucket, path), resolve="now")

    listings = fake_hub.matching(origin=HUB, method="GET")
    assert [request.query for request in listings] == ["recursive=true"]
    resolved = fake_hub.matching(origin=HUB, method="HEAD")
    assert [r.status for r in resolved if r.status == 302] == [302, 302]


def test_one_directory_glob_does_not_read_sub_directories(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 5))
    for i in range(30):
        fake_hub.put_parquet(
            fake_bucket, f"data/sub/{i}.parquet", _numbered_frame(i, 1)
        )
    fake_hub.tree_page_size = 10

    got = plhf.scan_bucket(_uri(fake_bucket, "data/*.parquet")).collect()

    assert_frame_equal(got, _numbered_frame(0, 5))
    # One page: the 30 files of data/sub are one directory entry.
    assert len(fake_hub.matching(origin=HUB, method="GET")) == 1


@pytest.mark.parametrize("path", ["data", "data/*.parquet"])
def test_paginated_listing_is_read_to_the_end(
    fake_hub: FakeHub, fake_bucket: str, path: str
) -> None:
    n_files = 7
    for i in range(n_files):
        fake_hub.put_parquet(fake_bucket, f"data/p{i}.parquet", _numbered_frame(i, 1))
    fake_hub.tree_page_size = 3

    got = plhf.scan_bucket(_uri(fake_bucket, path)).collect()

    assert sorted(got["id"].to_list()) == list(range(n_files))
    listings = fake_hub.matching(origin=HUB, method="GET")
    assert len(listings) == 3
    assert all(request.has_authorization for request in listings)
    resolved = fake_hub.matching(origin=HUB, method="HEAD")
    assert len([r for r in resolved if r.status == 302]) == n_files


def test_glob_with_trailing_slash_is_rejected(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/x/a.parquet", _numbered_frame(0, 5))

    with pytest.raises(ValueError, match="a glob cannot end with '/'"):
        plhf.scan_bucket(_uri(fake_bucket, "data/*/"))

    assert fake_hub.matching(origin=HUB) == []


def test_explicit_glob_that_selects_a_non_parquet_file_fails_at_collect(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # A glob does not filter by extension: polars reports the bad file.
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 5))
    fake_hub.put(fake_bucket, "data/notes.txt", b"these bytes are not parquet")

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/*"))

    with pytest.raises(pl.exceptions.PolarsError):
        lf.collect()


def test_file_without_parquet_extension_is_one_request(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/table", _numbered_frame(0, 5))

    plhf.scan_bucket(_uri(fake_bucket, "data/table"))

    assert _hub_calls(fake_hub) == [("HEAD", "resolve", 302)]


@pytest.mark.usefixtures("allow_signed_urls_in_plan")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("path", ["data/", ""])
def test_trailing_slash_and_whole_bucket_skip_the_file_request(
    fake_hub: FakeHub, fake_bucket: str, path: str, mode: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 5))
    fake_hub.put_parquet(fake_bucket, "data/b.parquet", _numbered_frame(5, 5))

    plhf.scan_bucket(_uri(fake_bucket, path), resolve=mode)

    resolves = [("HEAD", "resolve", 302)] * _resolves_in_the_call(mode, 2)
    assert _hub_calls(fake_hub) == [("GET", "tree", 200)] + resolves


@pytest.mark.parametrize("path", ["nope", "nope.parquet", "data/nope.bin"])
def test_missing_path_is_one_resolve_and_one_listing(
    fake_hub: FakeHub, fake_bucket: str, path: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 5))

    with pytest.raises(FileNotFoundError):
        plhf.scan_bucket(_uri(fake_bucket, path))

    assert _hub_calls(fake_hub) == [("HEAD", "resolve", 404), ("GET", "tree", 200)]


@pytest.mark.parametrize("path", ["train", "train/"])
def test_string_prefix_sibling_adds_no_listing_pages(
    fake_hub: FakeHub, fake_bucket: str, path: str
) -> None:
    # 'train_full/' and 'train.parquet' start with the string 'train'. They
    # are not part of the directory 'train/' and must not be listed with it.
    fake_hub.put_parquet(fake_bucket, "train/a.parquet", _numbered_frame(0, 5))
    fake_hub.put_parquet(fake_bucket, "train/b.parquet", _numbered_frame(5, 5))
    fake_hub.put_parquet(fake_bucket, "train.parquet", _numbered_frame(90, 1))
    for i in range(40):
        frame = _numbered_frame(100 + i, 1)
        fake_hub.put_parquet(fake_bucket, f"train_full/{i}.parquet", frame)
    fake_hub.tree_page_size = 10

    got = plhf.scan_bucket(_uri(fake_bucket, path)).collect()

    assert_frame_equal(got.sort("id"), _numbered_frame(0, 10))
    listings = fake_hub.matching(origin=HUB, method="GET")
    assert len(listings) == 1
    assert listings[0].path.endswith("/tree/train/")
    assert listings[0].query == "recursive=true"


def test_invalid_glob_makes_no_request(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 5))

    with pytest.raises(ValueError, match="must be a whole path segment"):
        plhf.scan_bucket(_uri(fake_bucket, "data/**.parquet"))

    assert fake_hub.matching(origin=HUB) == []


@pytest.mark.usefixtures("allow_signed_urls_in_plan")
@pytest.mark.parametrize("mode", MODES)
def test_directory_named_like_a_file_costs_one_extra_resolve(
    fake_hub: FakeHub, fake_bucket: str, mode: str
) -> None:
    fake_hub.put_parquet(
        fake_bucket, "out.parquet/part-0.parquet", _numbered_frame(0, 5)
    )

    # The requests of the call and of one query.
    plhf.scan_bucket(_uri(fake_bucket, "out.parquet"), resolve=mode).collect()

    assert _hub_calls(fake_hub) == [
        ("HEAD", "resolve", 404),
        ("GET", "tree", 200),
        ("HEAD", "resolve", 302),
    ]


def test_scan_does_not_touch_the_filesystem_cache(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # sink_bucket and user code share one cached HfFileSystem instance per
    # token: a scan must not drop its directory cache.
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 10))
    fs = HfFileSystem()
    fs.ls(f"buckets/{fake_bucket}/data")
    cached = dict(fs.dircache)
    assert cached

    plhf.scan_bucket(_uri(fake_bucket, "data")).collect()

    assert dict(HfFileSystem().dircache) == cached


def test_scan_after_write_sees_new_files(fake_hub: FakeHub, fake_bucket: str) -> None:
    base = _uri(fake_bucket, "grow")
    plhf.sink_bucket(_numbered_frame(0, 10), f"{base}/a.parquet")
    assert plhf.scan_bucket(base).collect().height == 10

    plhf.sink_bucket(_numbered_frame(10, 10), f"{base}/b.parquet")

    assert_frame_equal(
        plhf.scan_bucket(base).collect().sort("id"), _numbered_frame(0, 20)
    )


# ---- redirects on the Hub origin -------------------------------------------


def test_relative_redirect_is_followed_with_the_token(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The Hub answers a renamed bucket with a relative redirect.
    df = _numbered_frame(0, 10)
    fake_hub.put_parquet(fake_bucket, "new/one.parquet", df)
    moved = f"/buckets/{fake_bucket}/resolve/new%2Fone.parquet"
    fake_hub.add_fault(
        HUB, "HEAD", r"/resolve/old/one\.parquet$", 307, headers={"Location": moved}
    )

    got = plhf.scan_bucket(_uri(fake_bucket, "old/one.parquet")).collect()

    assert_frame_equal(got, df)
    heads = fake_hub.matching(origin=HUB, method="HEAD")
    assert [r.status for r in heads] == [307, 302]
    assert all(r.has_authorization for r in heads)
    assert not any(r.has_authorization for r in fake_hub.matching(origin=CDN))


# ---- errors ----------------------------------------------------------------


@pytest.mark.parametrize("path", ["data/one.parquet", "data", "data/*.parquet"])
def test_unknown_token_raises_permission_error(
    fake_hub: FakeHub, fake_bucket: str, path: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/one.parquet", _numbered_frame(0, 5))
    uri = _uri(fake_bucket, path)

    with pytest.raises(PermissionError) as error:
        plhf.scan_bucket(uri, token="hf_unknown_token")

    message = str(error.value)
    assert f"'{fake_bucket}'" in message and "lacks access" in message
    assert "hf_unknown_token" not in message
    cause = error.value.__cause__
    assert isinstance(cause, HfHubHTTPError)
    assert cause.response.status_code == 401
    # One refused request: a 401 is not retried.
    assert len(fake_hub.matching(origin=HUB)) == 1


@pytest.mark.parametrize(
    ("method", "route", "path"),
    [
        ("HEAD", r"/resolve/", "data/one.parquet"),
        ("HEAD", r"/resolve/", "data"),
        ("GET", r"/tree/", "data"),
    ],
)
def test_forbidden_raises_permission_error(
    fake_hub: FakeHub, fake_bucket: str, method: str, route: str, path: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/one.parquet", _numbered_frame(0, 5))
    fake_hub.add_fault(HUB, method, route, 403)

    with pytest.raises(PermissionError) as error:
        plhf.scan_bucket(_uri(fake_bucket, path))

    assert f"'{fake_bucket}'" in str(error.value)
    assert "lacks access" in str(error.value)
    cause = error.value.__cause__
    assert isinstance(cause, HfHubHTTPError)
    assert cause.response.status_code == 403


def test_other_http_error_keeps_its_type(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))
    uri = _uri(fake_bucket, "one.parquet")
    fake_hub.add_fault(HUB, "HEAD", r"/resolve/one\.parquet$", 400)

    with pytest.raises(HfHubHTTPError) as error:
        plhf.scan_bucket(uri)
    assert error.value.response.status_code == 400
    assert not isinstance(error.value, (PermissionError, FileNotFoundError))

    # The fault was scripted once: the next call succeeds.
    assert plhf.scan_bucket(uri).collect().height == 10
    statuses = [
        r.status for r in fake_hub.matching(origin=HUB, path_contains="/resolve/")
    ]
    assert statuses == [400, 302]


def test_server_error_on_resolve_is_retried(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 10))
    fake_hub.put_parquet(fake_bucket, "data/b.parquet", _numbered_frame(10, 10))
    fake_hub.add_fault(HUB, "HEAD", r"/resolve/data/b\.parquet$", 500, times=2)

    got = plhf.scan_bucket(_uri(fake_bucket, "data")).collect()

    assert got.height == 20
    statuses = [
        r.status for r in fake_hub.matching(origin=HUB, path_contains="/b.parquet")
    ]
    assert statuses == [500, 500, 302]


def test_persistent_rate_limit_raises_after_bounded_retries(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))
    fake_hub.add_fault(HUB, "HEAD", r"/resolve/one\.parquet$", 429, times=100)

    with pytest.raises(HfHubHTTPError) as error:
        plhf.scan_bucket(_uri(fake_bucket, "one.parquet"))

    assert error.value.response.status_code == 429
    attempts = fake_hub.matching(origin=HUB, path_contains="/resolve/")
    assert len(attempts) == read._MAX_RETRIES + 1


def test_empty_single_file_is_rejected(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put(fake_bucket, "data/empty.parquet", b"")
    uri = _uri(fake_bucket, "data/empty.parquet")

    with pytest.raises(ValueError, match="is empty") as error:
        plhf.scan_bucket(uri)

    assert uri in str(error.value)
    assert fake_hub.matching(origin=CDN) == []


def test_empty_file_in_directory_is_rejected(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", _numbered_frame(0, 10))
    fake_hub.put(fake_bucket, "data/empty.parquet", b"")

    with pytest.raises(ValueError, match="is empty") as error:
        plhf.scan_bucket(_uri(fake_bucket, "data"))

    assert _uri(fake_bucket, "data/empty.parquet") in str(error.value)
    # Found in the listing: no resolve request was spent on a file. (The one
    # HEAD is the try of "data" as a file.)
    heads = fake_hub.matching(origin=HUB, method="HEAD")
    assert [request.status for request in heads] == [404]


def test_file_served_without_redirect_is_rejected(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # huggingface.co answers the resolve request of an empty file, and of a
    # file that is not Xet-backed, with a direct 200. That URL needs the
    # token, so it must not reach polars. (The fake, like the Hub CI
    # instance, redirects every file; the direct answer is scripted.)
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))
    uri = _uri(fake_bucket, "one.parquet")
    fake_hub.add_fault(HUB, "HEAD", r"/resolve/one\.parquet$", 200, body=b"x" * 64)

    with pytest.raises(RuntimeError, match="did not redirect") as error:
        plhf.scan_bucket(uri)
    assert uri in str(error.value)

    fake_hub.add_fault(HUB, "HEAD", r"/resolve/one\.parquet$", 200)
    with pytest.raises(ValueError, match="is empty"):
        plhf.scan_bucket(uri)

    assert fake_hub.matching(origin=HUB, method="GET") == []
    assert fake_hub.matching(origin=CDN) == []


@pytest.mark.usefixtures("allow_signed_urls_in_plan")
@pytest.mark.parametrize("mode", MODES)
def test_expired_signed_url_fails_at_collect(
    fake_hub: FakeHub, fake_bucket: str, mode: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))
    lf = plhf.scan_bucket(_uri(fake_bucket, "one.parquet"), resolve=mode)
    # An expired presigned URL: the cdn refuses every request from now on.
    for method in ("HEAD", "GET"):
        fake_hub.add_fault(
            CDN, method, r"^/xet-bridge-us/", 403, times=1000, body=b"expired"
        )

    with pytest.raises((pl.exceptions.PolarsError, OSError)):
        lf.collect()

    assert all(r.status == 403 for r in fake_hub.matching(origin=CDN))
