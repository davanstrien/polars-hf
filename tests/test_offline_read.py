"""Offline read tests: ``scan_bucket`` against the fake Hub (no network)."""

from __future__ import annotations

import httpx
import polars as pl
import pytest
from fakehub import CDN, HUB, SIGNATURE, FakeHub
from polars.testing import assert_frame_equal

import polars_hf as plhf


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _numbered_frame(start: int, rows: int) -> pl.DataFrame:
    ids = pl.int_range(start, start + rows, eager=True)
    return pl.DataFrame({"id": ids, "label": (ids % 3).cast(pl.String)})


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


def test_scan_is_lazy_and_resolves_one_head_per_file(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    n_files = 4
    for i in range(n_files):
        frame = _numbered_frame(i * 100, 100)
        fake_hub.put_parquet(fake_bucket, f"data/run_{i}.parquet", frame)

    lf = plhf.scan_bucket(_uri(fake_bucket, "data/run_*.parquet"))

    # Building the LazyFrame transfers no file data: nothing reaches the cdn
    # and the resolve endpoint only sees HEAD requests, one per file.
    assert fake_hub.matching(origin=CDN) == []
    assert fake_hub.matching(origin=HUB, method="GET", path_contains="/resolve/") == []
    heads = fake_hub.matching(origin=HUB, method="HEAD", path_contains="/resolve/")
    assert len(heads) == n_files
    assert sorted(r.path for r in heads) == [
        f"/buckets/{fake_bucket}/resolve/data/run_{i}.parquet" for i in range(n_files)
    ]

    assert lf.collect().height == n_files * 100
    assert len(fake_hub.matching(origin=CDN, method="GET")) > 0


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


def test_native_parquet_scan(fake_hub: FakeHub, fake_bucket: str) -> None:
    # scan_bucket must produce a NATIVE parquet scan over a signed URL (range
    # reads + pushdown), not a PYTHON SCAN that buffers files.
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))

    plan = plhf.scan_bucket(_uri(fake_bucket, "one.parquet")).explain()

    assert "Parquet SCAN" in plan
    assert "PYTHON SCAN" not in plan


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


# ---- scripted faults (current behaviour; the retry tests are in ------------
# ---- test_known_bugs.py) ---------------------------------------------------


def test_fault_on_resolve_is_consumed_once(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))
    uri = _uri(fake_bucket, "one.parquet")
    fake_hub.add_fault(HUB, "HEAD", r"/resolve/one\.parquet$", 500)

    with pytest.raises(httpx.HTTPStatusError) as error:
        plhf.scan_bucket(uri)
    assert error.value.response.status_code == 500

    # The fault was scripted once: the next call succeeds.
    assert plhf.scan_bucket(uri).collect().height == 10
    statuses = [
        r.status for r in fake_hub.matching(origin=HUB, path_contains="/resolve/")
    ]
    assert statuses == [500, 302]


def test_expired_signed_url_fails_at_collect(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))
    lf = plhf.scan_bucket(_uri(fake_bucket, "one.parquet"))
    # An expired presigned URL: the cdn refuses every request from now on.
    for method in ("HEAD", "GET"):
        fake_hub.add_fault(
            CDN, method, r"^/xet-bridge-us/", 403, times=1000, body=b"expired"
        )

    with pytest.raises((pl.exceptions.PolarsError, OSError)):
        lf.collect()

    assert all(r.status == 403 for r in fake_hub.matching(origin=CDN))


def test_signed_url_requires_its_signature(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))
    plhf.scan_bucket(_uri(fake_bucket, "one.parquet")).collect()
    served = fake_hub.matching(origin=CDN, method="GET")[0]

    base = f"{fake_hub.cdn_endpoint}{served.path}"
    good = httpx.get(f"{base}?X-Amz-Signature={SIGNATURE}")
    bad = httpx.get(f"{base}?X-Amz-Signature=tampered")

    assert good.status_code == 200
    assert bad.status_code == 401
