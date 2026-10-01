"""Known bugs, written as tests of the DESIRED behaviour.

Every test here is ``xfail(strict=True)``: it fails today for the stated
reason, and the suite goes red as soon as it starts to pass. The pull request
that fixes a bug removes the marker (or moves the test to the matching test
module). ``raises=`` pins the failure to the documented symptom, so an
unrelated breakage does not hide behind the marker.

All tests run offline against the fake Hub (see ``fakehub.py``).

The write-path bugs (a, b, c, d, e, n, o) are fixed; their tests are in
``test_offline_write.py``.
"""

from __future__ import annotations

import polars as pl
import pytest
from fakehub import HUB, SIGNATURE, FakeHub
from polars.testing import assert_frame_equal

import polars_hf as plhf


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


# ---- URI parsing -----------------------------------------------------------

# Bug f is fixed: its test has no marker and stays as a regression test.


def test_at_sign_in_file_path_is_not_a_revision() -> None:
    bp = plhf.parse_bucket_uri("hf://buckets/ns/name/exports/user@example.com.parquet")

    assert bp.bucket_id == "ns/name"
    assert bp.path == "exports/user@example.com.parquet"


# ---- read path -------------------------------------------------------------

# Bugs g to l and p are fixed: their tests have no marker and stay as
# regression tests. Bug m is open.


def test_literal_bracket_file_name_reads_that_file(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    bracket = pl.DataFrame({"which": ["data[1].parquet"]})
    plain = pl.DataFrame({"which": ["data1.parquet"]})
    fake_hub.put_parquet(fake_bucket, "g/data[1].parquet", bracket)
    fake_hub.put_parquet(fake_bucket, "g/data1.parquet", plain)

    got = plhf.scan_bucket(_uri(fake_bucket, "g/data[1].parquet")).collect()

    assert_frame_equal(got, bracket)


def test_directory_with_parquet_suffix_scans_as_directory(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    parts = [pl.DataFrame({"x": [1, 2]}), pl.DataFrame({"x": [3]})]
    fake_hub.put_parquet(fake_bucket, "h/out.parquet/part-0.parquet", parts[0])
    fake_hub.put_parquet(fake_bucket, "h/out.parquet/part-1.parquet", parts[1])

    got = plhf.scan_bucket(_uri(fake_bucket, "h/out.parquet")).collect()

    assert_frame_equal(got.sort("x"), pl.concat(parts))


def test_star_glob_does_not_scan_sub_directories(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "i/a.parquet", pl.DataFrame({"x": [1]}))
    fake_hub.put_parquet(fake_bucket, "i/b.parquet", pl.DataFrame({"x": [2]}))
    fake_hub.put_parquet(fake_bucket, "i/sub/c.parquet", pl.DataFrame({"x": [3]}))

    got = plhf.scan_bucket(_uri(fake_bucket, "i/*")).collect()

    assert sorted(got["x"].to_list()) == [1, 2]


def test_directory_scan_includes_pq_and_upper_case_parquet(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "j/a.parquet", pl.DataFrame({"x": [1]}))
    fake_hub.put_parquet(fake_bucket, "j/b.pq", pl.DataFrame({"x": [2]}))
    fake_hub.put_parquet(fake_bucket, "j/c.PARQUET", pl.DataFrame({"x": [3]}))

    got = plhf.scan_bucket(_uri(fake_bucket, "j")).collect()

    assert sorted(got["x"].to_list()) == [1, 2, 3]


def test_missing_single_file_raises_file_not_found(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The error may be raised by scan_bucket() or later, by collect().
    with pytest.raises(FileNotFoundError):
        plhf.scan_bucket(_uri(fake_bucket, "nope.parquet")).collect()


@pytest.mark.parametrize("status", [429, 503])
def test_transient_resolve_error_is_retried(
    fake_hub: FakeHub, fake_bucket: str, status: int
) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    fake_hub.put_parquet(fake_bucket, "one.parquet", df)
    for method in ("HEAD", "GET"):
        fake_hub.add_fault(
            HUB, method, r"/resolve/one\.parquet$", status, headers={"Retry-After": "0"}
        )

    got = plhf.scan_bucket(_uri(fake_bucket, "one.parquet")).collect()

    assert_frame_equal(got, df)
    resolves = fake_hub.matching(origin=HUB, path_contains="/resolve/one.parquet")
    assert len(resolves) >= 2
    assert resolves[0].status == status


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug m: the query plan prints and serializes the presigned URL, "
    "signature included",
)
@pytest.mark.filterwarnings("ignore:.*json.*:UserWarning")
def test_presigned_url_not_in_plan(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", pl.DataFrame({"a": [1]}))

    lf = plhf.scan_bucket(_uri(fake_bucket, "one.parquet"))

    assert SIGNATURE not in lf.explain()
    assert SIGNATURE not in lf.explain(optimized=False)
    assert SIGNATURE.encode() not in lf.serialize(format="binary")
    assert SIGNATURE not in lf.serialize(format="json")


def test_directory_scan_with_prefix_sibling(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    inside = pl.DataFrame({"x": [1]})
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", inside)
    fake_hub.put_parquet(fake_bucket, "data.parquet", pl.DataFrame({"x": [2]}))
    fake_hub.put_parquet(fake_bucket, "data2/b.parquet", pl.DataFrame({"x": [3]}))

    got = plhf.scan_bucket(_uri(fake_bucket, "data")).collect()

    assert_frame_equal(got, inside)
