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

import httpx
import huggingface_hub
import polars as pl
import pytest
from fakehub import HUB, SIGNATURE, FakeHub
from polars.testing import assert_frame_equal

import polars_hf as plhf

HUB_MAJOR = int(huggingface_hub.__version__.split(".")[0])


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


# ---- URI parsing -----------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="bug f: any '@' in the URI is treated as a revision, including one "
    "in the file path",
)
def test_at_sign_in_file_path_is_not_a_revision() -> None:
    bp = plhf.parse_bucket_uri("hf://buckets/ns/name/exports/user@example.com.parquet")

    assert bp.bucket_id == "ns/name"
    assert bp.path == "exports/user@example.com.parquet"


# ---- read path -------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug g: a path with '[' and ']' is globbed, so 'data[1].parquet' is "
    "read as the character class and matches 'data1.parquet'. Intended fix: "
    "try the literal path first, then fall back to glob",
)
def test_literal_bracket_file_name_reads_that_file(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    bracket = pl.DataFrame({"which": ["data[1].parquet"]})
    plain = pl.DataFrame({"which": ["data1.parquet"]})
    fake_hub.put_parquet(fake_bucket, "g/data[1].parquet", bracket)
    fake_hub.put_parquet(fake_bucket, "g/data1.parquet", plain)

    got = plhf.scan_bucket(_uri(fake_bucket, "g/data[1].parquet")).collect()

    assert_frame_equal(got, bracket)


@pytest.mark.xfail(
    strict=True,
    raises=httpx.HTTPStatusError,
    reason="bug h: a path that ends in .parquet is always treated as one file, "
    "so a directory named 'out.parquet/' is resolved as a file and fails",
)
def test_directory_with_parquet_suffix_scans_as_directory(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    parts = [pl.DataFrame({"x": [1, 2]}), pl.DataFrame({"x": [3]})]
    fake_hub.put_parquet(fake_bucket, "h/out.parquet/part-0.parquet", parts[0])
    fake_hub.put_parquet(fake_bucket, "h/out.parquet/part-1.parquet", parts[1])

    got = plhf.scan_bucket(_uri(fake_bucket, "h/out.parquet")).collect()

    assert_frame_equal(got.sort("x"), pl.concat(parts))


@pytest.mark.xfail(
    strict=True,
    raises=httpx.HTTPStatusError,
    reason="bug i: every glob match is passed to the parquet scan, including "
    "sub-directories",
)
def test_star_glob_does_not_scan_sub_directories(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "i/a.parquet", pl.DataFrame({"x": [1]}))
    fake_hub.put_parquet(fake_bucket, "i/b.parquet", pl.DataFrame({"x": [2]}))
    fake_hub.put_parquet(fake_bucket, "i/sub/c.parquet", pl.DataFrame({"x": [3]}))

    got = plhf.scan_bucket(_uri(fake_bucket, "i/*")).collect()

    assert sorted(got["x"].to_list()) == [1, 2]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug j: a directory expands to '**/*.parquet' only, which skips "
    "'.pq' and upper-case '.PARQUET' files",
)
def test_directory_scan_includes_pq_and_upper_case_parquet(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "j/a.parquet", pl.DataFrame({"x": [1]}))
    fake_hub.put_parquet(fake_bucket, "j/b.pq", pl.DataFrame({"x": [2]}))
    fake_hub.put_parquet(fake_bucket, "j/c.PARQUET", pl.DataFrame({"x": [3]}))

    got = plhf.scan_bucket(_uri(fake_bucket, "j")).collect()

    assert sorted(got["x"].to_list()) == [1, 2, 3]


@pytest.mark.xfail(
    strict=True,
    raises=httpx.HTTPStatusError,
    reason="bug k: a missing single file surfaces the raw 404 "
    "httpx.HTTPStatusError of the resolve request",
)
def test_missing_single_file_raises_file_not_found(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The error may be raised by scan_bucket() or later, by collect().
    with pytest.raises(FileNotFoundError):
        plhf.scan_bucket(_uri(fake_bucket, "nope.parquet")).collect()


@pytest.mark.xfail(
    strict=True,
    raises=httpx.HTTPStatusError,
    reason="bug l: the resolve request is not retried, so one 429 or 503 fails "
    "the whole scan",
)
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


@pytest.mark.xfail(
    condition=HUB_MAJOR < 2,
    strict=True,
    raises=FileNotFoundError,
    reason="bug p: with huggingface_hub < 2.0 the recursive listing of 'data' "
    "also returns the sibling 'data.parquet' (string-prefix match), and the "
    "directory scan then raises FileNotFoundError",
)
def test_directory_scan_with_prefix_sibling(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    inside = pl.DataFrame({"x": [1]})
    fake_hub.put_parquet(fake_bucket, "data/a.parquet", inside)
    fake_hub.put_parquet(fake_bucket, "data.parquet", pl.DataFrame({"x": [2]}))
    fake_hub.put_parquet(fake_bucket, "data2/b.parquet", pl.DataFrame({"x": [3]}))

    got = plhf.scan_bucket(_uri(fake_bucket, "data")).collect()

    assert_frame_equal(got, inside)
