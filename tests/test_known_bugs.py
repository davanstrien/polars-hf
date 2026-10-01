"""Known bugs, written as tests of the DESIRED behaviour.

Every test here is ``xfail(strict=True)``: it fails today for the stated
reason, and the suite goes red as soon as it starts to pass. The pull request
that fixes a bug removes the marker (or moves the test to the matching test
module). ``raises=`` pins the failure to the documented symptom, so an
unrelated breakage does not hide behind the marker.

All tests run offline against the fake Hub (see ``fakehub.py``).
"""

from __future__ import annotations

import gc
import os
import tempfile
import threading

import httpx
import polars as pl
import pytest
from fakehub import HUB, SIGNATURE, FakeHub, ScriptedUploadError
from polars.testing import assert_frame_equal

import polars_hf as plhf

# pytest.raises reports a missing exception with this type.
DidNotRaise = pytest.fail.Exception


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _relative(paths: list[str], prefix: str) -> list[str]:
    return sorted(p[len(prefix) :] for p in paths)


def _failing_frame() -> pl.LazyFrame:
    """A LazyFrame that raises ``RuntimeError`` when the engine runs it."""

    def explode(batch: pl.DataFrame) -> pl.DataFrame:
        raise RuntimeError("scripted sink failure")

    return pl.LazyFrame({"a": range(10)}).map_batches(explode)


# ---- write path ------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug a: fsspec commits the temp file on close, even after an error, "
    "so a failed single-file sink replaces the object with an empty file",
)
def test_failed_single_file_sink_keeps_existing_object(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put_parquet(fake_bucket, "keep.parquet", pl.DataFrame({"a": [1, 2, 3]}))
    before = fake_hub.read(fake_bucket, "keep.parquet")

    with pytest.raises(RuntimeError, match="scripted sink failure"):
        plhf.sink_bucket(_failing_frame(), _uri(fake_bucket, "keep.parquet"))

    assert fake_hub.read(fake_bucket, "keep.parquet") == before
    assert fake_hub.batch_calls == []


@pytest.mark.xfail(
    strict=True,
    raises=DidNotRaise,
    reason="bug b: lazy=True is forwarded to polars, the returned plan is "
    "dropped and a 0-byte object is committed",
)
def test_lazy_sink_is_rejected(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="lazy"):
        plhf.sink_bucket(df, _uri(fake_bucket, "lazy.parquet"), lazy=True)

    assert fake_hub.files(fake_bucket) == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug c: atomic=False builds partition directories from the raw key "
    "(unencoded, None -> 'g=None') instead of the native Polars hive layout",
)
@pytest.mark.parametrize("key", [None, "a/b", "a b", "x=y"])
def test_streamed_partition_names_match_native_layout(
    fake_hub: FakeHub, fake_bucket: str, key: str | None
) -> None:
    df = pl.DataFrame({"g": [key], "n": [1]}, schema={"g": pl.String, "n": pl.Int64})

    plhf.sink_bucket(df, _uri(fake_bucket, "native"), partition_by="g", atomic=True)
    plhf.sink_bucket(df, _uri(fake_bucket, "stream"), partition_by="g", atomic=False)
    gc.collect()

    native = _relative(fake_hub.files(fake_bucket, "native/"), "native/")
    stream = _relative(fake_hub.files(fake_bucket, "stream/"), "stream/")
    assert len(native) == 1
    assert stream == native


@pytest.mark.xfail(
    strict=True,
    raises=DidNotRaise,
    reason="bug d: with atomic=False the upload runs when polars drops the file "
    "object, so an upload error is swallowed in __del__",
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_streamed_upload_error_propagates(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = pl.DataFrame({"g": ["a", "b", "c"], "n": [1, 2, 3]})
    fake_hub.fail_batch_on_call = 2

    try:
        with pytest.raises(ScriptedUploadError):
            plhf.sink_bucket(
                df, _uri(fake_bucket, "parts"), partition_by="g", atomic=False
            )
    finally:
        # Run the pending __del__ uploads inside this test.
        gc.collect()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug d: with atomic=False a write that fails part-way leaves the "
    "files uploaded before the failure committed in the bucket",
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_failed_streamed_write_leaves_no_partial_files(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    df = pl.DataFrame({"g": ["a", "b", "c"], "n": [1, 2, 3]})
    fake_hub.fail_batch_on_call = 2

    try:
        plhf.sink_bucket(df, _uri(fake_bucket, "parts"), partition_by="g", atomic=False)
    except ScriptedUploadError:
        pass
    gc.collect()

    assert any(call.failed for call in fake_hub.batch_calls)
    assert fake_hub.files(fake_bucket, "parts/") == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug e: an empty frame with partition_by writes no file at all, so "
    "the write succeeds but a later scan raises FileNotFoundError",
)
@pytest.mark.parametrize("atomic", [True, False])
def test_empty_partitioned_write_produces_one_empty_file(
    fake_hub: FakeHub, fake_bucket: str, atomic: bool
) -> None:
    # Chosen behaviour: one parquet file with the schema and zero rows, so the
    # prefix scans back as an empty frame.
    schema = {"g": pl.String, "n": pl.Int64}
    df = pl.DataFrame({"g": [], "n": []}, schema=schema)
    base = _uri(fake_bucket, "empty")

    plhf.sink_bucket(df, base, partition_by="g", atomic=atomic)
    gc.collect()

    files = fake_hub.files(fake_bucket, "empty/")
    assert len(files) == 1
    assert_frame_equal(plhf.scan_bucket(base).collect(), df)


def _directory_size(directory: str) -> int:
    total = 0
    for root, _, names in os.walk(directory):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass  # removed between the listing and the stat
    return total


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug n: the default partitioned write stages the complete output in "
    "a local temp dir before the single upload call",
)
def test_partitioned_write_bounds_local_staging(
    fake_hub: FakeHub, fake_bucket: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # All local staging (tempfile.mkdtemp and HfFileSystem's NamedTemporaryFile)
    # goes below this directory, so its size is the staging disk in use.
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))

    peak = 0
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        peak = max(peak, _directory_size(str(staging)))

    def sample_until_stopped() -> None:
        while not stop.is_set():
            sample()
            stop.wait(0.002)

    # Sample at every upload call (deterministic) and in the background.
    fake_hub.before_batch = sample
    sampler = threading.Thread(target=sample_until_stopped, daemon=True)

    rows = 200_000
    ids = pl.int_range(0, rows, eager=True)
    text = (ids * 2654435761 % 1000003).cast(pl.String) + "-" + ids.cast(pl.String)
    df = pl.DataFrame({"id": ids, "text": text})

    sampler.start()
    try:
        plhf.sink_bucket(
            df,
            _uri(fake_bucket, "many"),
            max_rows_per_file=5_000,
            compression="uncompressed",
        )
    finally:
        stop.set()
        sampler.join()

    written = fake_hub.files(fake_bucket, "many/")
    total = sum(len(fake_hub.read(fake_bucket, path)) for path in written)
    assert len(written) == 40
    # 40 files of about 100 kB each: staging must stay below a quarter of the
    # output, i.e. it must not grow with the number of files.
    assert peak <= total // 4


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="bug o: sink_bucket has no mode= parameter; rewriting a prefix with "
    "fewer files leaves the stale files of the previous write",
)
def test_overwrite_mode_removes_stale_files(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    base = _uri(fake_bucket, "shards")
    plhf.sink_bucket(pl.DataFrame({"n": range(400)}), base, max_rows_per_file=100)
    assert len(fake_hub.files(fake_bucket, "shards/")) == 4

    smaller = pl.DataFrame({"n": range(200)})
    plhf.sink_bucket(smaller, base, max_rows_per_file=100, mode="overwrite")

    assert len(fake_hub.files(fake_bucket, "shards/")) == 2
    assert_frame_equal(plhf.scan_bucket(base).collect().sort("n"), smaller)


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
    "read as the character class and matches 'data1.parquet'",
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
    "non-parquet files and sub-directories",
)
def test_star_glob_selects_only_parquet_files(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    top = pl.DataFrame({"x": [1]})
    fake_hub.put_parquet(fake_bucket, "i/a.parquet", top)
    fake_hub.put(fake_bucket, "i/readme.txt", b"not parquet")
    fake_hub.put_parquet(fake_bucket, "i/sub/b.parquet", pl.DataFrame({"x": [2]}))

    got = plhf.scan_bucket(_uri(fake_bucket, "i/*")).collect()

    assert_frame_equal(got, top)


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
    with pytest.raises(FileNotFoundError):
        plhf.scan_bucket(_uri(fake_bucket, "nope.parquet"))


@pytest.mark.xfail(
    strict=True,
    raises=httpx.HTTPStatusError,
    reason="bug l: the resolve HEAD is not retried, so one 429 or 503 fails the "
    "whole scan",
)
@pytest.mark.parametrize("status", [429, 503])
def test_transient_resolve_error_is_retried(
    fake_hub: FakeHub, fake_bucket: str, status: int
) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    fake_hub.put_parquet(fake_bucket, "one.parquet", df)
    fake_hub.add_fault(
        HUB, "HEAD", r"/resolve/one\.parquet$", status, headers={"Retry-After": "0"}
    )

    got = plhf.scan_bucket(_uri(fake_bucket, "one.parquet")).collect()

    assert_frame_equal(got, df)
    resolves = fake_hub.matching(origin=HUB, method="HEAD", path_contains="/resolve/")
    assert [r.status for r in resolves] == [status, 302]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug m: the query plan prints the presigned URL, signature included",
)
def test_presigned_url_not_in_explain(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", pl.DataFrame({"a": [1]}))

    lf = plhf.scan_bucket(_uri(fake_bucket, "one.parquet"))

    assert SIGNATURE not in lf.explain()
    assert SIGNATURE not in lf.explain(optimized=False)
