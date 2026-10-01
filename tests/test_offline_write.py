"""Offline write tests: ``sink_bucket`` against the fake Hub (no network)."""

from __future__ import annotations

import io

import polars as pl
import pytest
from fakehub import FakeHub, ScriptedUploadError
from polars.testing import assert_frame_equal
from sinks import fail_upload, sink_default, sink_streamed

import polars_hf as plhf

_READERS = {
    "parquet": pl.read_parquet,
    "csv": pl.read_csv,
    "ipc": pl.read_ipc,
    "ndjson": pl.read_ndjson,
}


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _relative(paths: list[str], prefix: str) -> list[str]:
    return sorted(p[len(prefix) :] for p in paths)


def test_single_file_round_trip(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = pl.DataFrame({"n": range(100), "g": ["a", "b"] * 50})
    uri = _uri(fake_bucket, "out/eager.parquet")

    plhf.sink_bucket(df, uri)

    assert fake_hub.files(fake_bucket) == ["out/eager.parquet"]
    assert_frame_equal(plhf.scan_bucket(uri).collect(), df)


def test_single_file_write_commits_through_batch_bucket_files(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # sink_bucket writes through HfFileSystem.open(..., "wb"); that file commits
    # with HfApi.batch_bucket_files on close, which the fake replaces.
    df = pl.DataFrame({"a": [1, 2, 3]})

    plhf.sink_bucket(df.lazy(), _uri(fake_bucket, "out/lazy.parquet"))

    assert len(fake_hub.batch_calls) == 1
    call = fake_hub.batch_calls[0]
    assert call.bucket_id == fake_bucket
    assert list(call.added) == ["out/lazy.parquet"]
    assert call.added["out/lazy.parquet"] == len(
        fake_hub.read(fake_bucket, "out/lazy.parquet")
    )


@pytest.mark.parametrize("ext", ["parquet", "csv", "ipc", "ndjson"])
def test_single_file_formats(fake_hub: FakeHub, fake_bucket: str, ext: str) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})

    plhf.sink_bucket(df.lazy(), _uri(fake_bucket, f"out/lazy.{ext}"))

    back = _READERS[ext](io.BytesIO(fake_hub.read(fake_bucket, f"out/lazy.{ext}")))
    assert_frame_equal(back, df)


def test_format_override(fake_hub: FakeHub, fake_bucket: str) -> None:
    # Extension says .data but we force parquet.
    df = pl.DataFrame({"a": [1, 2]})

    plhf.sink_bucket(df, _uri(fake_bucket, "out/override.data"), format="parquet")

    back = pl.read_parquet(io.BytesIO(fake_hub.read(fake_bucket, "out/override.data")))
    assert_frame_equal(back, df)


@pytest.mark.parametrize("sink", [sink_default, sink_streamed])
def test_partition_by_key_round_trip(fake_hub: FakeHub, fake_bucket: str, sink) -> None:
    df = pl.DataFrame({"g": ["a", "a", "b", "c", "c", "c"], "n": range(6)})
    base = _uri(fake_bucket, "parts")

    sink(df, base, partition_by="g")

    assert _relative(fake_hub.files(fake_bucket, "parts/"), "parts/") == [
        "g=a/00000000.parquet",
        "g=b/00000000.parquet",
        "g=c/00000000.parquet",
    ]
    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert_frame_equal(back.sort("n"), df)


@pytest.mark.parametrize("sink", [sink_default, sink_streamed])
def test_partition_by_size_round_trip(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    df = pl.DataFrame({"n": range(1000)})
    base = _uri(fake_bucket, "shards")

    sink(df, base, max_rows_per_file=250)

    assert len(fake_hub.files(fake_bucket, "shards/")) == 4  # 1000 rows / 250
    back = plhf.scan_bucket(base).collect()
    assert_frame_equal(back.sort("n"), df)


def test_partition_key_and_size(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = pl.DataFrame({"g": ["a"] * 500 + ["b"] * 500, "n": range(1000)})
    base = _uri(fake_bucket, "ks")

    sink_default(df, base, partition_by="g", max_rows_per_file=300)

    files = fake_hub.files(fake_bucket, "ks/")
    # 2 keys x ceil(500/300)=2 files each = 4
    assert len(files) == 4
    assert all(("g=a/" in f) or ("g=b/" in f) for f in files)
    assert plhf.scan_bucket(base).collect().height == 1000


def test_large_write_is_chunked(fake_hub: FakeHub, fake_bucket: str) -> None:
    # The Hub client sends at most 1,000 operations per call, so 1,001 files
    # need two calls.
    df = pl.DataFrame({"n": range(1001)})

    sink_default(df, _uri(fake_bucket, "many"), max_rows_per_file=1)

    assert [len(call.added) for call in fake_hub.batch_calls] == [1000, 1]
    assert len(fake_hub.files(fake_bucket, "many/")) == 1001


def test_default_upload_error_propagates_and_commits_nothing(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fail_upload(fake_hub, 1)
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})

    with pytest.raises(ScriptedUploadError):
        sink_default(df, _uri(fake_bucket, "parts"), partition_by="g")

    assert fake_hub.files(fake_bucket) == []


def test_explicit_token_is_used_for_writes(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.accept_token("hf_explicit_write_token")
    df = pl.DataFrame({"a": [1, 2]})

    plhf.sink_bucket(
        df, _uri(fake_bucket, "out/token.parquet"), token="hf_explicit_write_token"
    )

    assert fake_hub.files(fake_bucket) == ["out/token.parquet"]
    sent = {request.authorization for request in fake_hub.requests}
    assert sent == {"Bearer hf_explicit_write_token"}
