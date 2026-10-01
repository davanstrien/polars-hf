"""Tests for sink_bucket: pure format inference + staging round-trips."""

from __future__ import annotations

import polars as pl
import pytest
from conftest import STAGING_ENDPOINT, STAGING_TOKEN
from huggingface_hub import HfFileSystem

import polars_hf as plhf
from polars_hf.write import _infer_format

# ---- pure logic (no network) ----------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a/b.parquet", "parquet"),
        ("x.pq", "parquet"),
        ("data.csv", "csv"),
        ("t.ipc", "ipc"),
        ("t.arrow", "ipc"),
        ("t.feather", "ipc"),
        ("t.ndjson", "ndjson"),
        ("t.jsonl", "ndjson"),
    ],
)
def test_infer_format(path: str, expected: str) -> None:
    assert _infer_format(path) == expected


def test_infer_format_unknown() -> None:
    with pytest.raises(ValueError, match="could not infer format"):
        _infer_format("data.txt")


def test_sink_requires_file_path() -> None:
    with pytest.raises(ValueError, match="file path within the bucket is required"):
        plhf.sink_bucket(pl.DataFrame({"a": [1]}), "hf://buckets/ns/name")


def test_sink_revision_rejected() -> None:
    with pytest.raises(ValueError, match="do not support @revision"):
        plhf.sink_bucket(
            pl.DataFrame({"a": [1]}), "hf://buckets/ns/name@main/x.parquet"
        )


# ---- staging round-trips (pytest -m staging) --------------------------------

_READERS = {
    "parquet": pl.read_parquet,
    "csv": pl.read_csv,
    "ipc": pl.read_ipc,
    "ndjson": pl.read_ndjson,
}


def _staging_fs() -> HfFileSystem:
    return HfFileSystem(endpoint=STAGING_ENDPOINT, token=STAGING_TOKEN)


@pytest.mark.staging
@pytest.mark.parametrize("ext", ["parquet", "csv", "ipc", "ndjson"])
def test_roundtrip_lazyframe(staging_bucket: str, ext: str) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})
    uri = f"hf://buckets/{staging_bucket}/sink-tests/lazy.{ext}"
    plhf.sink_bucket(df.lazy(), uri)

    with _staging_fs().open(
        f"buckets/{staging_bucket}/sink-tests/lazy.{ext}", "rb"
    ) as f:
        back = _READERS[ext](f)
    assert back.shape == (3, 3)
    if ext in ("parquet", "ipc"):
        assert back.equals(df)


@pytest.mark.staging
def test_roundtrip_dataframe_via_scan_bucket(staging_bucket: str) -> None:
    # Eager DataFrame input, and read back through our own scan_bucket.
    df = pl.DataFrame({"n": range(100), "g": ["a", "b"] * 50})
    uri = f"hf://buckets/{staging_bucket}/sink-tests/eager.parquet"
    plhf.sink_bucket(df, uri)
    back = plhf.scan_bucket(uri).collect()
    assert back.shape == (100, 2)
    assert back.equals(df)


@pytest.mark.staging
def test_roundtrip_explicit_token(staging_bucket: str) -> None:
    df = pl.DataFrame({"a": [1, 2]})
    uri = f"hf://buckets/{staging_bucket}/sink-tests/token.parquet"
    plhf.sink_bucket(df, uri, token=STAGING_TOKEN)
    assert plhf.scan_bucket(uri, token=STAGING_TOKEN).collect().equals(df)


@pytest.mark.staging
def test_format_override(staging_bucket: str) -> None:
    # Extension says .data but we force parquet.
    df = pl.DataFrame({"a": [1, 2]})
    uri = f"hf://buckets/{staging_bucket}/sink-tests/override.data"
    plhf.sink_bucket(df, uri, format="parquet")
    with _staging_fs().open(
        f"buckets/{staging_bucket}/sink-tests/override.data", "rb"
    ) as f:
        back = pl.read_parquet(f)
    assert back.shape == (2, 1)
