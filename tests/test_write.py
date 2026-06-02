"""Tests for sink_bucket: pure format inference + live round-trips."""

from __future__ import annotations

import os

import polars as pl
import pytest
from huggingface_hub import HfFileSystem, get_token

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
        plhf.sink_bucket(pl.DataFrame({"a": [1]}), "hf://buckets/ns/name@main/x.parquet")


# ---- live round-trips (network, token-gated) -------------------------------

_HAS_TOKEN = bool(get_token() or os.environ.get("HF_TOKEN"))
network = pytest.mark.skipif(not _HAS_TOKEN, reason="no Hugging Face token available")

BUCKET = "davanstrien/polars-hf-wheels"
PREFIX = f"hf://buckets/{BUCKET}/sink-tests"

_READERS = {
    "parquet": pl.read_parquet,
    "csv": pl.read_csv,
    "ipc": pl.read_ipc,
    "ndjson": pl.read_ndjson,
}


@pytest.mark.network
@network
@pytest.mark.parametrize("ext", ["parquet", "csv", "ipc", "ndjson"])
def test_roundtrip_lazyframe(ext: str) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})
    uri = f"{PREFIX}/lazy.{ext}"
    plhf.sink_bucket(df.lazy(), uri)

    fs = HfFileSystem()
    with fs.open(f"buckets/{BUCKET}/sink-tests/lazy.{ext}", "rb") as f:
        back = _READERS[ext](f)
    assert back.shape == (3, 3)
    if ext in ("parquet", "ipc"):
        assert back.equals(df)


@pytest.mark.network
@network
def test_roundtrip_dataframe_via_scan_bucket() -> None:
    # Eager DataFrame input, and read back through our own scan_bucket.
    df = pl.DataFrame({"n": range(100), "g": ["a", "b"] * 50})
    uri = f"{PREFIX}/eager.parquet"
    plhf.sink_bucket(df, uri)
    back = plhf.scan_bucket(uri).collect()
    assert back.shape == (100, 2)
    assert back.equals(df)


@pytest.mark.network
@network
def test_format_override() -> None:
    # Extension says .data but we force parquet.
    df = pl.DataFrame({"a": [1, 2]})
    uri = f"{PREFIX}/override.data"
    plhf.sink_bucket(df, uri, format="parquet")
    fs = HfFileSystem()
    with fs.open(f"buckets/{BUCKET}/sink-tests/override.data", "rb") as f:
        back = pl.read_parquet(f)
    assert back.shape == (2, 1)
