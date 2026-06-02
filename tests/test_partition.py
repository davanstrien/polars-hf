"""Partitioned-write tests for sink_bucket (both atomic modes)."""

from __future__ import annotations

import os

import polars as pl
import pytest
from huggingface_hub import HfApi, get_token

import polars_hf as plhf

_HAS_TOKEN = bool(get_token() or os.environ.get("HF_TOKEN"))
network = pytest.mark.skipif(not _HAS_TOKEN, reason="no Hugging Face token available")

BUCKET = "davanstrien/polars-hf-wheels"


def _bucket_parquet(prefix_in_bucket: str) -> list[str]:
    """Authoritative parquet file list under a bucket prefix (via the API)."""
    api = HfApi()
    return sorted(
        it.path
        for it in api.list_bucket_tree(BUCKET, prefix=prefix_in_bucket, recursive=True)
        if getattr(it, "path", "").endswith(".parquet")
    )


@pytest.mark.network
@network
@pytest.mark.parametrize("atomic", [True, False])
def test_partition_by_key(atomic: bool) -> None:
    tag = "ka" if atomic else "kb"
    df = pl.DataFrame({"g": ["a", "a", "b", "c", "c", "c"], "n": range(6)})
    base = f"hf://buckets/{BUCKET}/ptest/{tag}"
    plhf.sink_bucket(df, base, partition_by="g", atomic=atomic)

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 6

    files = _bucket_parquet(f"ptest/{tag}")
    assert len(files) == 3  # 3 distinct keys
    assert any("g=a/" in f for f in files)
    assert any("g=c/" in f for f in files)


@pytest.mark.network
@network
@pytest.mark.parametrize("atomic", [True, False])
def test_partition_by_size(atomic: bool) -> None:
    tag = "sa" if atomic else "sb"
    df = pl.DataFrame({"n": range(1000)})
    base = f"hf://buckets/{BUCKET}/ptest/{tag}"
    plhf.sink_bucket(df, base, max_rows_per_file=250, atomic=atomic)

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 1000
    assert back["n"].n_unique() == 1000

    files = _bucket_parquet(f"ptest/{tag}")
    assert len(files) == 4  # 1000 rows / 250 per file


@pytest.mark.network
@network
def test_partition_key_and_size() -> None:
    df = pl.DataFrame({"g": ["a"] * 500 + ["b"] * 500, "n": range(1000)})
    base = f"hf://buckets/{BUCKET}/ptest/ks"
    plhf.sink_bucket(df, base, partition_by="g", max_rows_per_file=300, atomic=True)

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 1000

    files = _bucket_parquet("ptest/ks")
    # 2 keys x ceil(500/300)=2 files each = 4
    assert len(files) == 4
    assert all(("g=a/" in f) or ("g=b/" in f) for f in files)
