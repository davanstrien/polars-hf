"""Partitioned-write tests for sink_bucket (both atomic modes), on staging.

Selected with ``pytest -m staging``. Each test writes to its own new bucket
(the ``staging_bucket`` fixture), so tests cannot race on shared paths.
"""

from __future__ import annotations

import polars as pl
import pytest
from huggingface_hub import HfApi

import polars_hf as plhf

pytestmark = pytest.mark.staging


def _bucket_parquet(api: HfApi, bucket_id: str, prefix_in_bucket: str) -> list[str]:
    """Authoritative parquet file list under a bucket prefix (via the API)."""
    return sorted(
        it.path
        for it in api.list_bucket_tree(
            bucket_id, prefix=prefix_in_bucket, recursive=True
        )
        if getattr(it, "path", "").endswith(".parquet")
    )


@pytest.mark.parametrize("atomic", [True, False])
def test_partition_by_key(
    staging_api: HfApi, staging_bucket: str, atomic: bool
) -> None:
    df = pl.DataFrame({"g": ["a", "a", "b", "c", "c", "c"], "n": range(6)})
    base = f"hf://buckets/{staging_bucket}/ptest/k"
    plhf.sink_bucket(df, base, partition_by="g", atomic=atomic)

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 6

    files = _bucket_parquet(staging_api, staging_bucket, "ptest/k")
    assert len(files) == 3  # 3 distinct keys
    assert any("g=a/" in f for f in files)
    assert any("g=c/" in f for f in files)


@pytest.mark.parametrize("atomic", [True, False])
def test_partition_by_size(
    staging_api: HfApi, staging_bucket: str, atomic: bool
) -> None:
    df = pl.DataFrame({"n": range(1000)})
    base = f"hf://buckets/{staging_bucket}/ptest/s"
    plhf.sink_bucket(df, base, max_rows_per_file=250, atomic=atomic)

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 1000
    assert back["n"].n_unique() == 1000

    files = _bucket_parquet(staging_api, staging_bucket, "ptest/s")
    assert len(files) == 4  # 1000 rows / 250 per file


def test_partition_key_and_size(staging_api: HfApi, staging_bucket: str) -> None:
    df = pl.DataFrame({"g": ["a"] * 500 + ["b"] * 500, "n": range(1000)})
    base = f"hf://buckets/{staging_bucket}/ptest/ks"
    plhf.sink_bucket(df, base, partition_by="g", max_rows_per_file=300, atomic=True)

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 1000

    files = _bucket_parquet(staging_api, staging_bucket, "ptest/ks")
    # 2 keys x ceil(500/300)=2 files each = 4
    assert len(files) == 4
    assert all(("g=a/" in f) or ("g=b/" in f) for f in files)
