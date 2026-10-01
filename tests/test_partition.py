"""Partitioned-write tests for sink_bucket (default and streamed writes), on staging.

Selected with ``pytest -m staging``. Each test writes to its own new bucket
(the ``staging_bucket`` fixture), so tests cannot race on shared paths.
"""

from __future__ import annotations

import polars as pl
import pytest
from huggingface_hub import HfApi
from sinks import sink_default, sink_streamed

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


@pytest.mark.parametrize("sink", [sink_default, sink_streamed])
def test_partition_by_key(staging_api: HfApi, staging_bucket: str, sink) -> None:
    df = pl.DataFrame({"g": ["a", "a", "b", "c", "c", "c"], "n": range(6)})
    base = f"hf://buckets/{staging_bucket}/ptest/k"
    sink(df, base, partition_by="g")

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 6

    files = _bucket_parquet(staging_api, staging_bucket, "ptest/k")
    assert len(files) == 3  # 3 distinct keys
    assert any("g=a/" in f for f in files)
    assert any("g=c/" in f for f in files)


@pytest.mark.parametrize("sink", [sink_default, sink_streamed])
def test_partition_by_size(staging_api: HfApi, staging_bucket: str, sink) -> None:
    df = pl.DataFrame({"n": range(1000)})
    base = f"hf://buckets/{staging_bucket}/ptest/s"
    sink(df, base, max_rows_per_file=250)

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 1000
    assert back["n"].n_unique() == 1000

    files = _bucket_parquet(staging_api, staging_bucket, "ptest/s")
    assert len(files) == 4  # 1000 rows / 250 per file


def test_partition_key_and_size(staging_api: HfApi, staging_bucket: str) -> None:
    df = pl.DataFrame({"g": ["a"] * 500 + ["b"] * 500, "n": range(1000)})
    base = f"hf://buckets/{staging_bucket}/ptest/ks"
    sink_default(df, base, partition_by="g", max_rows_per_file=300)

    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert back.height == 1000

    files = _bucket_parquet(staging_api, staging_bucket, "ptest/ks")
    # 2 keys x ceil(500/300)=2 files each = 4
    assert len(files) == 4
    assert all(("g=a/" in f) or ("g=b/" in f) for f in files)
