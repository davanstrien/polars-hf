"""End-to-end read tests against the Hugging Face Hub.

Gated on an available HF token (cached login or ``HF_TOKEN``); skipped otherwise
so CI without secrets stays green. Uses the known fixture from the fork's smoke
tests: ``davanstrien/polars-hf-wheels/smoke-test-full/filtered.parquet`` → (500, 4).
"""

from __future__ import annotations

import os

import polars as pl
import pytest
from huggingface_hub import get_token

import polars_hf as plhf

_HAS_TOKEN = bool(get_token() or os.environ.get("HF_TOKEN"))

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not _HAS_TOKEN, reason="no Hugging Face token available"),
]

BUCKET = "davanstrien/polars-hf-wheels"
BASE = f"hf://buckets/{BUCKET}/smoke-test-full"
SINGLE = f"{BASE}/filtered.parquet"
# bench/run_*.parquet are 3 homogeneous files (id, category, value, text),
# 100k rows each. The top-level *.parquet files have mixed schemas, which —
# like native pl.scan_parquet — raises without an explicit missing_columns opt-in.
GLOB = f"{BASE}/bench/run_*.parquet"
GLOB_FILES = 3
GLOB_ROWS = GLOB_FILES * 100_000


@pytest.fixture(scope="module")
def eager() -> pl.DataFrame:
    """The fixture file read fully, used as ground truth."""
    return plhf.scan_bucket(SINGLE).collect()


def test_single_file_shape(eager: pl.DataFrame) -> None:
    assert eager.shape == (500, 4)


def test_projection_pushdown(eager: pl.DataFrame) -> None:
    col = eager.columns[0]
    got = plhf.scan_bucket(SINGLE).select(col).collect()
    assert got.columns == [col]
    assert got.equals(eager.select(col))


def test_row_limit_pushdown() -> None:
    got = plhf.scan_bucket(SINGLE).head(10).collect()
    assert got.height == 10


def test_predicate_correctness(eager: pl.DataFrame) -> None:
    col = eager.columns[0]
    value = eager[col][0]
    got = plhf.scan_bucket(SINGLE).filter(pl.col(col) == value).collect()
    assert got.equals(eager.filter(pl.col(col) == value))


def test_glob_listing() -> None:
    got = plhf.scan_bucket(GLOB).collect()
    assert got.shape == (GLOB_ROWS, 4)


def test_glob_projection() -> None:
    got = plhf.scan_bucket(GLOB).select("id").collect()
    assert got.columns == ["id"]
    assert got.height == GLOB_ROWS


def test_revision_rejected() -> None:
    with pytest.raises(ValueError, match="do not support @revision"):
        plhf.scan_bucket(f"hf://buckets/{BUCKET}@main/x.parquet").collect()


def test_scan_kwargs_forwarded_mixed_schemas() -> None:
    # The top-level *.parquet fixtures have mixed schemas: scanning them raises
    # by default (native behavior), but the scan_parquet opt-ins forwarded
    # through scan_bucket make the union scan work.
    mixed = f"{BASE}/*.parquet"
    with pytest.raises(pl.exceptions.PolarsError):
        plhf.scan_bucket(mixed).collect()
    got = plhf.scan_bucket(
        mixed, missing_columns="insert", extra_columns="ignore"
    ).collect()
    assert got.height > 0


def test_native_parquet_scan_with_range_reads() -> None:
    # Guard the perf fix: scan_bucket must produce a NATIVE parquet scan over a
    # signed URL (range reads + pushdown), not a PYTHON SCAN that buffers files.
    plan = plhf.scan_bucket(SINGLE).explain()
    assert "Parquet SCAN" in plan
    assert "PYTHON SCAN" not in plan
