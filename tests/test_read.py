"""End-to-end read tests against the Hub CI staging instance.

Selected with ``pytest -m staging``. The ``staging_read_bucket`` fixture seeds a
bucket once per module with synthetic data (see ``conftest.staging_seed_files``):
``smoke/filtered.parquet`` is 500 x 4, ``smoke/bench/run_*.parquet`` are three
homogeneous 100k-row files, and ``smoke/*.parquet`` has mixed schemas.
"""

from __future__ import annotations

import polars as pl
import pytest
from conftest import staging_seed_files
from polars.testing import assert_frame_equal

import polars_hf as plhf

pytestmark = pytest.mark.staging

GLOB_FILES = 3
GLOB_ROWS = GLOB_FILES * 100_000


@pytest.fixture(scope="module")
def base(staging_read_bucket: str) -> str:
    return f"hf://buckets/{staging_read_bucket}/smoke"


@pytest.fixture(scope="module")
def single(base: str) -> str:
    return f"{base}/filtered.parquet"


@pytest.fixture(scope="module")
def glob(base: str) -> str:
    # bench/run_*.parquet are 3 homogeneous files (id, category, value, text),
    # 100k rows each. The top-level *.parquet files have mixed schemas, which —
    # like native pl.scan_parquet — raises without an explicit missing_columns
    # opt-in.
    return f"{base}/bench/run_*.parquet"


@pytest.fixture(scope="module")
def eager(single: str) -> pl.DataFrame:
    """The fixture file read fully, used as ground truth."""
    return plhf.scan_bucket(single).collect()


def test_single_file_shape(eager: pl.DataFrame) -> None:
    assert eager.shape == (500, 4)
    assert_frame_equal(eager, staging_seed_files()["smoke/filtered.parquet"])


def test_projection_pushdown(single: str, eager: pl.DataFrame) -> None:
    col = eager.columns[0]
    got = plhf.scan_bucket(single).select(col).collect()
    assert got.columns == [col]
    assert got.equals(eager.select(col))


def test_row_limit_pushdown(single: str) -> None:
    got = plhf.scan_bucket(single).head(10).collect()
    assert got.height == 10


def test_predicate_correctness(single: str, eager: pl.DataFrame) -> None:
    col = eager.columns[0]
    value = eager[col][0]
    got = plhf.scan_bucket(single).filter(pl.col(col) == value).collect()
    assert got.equals(eager.filter(pl.col(col) == value))


def test_glob_listing(glob: str) -> None:
    got = plhf.scan_bucket(glob).collect()
    assert got.shape == (GLOB_ROWS, 4)
    assert got.columns == ["id", "category", "value", "text"]


def test_glob_projection(glob: str) -> None:
    got = plhf.scan_bucket(glob).select("id").collect()
    assert got.columns == ["id"]
    assert got.height == GLOB_ROWS
    assert got["id"].n_unique() == GLOB_ROWS


def test_explicit_token(single: str) -> None:
    from conftest import STAGING_TOKEN

    got = plhf.scan_bucket(single, token=STAGING_TOKEN).head(1).collect()
    assert got.height == 1


def test_revision_rejected(staging_read_bucket: str) -> None:
    with pytest.raises(ValueError, match="do not support @revision"):
        plhf.scan_bucket(f"hf://buckets/{staging_read_bucket}@main/x.parquet").collect()


def test_scan_kwargs_forwarded_mixed_schemas(base: str) -> None:
    # The top-level *.parquet fixtures have mixed schemas: scanning them raises
    # by default (native behavior), but the scan_parquet opt-ins forwarded
    # through scan_bucket make the union scan work.
    mixed = f"{base}/*.parquet"
    with pytest.raises(pl.exceptions.PolarsError):
        plhf.scan_bucket(mixed).collect()
    got = plhf.scan_bucket(
        mixed, missing_columns="insert", extra_columns="ignore"
    ).collect()
    assert got.height > 0
