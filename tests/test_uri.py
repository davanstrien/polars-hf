"""Pure-logic tests for bucket URI parsing (no network)."""

from __future__ import annotations

import pytest

from polars_hf._uri import BucketPath, parse_bucket_uri


def test_single_file() -> None:
    bp = parse_bucket_uri("hf://buckets/davanstrien/polars-hf-wheels/a/b/file.parquet")
    assert bp == BucketPath(
        bucket_id="davanstrien/polars-hf-wheels", path="a/b/file.parquet"
    )
    assert not bp.is_glob
    assert bp.fs_path == "buckets/davanstrien/polars-hf-wheels/a/b/file.parquet"


def test_glob_path() -> None:
    bp = parse_bucket_uri("hf://buckets/ns/name/data/*.parquet")
    assert bp.bucket_id == "ns/name"
    assert bp.path == "data/*.parquet"
    assert bp.is_glob


def test_whole_bucket_empty_path() -> None:
    bp = parse_bucket_uri("hf://buckets/ns/name")
    assert bp.bucket_id == "ns/name"
    assert bp.path == ""
    assert not bp.is_glob
    assert bp.fs_path == "buckets/ns/name"


def test_revision_rejected() -> None:
    with pytest.raises(ValueError, match="do not support @revision"):
        parse_bucket_uri("hf://buckets/ns/name@main/x.parquet")


def test_datasets_points_to_native() -> None:
    with pytest.raises(ValueError, match="read natively by polars"):
        parse_bucket_uri("hf://datasets/nyu-mll/glue/cola/train.parquet")


def test_spaces_points_to_native() -> None:
    with pytest.raises(ValueError, match="read natively by polars"):
        parse_bucket_uri("hf://spaces/ns/name/x.parquet")


def test_not_hf_uri() -> None:
    with pytest.raises(ValueError, match="must start with 'hf://'"):
        parse_bucket_uri("s3://bucket/key.parquet")


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError, match="invalid Hugging Face bucket URI"):
        parse_bucket_uri("hf://models/ns/name/x.parquet")


def test_missing_name_rejected() -> None:
    with pytest.raises(ValueError, match="invalid Hugging Face bucket URI"):
        parse_bucket_uri("hf://buckets/ns")
