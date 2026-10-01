"""Pure-logic tests for bucket URI parsing (no network)."""

from __future__ import annotations

import pytest

from polars_hf._uri import BucketPath, parse_bucket_uri


def test_single_file() -> None:
    bp = parse_bucket_uri("hf://buckets/some-user/some-bucket/a/b/file.parquet")
    assert bp == BucketPath(bucket_id="some-user/some-bucket", path="a/b/file.parquet")
    assert not bp.is_glob
    assert bp.fs_path == "buckets/some-user/some-bucket/a/b/file.parquet"


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


@pytest.mark.parametrize(
    "uri",
    [
        "hf://buckets/ns/name@main",
        "hf://buckets/ns/name@main/x.parquet",
        "hf://buckets/ns@main/name/x.parquet",
    ],
)
def test_at_sign_in_bucket_id_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match="do not support @revision"):
        parse_bucket_uri(uri)


@pytest.mark.parametrize(
    "path",
    [
        "exports/user@example.com.parquet",
        "snapshots@2026-01-01/part-0.parquet",
        "@",
    ],
)
def test_at_sign_below_the_bucket_is_part_of_the_path(path: str) -> None:
    bp = parse_bucket_uri(f"hf://buckets/ns/name/{path}")
    assert bp == BucketPath(bucket_id="ns/name", path=path)


@pytest.mark.parametrize(
    ("uri", "path"),
    [
        ("hf://buckets/ns/name/", ""),
        ("hf://buckets/ns/name/data/", "data/"),
        ("hf://buckets/ns/name/a/b/", "a/b/"),
        ("hf://buckets/ns/name/dir with space /", "dir with space /"),
    ],
)
def test_trailing_slash_is_kept(uri: str, path: str) -> None:
    bp = parse_bucket_uri(uri)
    assert bp == BucketPath(bucket_id="ns/name", path=path)
    assert not bp.is_glob


@pytest.mark.parametrize(
    "uri",
    [
        "hf://buckets/ns/name//x.parquet",
        "hf://buckets/ns/name/a//b.parquet",
        "hf://buckets/ns/name/a//",
        "hf://buckets/ns/name//",
    ],
)
def test_empty_path_segment_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match="empty path segment") as error:
        parse_bucket_uri(uri)
    assert repr(uri) in str(error.value)


@pytest.mark.parametrize(
    "uri",
    [
        "hf://buckets/ns/name/..",
        "hf://buckets/ns/name/../other/x.parquet",
        "hf://buckets/ns/name/a/../b.parquet",
        "hf://buckets/ns/name/a/../",
    ],
)
def test_dot_dot_segment_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match=r"'\.\.' path segment") as error:
        parse_bucket_uri(uri)
    assert repr(uri) in str(error.value)


@pytest.mark.parametrize("name", ["..parquet", "a..b.parquet", "...", ".hidden"])
def test_dots_inside_a_name_are_allowed(name: str) -> None:
    bp = parse_bucket_uri(f"hf://buckets/ns/name/dir/{name}")
    assert bp.path == f"dir/{name}"


@pytest.mark.parametrize(
    "uri",
    [
        "hf://buckets/ns/name/x.parquet ",
        "hf://buckets/ns/name/x.parquet\n",
        "hf://buckets/ns/name/data/ ",
        "hf://buckets/ns/name ",
        "hf://buckets/ns/name/\t",
    ],
)
def test_trailing_whitespace_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match="ends with whitespace") as error:
        parse_bucket_uri(uri)
    assert repr(uri) in str(error.value)


def test_space_inside_the_path_is_allowed() -> None:
    bp = parse_bucket_uri("hf://buckets/ns/name/my dir /my file.parquet")
    assert bp.path == "my dir /my file.parquet"
