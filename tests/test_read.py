"""End-to-end read tests against the Hub CI staging instance.

Selected with ``pytest -m staging``. The ``staging_read_bucket`` fixture seeds a
bucket once per module with synthetic data (see ``conftest.staging_seed_files``):
``smoke/filtered.parquet`` is 500 x 4, ``smoke/bench/run_*.parquet`` are three
homogeneous 100k-row files, and ``smoke/*.parquet`` has mixed schemas.
"""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pytest
from conftest import (
    _create_staging_bucket,
    _delete_staging_bucket,
    _parquet_bytes,
    _staging_retry,
    staging_seed_files,
)
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
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


# ---- what a path names (same cases as the offline tests, on the real Hub) ---

# Each file holds one row with its own path, so a scan shows which files it read.
_EDGE_FILES = [
    "edge/data/a.parquet",
    "edge/data/sub/b.pq",
    "edge/data/C.PARQUET",
    "edge/data/notes.txt",
    # String-prefix siblings of the directory "edge/data".
    "edge/data.parquet",
    "edge/data2/z.parquet",
    "edge/g/data[1].parquet",
    "edge/g/data1.parquet",
    "edge/out.parquet/part-0.parquet",
    "edge/run[1]/sub/x.parquet",
    "edge/run1/y.parquet",
    "edge/table",
    "edge/user@example.com.parquet",
]
_EMPTY_FILE = "edge/empty/empty.parquet"


@pytest.fixture(scope="module")
def edge_bucket(staging_api: HfApi) -> Iterator[str]:
    """A staging bucket with the files of ``_EDGE_FILES``; read-only."""
    bucket_id = _create_staging_bucket(staging_api)
    try:
        add = [(b"", _EMPTY_FILE)]
        for path in _EDGE_FILES:
            add.append((_parquet_bytes(pl.DataFrame({"path": [path]})), path))
        _staging_retry(lambda: staging_api.batch_bucket_files(bucket_id, add=add))
        yield bucket_id
    finally:
        _delete_staging_bucket(staging_api, bucket_id)


def _paths_read(bucket_id: str, path: str) -> list[str]:
    lf = plhf.scan_bucket(f"hf://buckets/{bucket_id}/{path}")
    return sorted(lf.collect()["path"].to_list())


def test_directory_scan_ignores_prefix_siblings(edge_bucket: str) -> None:
    # Recursive, .parquet and .pq in any case, and neither "edge/data.parquet"
    # nor "edge/data2/..." (the Hub lists by string prefix).
    expected = ["edge/data/C.PARQUET", "edge/data/a.parquet", "edge/data/sub/b.pq"]

    assert _paths_read(edge_bucket, "edge/data") == expected
    assert _paths_read(edge_bucket, "edge/data/") == expected


def test_star_glob_matches_files_of_one_directory(edge_bucket: str) -> None:
    assert _paths_read(edge_bucket, "edge/data/*.parquet") == ["edge/data/a.parquet"]
    assert _paths_read(edge_bucket, "edge/data/**/*.pq") == ["edge/data/sub/b.pq"]


def test_literal_bracket_file_name(edge_bucket: str) -> None:
    assert _paths_read(edge_bucket, "edge/g/data[1].parquet") == [
        "edge/g/data[1].parquet"
    ]
    assert _paths_read(edge_bucket, "edge/g/data[0-9].parquet") == [
        "edge/g/data1.parquet"
    ]


def test_glob_over_directory_names(edge_bucket: str) -> None:
    # "run[0-9]" matches the directory "run1", not the one named "run[1]".
    assert _paths_read(edge_bucket, "edge/run[0-9]/*.parquet") == [
        "edge/run1/y.parquet"
    ]
    assert _paths_read(edge_bucket, "edge/run[[]1]/**/*.parquet") == [
        "edge/run[1]/sub/x.parquet"
    ]


def test_directory_listing_has_no_prefix_siblings(
    edge_bucket: str, staging_api: HfApi
) -> None:
    # What scan_bucket relies on: a listing with a trailing-slash prefix
    # returns the files of that directory only.
    listed = staging_api.list_bucket_tree(
        edge_bucket, prefix="edge/data/", recursive=True
    )
    assert sorted(entry.path for entry in listed) == [
        "edge/data/C.PARQUET",
        "edge/data/a.parquet",
        "edge/data/notes.txt",
        "edge/data/sub/b.pq",
    ]


def test_directory_with_parquet_suffix(edge_bucket: str) -> None:
    assert _paths_read(edge_bucket, "edge/out.parquet") == [
        "edge/out.parquet/part-0.parquet"
    ]


def test_single_file_without_extension(edge_bucket: str) -> None:
    assert _paths_read(edge_bucket, "edge/table") == ["edge/table"]


def test_at_sign_in_file_name(edge_bucket: str) -> None:
    assert _paths_read(edge_bucket, "edge/user@example.com.parquet") == [
        "edge/user@example.com.parquet"
    ]


@pytest.mark.parametrize(
    "path", ["edge/nope.parquet", "edge/nope", "edge/nope/*.parquet", "edge/dat"]
)
def test_missing_path_raises_file_not_found(edge_bucket: str, path: str) -> None:
    uri = f"hf://buckets/{edge_bucket}/{path}"

    with pytest.raises(FileNotFoundError) as error:
        plhf.scan_bucket(uri)

    assert uri in str(error.value)


def test_missing_bucket_raises_file_not_found(edge_bucket: str) -> None:
    namespace = edge_bucket.split("/")[0]
    uri = f"hf://buckets/{namespace}/polars-hf-test-no-such-bucket/data"

    with pytest.raises(FileNotFoundError, match="not found"):
        plhf.scan_bucket(uri)
    with pytest.raises(FileNotFoundError, match="not found"):
        plhf.scan_bucket(f"{uri}/one.parquet")


@pytest.mark.parametrize("path", [_EMPTY_FILE, "edge/empty"])
def test_empty_file_is_rejected(edge_bucket: str, path: str) -> None:
    with pytest.raises(ValueError, match="is empty") as error:
        plhf.scan_bucket(f"hf://buckets/{edge_bucket}/{path}")

    assert f"hf://buckets/{edge_bucket}/{_EMPTY_FILE}" in str(error.value)


@pytest.mark.parametrize("path", ["edge/table", "edge/data", "edge/data.parquet"])
def test_invalid_token_raises_permission_error(edge_bucket: str, path: str) -> None:
    uri = f"hf://buckets/{edge_bucket}/{path}"

    with pytest.raises(PermissionError) as error:
        plhf.scan_bucket(uri, token="invalid-token-for-tests")

    assert f"'{edge_bucket}'" in str(error.value)
    assert "lacks access" in str(error.value)
    assert isinstance(error.value.__cause__, HfHubHTTPError)
    assert error.value.__cause__.response.status_code == 401


# ---- the listing of polars-hf against HfApi.list_bucket_tree ---------------

_ODD_NAMES = [
    "odd/with space/a b.parquet",
    "odd/ünï/日本.parquet",
    "odd/100%/50%25.parquet",
    "odd/#hash/q?.parquet",
    "odd/plus+and&/x=1;y.parquet",
]


@pytest.fixture(scope="module")
def odd_bucket(staging_api: HfApi) -> Iterator[str]:
    bucket_id = _create_staging_bucket(staging_api)
    try:
        add = [(b"x", path) for path in _ODD_NAMES]
        add += [(b"x", f"many/f{i:02d}.bin") for i in range(7)]
        add += [(b"x", "many/sub/g.bin"), (b"x", "many2/h.bin")]
        _staging_retry(lambda: staging_api.batch_bucket_files(bucket_id, add=add))
        yield bucket_id
    finally:
        _delete_staging_bucket(staging_api, bucket_id)


def _own_listing(bucket_id: str, prefix: str, recursive: bool) -> list:
    from huggingface_hub import constants
    from huggingface_hub.utils import build_hf_headers

    from polars_hf import read

    entries = read._list_tree(
        constants.ENDPOINT,
        build_hf_headers(),
        bucket_id,
        prefix,
        recursive=recursive,
        uri=f"hf://buckets/{bucket_id}/{prefix}",
        budget=read._Budget(bucket_id),
    )
    return [(e.type, e.path, e.size, e.xet_hash) for e in entries]


def _client_listing(api: HfApi, bucket_id: str, prefix: str, recursive: bool) -> list:
    listed = api.list_bucket_tree(bucket_id, prefix=prefix or None, recursive=recursive)
    return [
        (e.type, e.path, getattr(e, "size", None), getattr(e, "xet_hash", None))
        for e in listed
    ]


@pytest.mark.parametrize(
    ("prefix", "recursive"),
    [
        ("", True),
        ("", False),
        ("many/", True),
        ("many", False),
        ("many/f0", False),
        ("odd/", True),
        ("odd/with space/", True),
        ("odd/100%/", True),
        ("odd/#hash/", True),
        ("odd/#hash", False),
        ("odd/ünï/", True),
        ("nope/", True),
    ],
)
def test_listing_equals_list_bucket_tree(
    odd_bucket: str, staging_api: HfApi, prefix: str, recursive: bool
) -> None:
    own = _own_listing(odd_bucket, prefix, recursive)

    assert own == _client_listing(staging_api, odd_bucket, prefix, recursive)
    if prefix == "many/" and recursive:
        assert len(own) == 8


@pytest.mark.parametrize("recursive", [True, False])
def test_multi_page_listing_equals_one_page(
    odd_bucket: str, monkeypatch: pytest.MonkeyPatch, recursive: bool
) -> None:
    from polars_hf import read

    one_page = _own_listing(odd_bucket, "many/", recursive)
    monkeypatch.setattr(read, "_LIST_PAGE_LIMIT", 3)

    assert _own_listing(odd_bucket, "many/", recursive) == one_page
    assert len(one_page) == 8


def test_odd_names_are_listed_and_resolved(
    odd_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The files are one byte each, not parquet: capture the signed URLs
    # instead of scanning them.
    from polars_hf import read

    resolved: list[str] = []
    monkeypatch.setattr(
        read.pl, "scan_parquet", lambda urls, **kw: resolved.extend(urls)
    )

    plhf.scan_bucket(f"hf://buckets/{odd_bucket}/odd/")

    assert len(resolved) == len(_ODD_NAMES)


def test_missing_bucket_listing_raises_file_not_found(odd_bucket: str) -> None:
    namespace = odd_bucket.split("/")[0]
    missing = f"{namespace}/polars-hf-test-no-such-bucket"

    with pytest.raises(FileNotFoundError, match="not found"):
        _own_listing(missing, "", True)
