"""Unit tests for ``_list_files`` path handling (no network)."""

from __future__ import annotations

import pytest

from polars_hf.read import _list_files


class _NoListingFS:
    """Stand-in filesystem that fails if a single-file path triggers listing."""

    def invalidate_cache(self) -> None:
        raise AssertionError("single-file paths must not invalidate the cache")

    def glob(self, pattern: str) -> list[str]:
        raise AssertionError("single-file paths must not be globbed")


@pytest.mark.parametrize("ext", ["parquet", "pq", "PARQUET", "Pq"])
def test_single_file_extensions_skip_listing(ext: str) -> None:
    # sink_bucket accepts .pq as a parquet extension and matches extensions
    # case-insensitively; scanning the same file back must take the
    # single-file path, not directory expansion.
    files = _list_files(_NoListingFS(), f"hf://buckets/ns/name/data.{ext}")
    assert files == [f"buckets/ns/name/data.{ext}"]


class _EmptyFS:
    def invalidate_cache(self) -> None:
        pass

    def glob(self, pattern: str) -> list[str]:
        return []


def test_no_matches_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="no parquet files matched"):
        _list_files(_EmptyFS(), "hf://buckets/ns/name/empty-dir")
