"""Unit tests for ``_list_files``: which listed files a path names (no network)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from polars_hf._uri import parse_bucket_uri
from polars_hf.read import _list_files


@dataclass
class _Entry:
    path: str
    size: int | None = 100
    type: str = "file"


class _StubApi:
    """Stand-in for ``HfApi``: a bucket with string-prefix listing, like the Hub."""

    def __init__(self, entries: list[_Entry]) -> None:
        self.entries = entries
        self.calls: list[tuple[str, str | None, bool]] = []

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        self.calls.append((bucket_id, prefix, recursive))
        for entry in self.entries:
            if entry.path.startswith(prefix or ""):
                yield entry


def _files(*paths: str) -> list[_Entry]:
    return [_Entry(path) for path in paths]


def _list(api: _StubApi, path: str) -> list[str]:
    uri = f"hf://buckets/ns/name/{path}" if path else "hf://buckets/ns/name"
    return _list_files(api, parse_bucket_uri(uri), uri)


def test_one_recursive_listing_with_the_path_as_prefix() -> None:
    api = _StubApi(_files("data/a.parquet", "data/sub/b.parquet"))

    assert _list(api, "data") == ["data/a.parquet", "data/sub/b.parquet"]
    assert api.calls == [("ns/name", "data", True)]


def test_whole_bucket_lists_without_prefix() -> None:
    api = _StubApi(_files("a.parquet", "x/b.pq", "x/notes.txt"))

    assert _list(api, "") == ["a.parquet", "x/b.pq"]
    assert api.calls == [("ns/name", None, True)]


def test_glob_lists_the_prefix_before_the_first_glob_character() -> None:
    api = _StubApi(_files("data/run_0.parquet", "data/run_1.parquet", "data/x.csv"))

    assert _list(api, "data/run_*.parquet") == [
        "data/run_0.parquet",
        "data/run_1.parquet",
    ]
    assert api.calls == [("ns/name", "data/run_", True)]


def test_directory_entries_are_never_selected() -> None:
    entries = _files("i/a.parquet") + [_Entry("i/sub", size=None, type="directory")]
    api = _StubApi(entries)

    assert _list(api, "i/*") == ["i/a.parquet"]


def test_exact_file_wins_whatever_its_extension() -> None:
    api = _StubApi(_files("data", "data/a.parquet", "data.parquet"))

    assert _list(api, "data") == ["data"]


def test_trailing_slash_forces_the_directory_reading() -> None:
    api = _StubApi(_files("data", "data/a.parquet", "data.parquet"))

    assert _list(api, "data/") == ["data/a.parquet"]


def test_directory_excludes_string_prefix_siblings() -> None:
    api = _StubApi(_files("data/a.parquet", "data.parquet", "data2/b.parquet"))

    assert _list(api, "data") == ["data/a.parquet"]


@pytest.mark.parametrize("name", ["b.pq", "c.PARQUET", "d.Pq", "e.parquet"])
def test_directory_matches_parquet_extensions_case_insensitively(name: str) -> None:
    api = _StubApi(_files(f"j/{name}", "j/notes.txt", "j/parquet"))

    assert _list(api, "j") == [f"j/{name}"]


def test_explicit_glob_does_not_filter_by_extension() -> None:
    api = _StubApi(_files("i/a.parquet", "i/b.bin", "i/sub/c.parquet"))

    assert _list(api, "i/*") == ["i/a.parquet", "i/b.bin"]


def test_literal_file_with_glob_characters_wins_over_the_glob() -> None:
    api = _StubApi(_files("g/data[1].parquet", "g/data1.parquet"))

    assert _list(api, "g/data[1].parquet") == ["g/data[1].parquet"]


def test_literal_directory_with_glob_characters_wins_over_the_glob() -> None:
    api = _StubApi(_files("run[1]/a.parquet", "run1/b.parquet"))

    assert _list(api, "run[1]") == ["run[1]/a.parquet"]


def test_glob_is_used_when_no_literal_file_or_directory_exists() -> None:
    api = _StubApi(_files("g/data1.parquet", "g/data2.parquet"))

    assert _list(api, "g/data[1].parquet") == ["g/data1.parquet"]


def test_result_is_sorted() -> None:
    api = _StubApi(_files("d/b.parquet", "d/a.parquet", "d/C.parquet"))

    assert _list(api, "d") == ["d/C.parquet", "d/a.parquet", "d/b.parquet"]


@pytest.mark.parametrize("path", ["empty-dir", "missing.parquet", "data/*.parquet", ""])
def test_no_matches_raises_file_not_found(path: str) -> None:
    api = _StubApi(_files("other/notes.txt"))

    with pytest.raises(FileNotFoundError, match="no parquet files matched") as error:
        _list(api, path)
    assert "hf://buckets/ns/name" in str(error.value)


@pytest.mark.parametrize("path", ["d", "d/*.parquet", "d/empty.parquet"])
def test_empty_file_is_rejected_by_name(path: str) -> None:
    api = _StubApi([_Entry("d/a.parquet"), _Entry("d/empty.parquet", size=0)])

    with pytest.raises(ValueError, match="is empty") as error:
        _list(api, path)
    assert "hf://buckets/ns/name/d/empty.parquet" in str(error.value)
