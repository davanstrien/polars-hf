"""Unit tests for ``_list_files``: which listed files a path names (no network)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from polars_hf._uri import parse_bucket_uri
from polars_hf.read import _Budget, _list_files


@dataclass
class _Entry:
    path: str
    size: int | None = 100
    type: str = "file"


class _StubApi:
    """Stand-in for ``HfApi``: one bucket that lists like the Hub.

    Recursive: every file whose path starts with the prefix (a string prefix).
    Not recursive: the direct children (files and directories) of the prefix
    when it is a directory, else the entries of its parent directory that
    start with the prefix.
    """

    def __init__(self, entries: list[_Entry]) -> None:
        self.entries = entries
        self.calls: list[tuple[str, str | None, bool]] = []

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        self.calls.append((bucket_id, prefix, recursive))
        prefix = prefix or ""
        paths = [entry.path for entry in self.entries]
        if recursive:
            return [entry for entry in self.entries if entry.path.startswith(prefix)]

        directory = prefix.rstrip("/")
        is_directory = directory == "" or any(
            path.startswith(directory + "/") for path in paths
        )
        if not is_directory:
            directory = prefix.rpartition("/")[0]
        base = directory + "/" if directory else ""
        children: dict[str, _Entry] = {}
        for entry in self.entries:
            if not entry.path.startswith(base):
                continue
            if not is_directory and not entry.path.startswith(prefix):
                continue
            child, _, rest = entry.path[len(base) :].partition("/")
            if rest:
                children[base + child] = _Entry(base + child, None, "directory")
            else:
                children[entry.path] = entry
        return list(children.values())


def _files(*paths: str) -> list[_Entry]:
    return [_Entry(path) for path in paths]


def _list(api: _StubApi, path: str) -> list[str]:
    uri = f"hf://buckets/ns/name/{path}" if path else "hf://buckets/ns/name"
    return _list_files(api, parse_bucket_uri(uri), uri, _Budget("ns/name"))


def test_one_recursive_listing_with_the_path_as_prefix() -> None:
    api = _StubApi(_files("data/a.parquet", "data/sub/b.parquet"))

    assert _list(api, "data") == ["data/a.parquet", "data/sub/b.parquet"]
    assert api.calls == [("ns/name", "data", True)]


def test_whole_bucket_lists_without_prefix() -> None:
    api = _StubApi(_files("a.parquet", "x/b.pq", "x/notes.txt"))

    assert _list(api, "") == ["a.parquet", "x/b.pq"]
    assert api.calls == [("ns/name", None, True)]


def test_glob_in_the_last_segment_lists_one_directory() -> None:
    api = _StubApi(
        _files(
            "data/run_0.parquet",
            "data/run_1.parquet",
            "data/x.csv",
            "data/sub/run_2.parquet",
        )
    )

    assert _list(api, "data/run_*.parquet") == [
        "data/run_0.parquet",
        "data/run_1.parquet",
    ]
    assert api.calls == [("ns/name", "data", False)]


def test_glob_at_the_bucket_root_lists_the_root_only() -> None:
    api = _StubApi(_files("a.parquet", "b.parquet", "x/c.parquet"))

    assert _list(api, "*.parquet") == ["a.parquet", "b.parquet"]
    assert api.calls == [("ns/name", None, False)]


@pytest.mark.parametrize(
    ("pattern", "prefix"),
    [
        ("data/**/*.parquet", "data/"),
        ("data/**", "data/"),
        ("da*/x.parquet", "da"),
        ("data/run_[01]/x.parquet", "data/run_"),
    ],
)
def test_other_globs_list_the_subtree_of_the_literal_prefix(
    pattern: str, prefix: str
) -> None:
    api = _StubApi(_files("data/x.parquet", "data/run_0/x.parquet"))

    _list(api, pattern)

    assert api.calls == [("ns/name", prefix, True)]


def test_directory_named_like_the_listing_prefix_does_not_hide_siblings() -> None:
    # 'data/part' is a directory and a prefix of the files the glob names.
    api = _StubApi(_files("data/part/x.parquet", "data/part1.parquet"))

    assert _list(api, "data/part*") == ["data/part1.parquet"]


@pytest.mark.parametrize("pattern", ["data/*/", "*/", "data/run[1]/", "data/**/"])
def test_glob_with_trailing_slash_is_rejected(pattern: str) -> None:
    api = _StubApi(_files("data/run1/x.parquet"))

    with pytest.raises(ValueError, match="a glob cannot end with '/'") as error:
        _list(api, pattern)
    assert "*.parquet" in str(error.value)
    assert api.calls == []


@pytest.mark.parametrize("pattern", ["data/**.parquet", "data/a**/x.parquet", "**x"])
def test_double_star_inside_a_segment_is_rejected(pattern: str) -> None:
    api = _StubApi(_files("data/x.parquet"))

    with pytest.raises(ValueError, match="must be a whole path segment"):
        _list(api, pattern)


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
    api = _StubApi(_files("run[1]/a.parquet", "run[1]/sub/c.pq", "run1/b.parquet"))

    assert _list(api, "run[1]") == ["run[1]/a.parquet", "run[1]/sub/c.pq"]
    # The directory is found in the listing of its parent, then listed.
    assert api.calls == [("ns/name", None, False), ("ns/name", "run[1]", True)]


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
