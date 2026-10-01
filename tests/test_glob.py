"""Unit tests for the bucket-path glob matcher (no network)."""

from __future__ import annotations

import pytest

from polars_hf._glob import glob_to_regex, literal_prefix


def _matches(pattern: str, path: str) -> bool:
    return glob_to_regex(pattern).fullmatch(path) is not None


@pytest.mark.parametrize(
    ("pattern", "prefix"),
    [
        ("data/*.parquet", "data/"),
        ("data/run_?.parquet", "data/run_"),
        ("a/b[1].parquet", "a/b"),
        ("**/*.parquet", ""),
        ("no/glob.parquet", "no/glob.parquet"),
        ("a]b", "a]b"),
    ],
)
def test_literal_prefix(pattern: str, prefix: str) -> None:
    assert literal_prefix(pattern) == prefix


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        # '*' stays inside one segment.
        ("d/*", "d/a.parquet", True),
        ("d/*", "d/sub/a.parquet", False),
        ("d/*.parquet", "d/a.parquet", True),
        ("d/*.parquet", "d/a.pq", False),
        ("d/*.parquet", "d/.hidden.parquet", True),
        ("*/a.parquet", "d/a.parquet", True),
        ("*/a.parquet", "d/e/a.parquet", False),
        ("d/run_*.parquet", "d/run_10.parquet", True),
        # '?' is one character, never '/'.
        ("d/run_?.parquet", "d/run_1.parquet", True),
        ("d/run_?.parquet", "d/run_10.parquet", False),
        ("d?a.parquet", "d/a.parquet", False),
        # Character classes.
        ("d/data[12].parquet", "d/data1.parquet", True),
        ("d/data[12].parquet", "d/data3.parquet", False),
        ("d/data[0-9].parquet", "d/data7.parquet", True),
        ("d/data[!0-9].parquet", "d/dataX.parquet", True),
        ("d/data[!0-9].parquet", "d/data7.parquet", False),
        ("d[!x]a.parquet", "d/a.parquet", False),
        ("d/[]a].parquet", "d/].parquet", True),
        ("d/[a.b].parquet", "d/..parquet", True),
        ("d/[a.b].parquet", "d/c.parquet", False),
        # A range can start at '-'; a '-' at either end is a literal.
        ("d/[--0].parquet", "d/..parquet", True),
        ("d/[--0].parquet", "d/-.parquet", True),
        ("d/[--0].parquet", "d/0.parquet", True),
        ("d/[--0].parquet", "d/1.parquet", False),
        ("d[--0]a.parquet", "d/a.parquet", False),
        ("d/[a-].parquet", "d/-.parquet", True),
        ("d/[-a].parquet", "d/-.parquet", True),
        ("d/[a-c-e].parquet", "d/-.parquet", True),
        ("d/[a-c-e].parquet", "d/d.parquet", False),
        # Braces are not expanded.
        ("d/{a,b}.parquet", "d/{a,b}.parquet", True),
        ("d/{a,b}.parquet", "d/a.parquet", False),
        # An unclosed bracket is a literal.
        ("d/a[.parquet", "d/a[.parquet", True),
        # A reversed range matches nothing.
        ("d/data[z-a].parquet", "d/dataz.parquet", False),
        # '**' crosses directories.
        ("d/**/*.parquet", "d/a.parquet", True),
        ("d/**/*.parquet", "d/x/y/a.parquet", True),
        ("d/**/*.parquet", "e/a.parquet", False),
        ("**/*.parquet", "a.parquet", True),
        ("**/*.parquet", "x/y/a.parquet", True),
        ("d/**", "d/x/y/a.bin", True),
        ("d/**", "d2/a.bin", False),
        # Regex metacharacters in the pattern are literals.
        ("d/a+b(1).parquet", "d/a+b(1).parquet", True),
        ("d/a.parquet", "d/aXparquet", False),
        ("d/a$.parquet", "d/a$.parquet", True),
    ],
)
def test_glob_matching(pattern: str, path: str, expected: bool) -> None:
    assert _matches(pattern, path) is expected


def test_newline_in_a_name_is_matched_by_star() -> None:
    assert _matches("d/*.parquet", "d/a\nb.parquet")


@pytest.mark.parametrize("pattern", ["d/**.parquet", "d/a**", "**a/x", "d/***"])
def test_double_star_inside_a_segment_is_an_error(pattern: str) -> None:
    with pytest.raises(ValueError, match="must be a whole path segment"):
        glob_to_regex(pattern)
