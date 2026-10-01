"""Glob matching for bucket paths.

A bucket listing returns complete file paths (``a/b/c.parquet``), so a glob is
matched against the whole path, client-side:

* ``*`` matches any run of characters inside one path segment;
* ``?`` matches one character inside a segment;
* ``[abc]`` / ``[a-z]`` / ``[!abc]`` match one character inside a segment;
* a segment that is exactly ``**`` matches zero or more directories
  (``a/**/x.parquet`` matches ``a/x.parquet`` and ``a/b/c/x.parquet``); as the
  last segment it matches every file below the directory. ``**`` as a part of
  a segment (``a/**.parquet``) is an error, as in fsspec and Polars;
* braces (``{a,b}``) are not expanded: they are literal characters.

A leading ``.`` is not special: ``*`` matches hidden files too.
"""

from __future__ import annotations

import re

from polars_hf._uri import _GLOB_CHARS


def has_glob(text: str) -> bool:
    """Whether ``text`` contains a glob character."""
    return any(char in _GLOB_CHARS for char in text)


def literal_prefix(pattern: str) -> str:
    """The part of ``pattern`` before its first glob character.

    Every path the pattern can match starts with this string, so it is the
    narrowest prefix to list.
    """
    for index, char in enumerate(pattern):
        if char in _GLOB_CHARS:
            return pattern[:index]
    return pattern


def _class_end(segment: str, start: int) -> int:
    """Index of the ``]`` that closes the class opened at ``start``, or -1."""
    index = start + 1
    if index < len(segment) and segment[index] in "!^":
        index += 1
    # A ']' directly after the opening bracket is a member of the class.
    if index < len(segment) and segment[index] == "]":
        index += 1
    return segment.find("]", index)


def _class_body(body: str) -> str:
    """Escape the members of a character class; keep ``a-z`` ranges.

    Members are read left to right: ``x-y`` is a range whenever a ``-`` stands
    between two characters (``[--0]`` is the range from ``-`` to ``0``); a
    ``-`` in any other position is a literal.
    """
    escaped = []
    index = 0
    while index < len(body):
        if index + 2 < len(body) and body[index + 1] == "-":
            escaped.append(re.escape(body[index]) + "-" + re.escape(body[index + 2]))
            index += 3
        else:
            escaped.append(re.escape(body[index]))
            index += 1
    return "".join(escaped)


def _segment_to_regex(segment: str) -> str:
    """Translate one path segment (no ``/``) of a glob to a regex."""
    parts = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        elif char == "[":
            end = _class_end(segment, index)
            if end == -1:
                # No closing bracket: a literal '['.
                parts.append(re.escape(char))
            else:
                body = segment[index + 1 : end]
                negated = body[:1] in ("!", "^")
                if negated:
                    body = body[1:]
                # The lookahead keeps a negated class from matching '/'.
                parts.append(
                    "(?!/)[" + ("^" if negated else "") + _class_body(body) + "]"
                )
                index = end
        else:
            parts.append(re.escape(char))
        index += 1
    return "".join(parts)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob to a regex; use ``fullmatch`` on a complete file path.

    Raises ``ValueError`` when ``**`` is only part of a segment (``**.parquet``).
    """
    segments = pattern.split("/")
    parts = []
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        if segment == "**":
            # Zero or more directories; as the last segment, any path below.
            parts.append(".*" if is_last else "(?:[^/]+/)*")
        elif "**" in segment:
            raise ValueError(
                f"invalid glob {pattern!r}: '**' must be a whole path segment "
                "(use '**/*.parquet' for all parquet files below a directory)"
            )
        elif is_last:
            parts.append(_segment_to_regex(segment))
        else:
            parts.append(_segment_to_regex(segment) + "/")
    try:
        return re.compile("".join(parts), re.DOTALL)
    except re.error:
        # A class with a reversed range ('[z-a]') is not a valid pattern: it
        # matches nothing.
        return re.compile(r"(?!)")
