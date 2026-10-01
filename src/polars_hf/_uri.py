"""Parsing for ``hf://buckets/...`` URIs.

* ``hf://buckets/{namespace}/{name}/{path from root}``
* Buckets have **no** revision concept, so ``@revision`` after the bucket name
  is rejected. An ``@`` in the path below the bucket is a normal character.
* An empty path segment, a ``..`` segment and whitespace at the end of the URI
  are rejected; one trailing ``/`` names a directory.

Datasets and Spaces are intentionally *not* handled here: stock polars already
reads ``hf://datasets/...`` / ``hf://spaces/...`` natively via ``pl.scan_parquet``.
This plugin exists to add the missing ``buckets`` path space.
"""

from __future__ import annotations

from dataclasses import dataclass

_GLOB_CHARS = frozenset("*?[]")


@dataclass(frozen=True)
class BucketPath:
    """A parsed ``hf://buckets/...`` URI.

    Attributes
    ----------
    bucket_id
        ``"{namespace}/{name}"`` identifying the bucket.
    path
        Path within the bucket, relative to its root. May be empty (the whole
        bucket), may contain glob characters, and keeps a trailing ``/`` if
        the URI had one.
    """

    bucket_id: str
    path: str

    @property
    def is_glob(self) -> bool:
        """Whether ``path`` contains glob metacharacters."""
        return any(c in _GLOB_CHARS for c in self.path)

    @property
    def fs_path(self) -> str:
        """The URI without its ``hf://`` scheme (``buckets/{bucket_id}/{path}``).

        This is the form ``HfFileSystem`` takes; the write path uses it.
        """
        root = f"buckets/{self.bucket_id}"
        return f"{root}/{self.path}" if self.path else root


def _check_path_segments(segments: list[str], uri: str) -> None:
    """Raise ``ValueError`` for a path the Hub cannot store or return."""
    # One trailing '/' names a directory ('.../data/'); it is not an empty
    # segment.
    if segments and segments[-1] == "":
        segments = segments[:-1]
    for segment in segments:
        if segment == "":
            raise ValueError(
                f"Hugging Face bucket URI has an empty path segment ('//'): {uri!r}"
            )
        if segment == "..":
            raise ValueError(
                f"Hugging Face bucket URI has a '..' path segment: {uri!r} "
                "(bucket paths are not resolved relative to a directory)"
            )


def parse_bucket_uri(uri: str) -> BucketPath:
    """Parse an ``hf://buckets/{namespace}/{name}/{path}`` URI.

    Parameters
    ----------
    uri
        The Hugging Face bucket URI.

    Returns
    -------
    BucketPath

    Raises
    ------
    ValueError
        If the URI is not a well-formed bucket URI: an ``@revision`` after the
        bucket name (buckets do not support revisions), an empty path segment
        (``a//b``), a ``..`` segment, or whitespace at the end of the URI. One
        trailing ``/`` is allowed and names a directory.
    """
    if not uri.startswith("hf://"):
        raise ValueError(f"not a Hugging Face URI (must start with 'hf://'): {uri!r}")

    rest = uri[len("hf://") :]
    kind, _, remainder = rest.partition("/")

    if kind != "buckets":
        if kind in ("datasets", "spaces"):
            raise ValueError(
                f"hf://{kind}/... is read natively by polars; "
                f"use pl.scan_parquet({uri!r}) instead. "
                "polars-hf only handles hf://buckets/... URIs."
            )
        raise ValueError(
            f"invalid Hugging Face bucket URI: {uri!r} "
            "(expected 'hf://buckets/{namespace}/{name}/{path}')"
        )

    parts = remainder.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"invalid Hugging Face bucket URI: {uri!r} "
            "(expected 'hf://buckets/{namespace}/{name}/{path}')"
        )

    # Buckets have no revision concept: reject one explicitly, but only where
    # a revision would be, after the bucket name. Below the
    # bucket, '@' is a normal character of a file or directory name.
    if "@" in parts[0] or "@" in parts[1]:
        raise ValueError(f"Hugging Face bucket URIs do not support @revision: {uri!r}")

    # A URI copied with a trailing space or newline names a path that does not
    # exist; the Hub would answer "not found" for it.
    if uri != uri.rstrip():
        raise ValueError(
            f"Hugging Face bucket URI ends with whitespace: {uri!r} "
            "(remove it, or add a trailing '/' if a directory name really "
            "ends with a space)"
        )

    segments = parts[2:]
    _check_path_segments(segments, uri)

    bucket_id = f"{parts[0]}/{parts[1]}"
    path = "/".join(segments)
    return BucketPath(bucket_id=bucket_id, path=path)
