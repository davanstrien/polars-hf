"""Parsing for ``hf://buckets/...`` URIs.

Mirrors the bucket semantics implemented in the polars fork
(``crates/polars-io/src/path_utils/hugging_face.rs``):

* ``hf://buckets/{namespace}/{name}/{path from root}``
* Buckets have **no** revision concept, so ``@revision`` is rejected.

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
        bucket) and may contain glob characters.
    """

    bucket_id: str
    path: str

    @property
    def is_glob(self) -> bool:
        """Whether ``path`` contains glob metacharacters."""
        return any(c in _GLOB_CHARS for c in self.path)

    @property
    def fs_path(self) -> str:
        """The path as understood by ``HfFileSystem`` (``buckets/...``)."""
        root = f"buckets/{self.bucket_id}"
        return f"{root}/{self.path}" if self.path else root


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
        If the URI is not a well-formed bucket URI, or if an ``@revision`` is
        present (buckets do not support revisions).
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

    # Buckets have no revision concept. Match the fork's explicit rejection.
    if "@" in remainder:
        raise ValueError(
            f"Hugging Face bucket URIs do not support @revision: {uri!r}"
        )

    parts = remainder.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"invalid Hugging Face bucket URI: {uri!r} "
            "(expected 'hf://buckets/{namespace}/{name}/{path}')"
        )

    bucket_id = f"{parts[0]}/{parts[1]}"
    path = "/".join(parts[2:])
    return BucketPath(bucket_id=bucket_id, path=path)
