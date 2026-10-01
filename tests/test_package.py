"""Packaging checks (no network)."""

from __future__ import annotations

from importlib.metadata import version

import polars_hf as plhf


def test_version_comes_from_package_metadata() -> None:
    assert plhf.__version__ == version("polars-hf")


def test_public_api() -> None:
    assert sorted(plhf.__all__) == [
        "BucketPath",
        "parse_bucket_uri",
        "scan_bucket",
        "sink_bucket",
    ]
