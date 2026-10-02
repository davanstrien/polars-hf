"""Packaging checks (no network)."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import version

import polars_hf as plhf


def test_version_comes_from_package_metadata() -> None:
    assert plhf.__version__ == version("polars-hf")


def test_public_api() -> None:
    assert sorted(plhf.__all__) == [
        "BucketPath",
        "BucketRegistrationError",
        "count_rows",
        "parse_bucket_uri",
        "scan_bucket",
        "sink_bucket",
    ]


def test_import_does_not_set_polars_settings() -> None:
    # POLARS_CONCURRENCY_BUDGET applies to every cloud scan of the process:
    # the import must leave it to the user.
    env = dict(os.environ)
    env.pop("POLARS_CONCURRENCY_BUDGET", None)
    code = (
        "import os, polars_hf; "
        "print([name for name in os.environ if name.startswith('POLARS_')])"
    )

    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr
    polars_names = [name for name in env if name.startswith("POLARS_")]
    assert done.stdout.strip() == str(polars_names)
