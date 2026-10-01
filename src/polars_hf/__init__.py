"""polars-hf: read and write Hugging Face buckets with polars."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from polars_hf._sinks import BucketRegistrationError
from polars_hf._uri import BucketPath, parse_bucket_uri
from polars_hf.read import scan_bucket
from polars_hf.write import sink_bucket

try:
    __version__ = version("polars-hf")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "BucketPath",
    "BucketRegistrationError",
    "parse_bucket_uri",
    "scan_bucket",
    "sink_bucket",
]
