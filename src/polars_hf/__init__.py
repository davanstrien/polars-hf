"""polars-hf: read and write Hugging Face buckets with polars."""

from __future__ import annotations

from polars_hf._uri import BucketPath, parse_bucket_uri
from polars_hf.read import scan_bucket
from polars_hf.write import sink_bucket

__version__ = "0.1.0"

__all__ = ["BucketPath", "parse_bucket_uri", "scan_bucket", "sink_bucket"]
