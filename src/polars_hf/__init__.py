"""polars-hf: read and write Hugging Face buckets with polars."""

from __future__ import annotations

import os

# Bucket reads fetch many small, high-latency range requests from the XET CDN.
# polars' default cloud-IO concurrency is max(cpu_threads, 10) ≈ 10, which
# starves those reads. Raise it (unless the user set their own) for a large
# speedup. Must be set before polars' first cloud IO; it is read once and cached.
os.environ.setdefault("POLARS_CONCURRENCY_BUDGET", "64")

from polars_hf._uri import BucketPath, parse_bucket_uri  # noqa: E402
from polars_hf.read import scan_bucket
from polars_hf.write import sink_bucket

__version__ = "0.1.0"

__all__ = ["BucketPath", "parse_bucket_uri", "scan_bucket", "sink_bucket"]
