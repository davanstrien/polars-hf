"""The write entry points the known-bug and property tests go through.

These helpers are the only place that names the current write design
(``atomic=`` and the upload call counter of the fake). A change of the write
API only needs an edit here.
"""

from __future__ import annotations

import gc

import polars as pl
from fakehub import FakeHub

import polars_hf as plhf


def sink_default(frame: pl.DataFrame | pl.LazyFrame, uri: str, **kwargs) -> None:
    """Write with the default options of ``sink_bucket``."""
    plhf.sink_bucket(frame, uri, **kwargs)


def sink_streamed(frame: pl.DataFrame | pl.LazyFrame, uri: str, **kwargs) -> None:
    """Write through the path that streams each file straight to the bucket.

    Today that is ``atomic=False``. The files are uploaded when Polars drops
    the file objects, so the garbage collector runs before the helper returns.
    """
    try:
        plhf.sink_bucket(frame, uri, atomic=False, **kwargs)
    finally:
        gc.collect()


def fail_upload(fake_hub: FakeHub, number: int) -> None:
    """Make the ``number``-th upload to the bucket (1-based) fail."""
    fake_hub.fail_batch_on_call = number
