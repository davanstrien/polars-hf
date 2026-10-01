"""The write entry points the write, known-bug and property tests go through.

These helpers are the only place that names the current write design (the
``backend=`` values and the failure switches of the fake). A change of the
write API only needs an edit here.

Offline, both backends run against the fake Hub (see ``fakehub.py``): the hub
backend through the patched ``HfApi._batch_bucket_files``, the xet backend
through an in-memory upload commit and the ``/batch`` route. On staging the
same helpers run the real uploads.
"""

from __future__ import annotations

import polars as pl
import pytest
from fakehub import FakeHub

import polars_hf as plhf
from polars_hf import _sinks


def sink_default(frame: pl.DataFrame | pl.LazyFrame, uri: str, **kwargs) -> None:
    """Write with the default options of ``sink_bucket``."""
    plhf.sink_bucket(frame, uri, **kwargs)


def sink_streamed(frame: pl.DataFrame | pl.LazyFrame, uri: str, **kwargs) -> None:
    """Write through the backend that streams each file straight to the bucket."""
    plhf.sink_bucket(frame, uri, backend="xet", **kwargs)


def sink_staged(frame: pl.DataFrame | pl.LazyFrame, uri: str, **kwargs) -> None:
    """Write through the backend that stages the output on local disk."""
    plhf.sink_bucket(frame, uri, backend="hub", **kwargs)


# Every sink backend. Offline, both always run.
ALL_SINKS = [sink_streamed, sink_staged]


def staging_sinks() -> list:
    """Both backends as parameters for a staging test.

    The xet backend is skipped when the installed ``huggingface_hub`` /
    ``hf_xet`` cannot run it (the real check: nothing is patched on staging).
    """
    reason = _sinks.xet_unavailable_reason()
    streamed = pytest.param(
        sink_streamed,
        id="xet",
        marks=pytest.mark.skipif(reason is not None, reason=f"xet backend: {reason}"),
    )
    return [streamed, pytest.param(sink_staged, id="hub")]


def fail_stream_upload(fake_hub: FakeHub, number: int) -> None:
    """Make the upload of the ``number``-th file (1-based) of the xet backend fail.

    The stream raises ``ScriptedUploadError`` when it is finished. The hub
    backend has no per-file upload: use :func:`fail_batch`.
    """
    fake_hub.fail_stream_finish_on_call = number


def fail_batch(fake_hub: FakeHub, number: int) -> None:
    """Make the ``number``-th batch of bucket operations (1-based) fail.

    For the hub backend a batch is the upload and the registration of at most
    1,000 files (or a delete request); it raises ``ScriptedUploadError``. For
    the xet backend it is a registration or delete request, sent after the
    file data is uploaded; the fake answers 403 and the backend raises
    ``BucketRegistrationError``.
    """
    fake_hub.fail_batch_on_call = number
