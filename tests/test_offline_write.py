"""Offline write tests: ``sink_bucket`` against the fake Hub (no network).

Tests with a ``sink`` parameter run once per sink backend (see ``sinks.py``).
The tests named after a bug letter are the regression tests of the bugs that
were listed in ``test_known_bugs.py``.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading

import polars as pl
import pytest
from fakehub import HUB, FakeHub, ScriptedUploadError
from huggingface_hub.errors import HfHubHTTPError
from polars.testing import assert_frame_equal
from sinks import (
    ALL_SINKS,
    fail_batch,
    fail_stream_upload,
    sink_default,
    sink_staged,
    sink_streamed,
)

import polars_hf as plhf
from polars_hf import BucketRegistrationError, _sinks

_READERS = {
    "parquet": pl.read_parquet,
    "csv": pl.read_csv,
    "ipc": pl.read_ipc,
    "ndjson": pl.read_ndjson,
}

both_sinks = pytest.mark.parametrize("sink", ALL_SINKS)


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _relative(paths: list[str], prefix: str) -> list[str]:
    return sorted(p[len(prefix) :] for p in paths)


def _failing_frame() -> pl.LazyFrame:
    """A LazyFrame that raises ``RuntimeError`` when the engine runs it."""

    def explode(batch: pl.DataFrame) -> pl.DataFrame:
        raise RuntimeError("scripted sink failure")

    return pl.LazyFrame({"g": ["a", "b"], "n": [1, 2]}).map_batches(explode)


def _snapshot(fake_hub: FakeHub, bucket_id: str) -> dict[str, bytes]:
    """Every object of the bucket, by path."""
    return {path: fake_hub.read(bucket_id, path) for path in fake_hub.files(bucket_id)}


# ---- single file -----------------------------------------------------------


@both_sinks
def test_single_file_round_trip(fake_hub: FakeHub, fake_bucket: str, sink) -> None:
    df = pl.DataFrame({"n": range(100), "g": ["a", "b"] * 50})
    uri = _uri(fake_bucket, "out/eager.parquet")

    sink(df, uri)

    assert fake_hub.files(fake_bucket) == ["out/eager.parquet"]
    assert_frame_equal(plhf.scan_bucket(uri).collect(), df)


@both_sinks
def test_single_file_write_is_one_batch(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    sink(df.lazy(), _uri(fake_bucket, "out/lazy.parquet"))

    assert len(fake_hub.batch_calls) == 1
    call = fake_hub.batch_calls[0]
    assert call.bucket_id == fake_bucket
    assert list(call.added) == ["out/lazy.parquet"]
    assert call.added["out/lazy.parquet"] == len(
        fake_hub.read(fake_bucket, "out/lazy.parquet")
    )


@both_sinks
@pytest.mark.parametrize("ext", ["parquet", "csv", "ipc", "ndjson"])
def test_single_file_formats(
    fake_hub: FakeHub, fake_bucket: str, sink, ext: str
) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})

    sink(df.lazy(), _uri(fake_bucket, f"out/lazy.{ext}"))

    back = _READERS[ext](io.BytesIO(fake_hub.read(fake_bucket, f"out/lazy.{ext}")))
    assert_frame_equal(back, df)


@both_sinks
def test_format_override(fake_hub: FakeHub, fake_bucket: str, sink) -> None:
    # Extension says .data but we force parquet.
    df = pl.DataFrame({"a": [1, 2]})

    sink(df, _uri(fake_bucket, "out/override.data"), format="parquet")

    back = pl.read_parquet(io.BytesIO(fake_hub.read(fake_bucket, "out/override.data")))
    assert_frame_equal(back, df)


@both_sinks
def test_single_file_rejects_directory_uri(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    with pytest.raises(ValueError, match="names a directory"):
        sink(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "out/"), format="parquet")

    assert fake_hub.batch_calls == []


@both_sinks
def test_explicit_token_is_used_for_writes(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    fake_hub.accept_token("hf_explicit_write_token")
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})

    sink(df, _uri(fake_bucket, "out/token.parquet"), token="hf_explicit_write_token")
    sink(
        df,
        _uri(fake_bucket, "parts"),
        partition_by="g",
        mode="overwrite",
        token="hf_explicit_write_token",
    )

    assert len(fake_hub.files(fake_bucket)) == 3
    sent = {request.authorization for request in fake_hub.requests}
    assert sent == {"Bearer hf_explicit_write_token"}


# ---- partitioned -----------------------------------------------------------


@both_sinks
def test_partition_by_key_round_trip(fake_hub: FakeHub, fake_bucket: str, sink) -> None:
    df = pl.DataFrame({"g": ["a", "a", "b", "c", "c", "c"], "n": range(6)})
    base = _uri(fake_bucket, "parts")

    sink(df, base, partition_by="g")

    assert _relative(fake_hub.files(fake_bucket, "parts/"), "parts/") == [
        "g=a/00000000.parquet",
        "g=b/00000000.parquet",
        "g=c/00000000.parquet",
    ]
    back = plhf.scan_bucket(f"{base}/**/*.parquet").collect()
    assert_frame_equal(back.sort("n"), df)


@both_sinks
def test_partition_by_size_round_trip(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    df = pl.DataFrame({"n": range(1000)})
    base = _uri(fake_bucket, "shards")

    sink(df, base, max_rows_per_file=250)

    assert len(fake_hub.files(fake_bucket, "shards/")) == 4  # 1000 rows / 250
    back = plhf.scan_bucket(base).collect()
    assert_frame_equal(back.sort("n"), df)


@both_sinks
def test_partition_key_and_size(fake_hub: FakeHub, fake_bucket: str, sink) -> None:
    df = pl.DataFrame({"g": ["a"] * 500 + ["b"] * 500, "n": range(1000)})
    base = _uri(fake_bucket, "ks")

    sink(df, base, partition_by="g", max_rows_per_file=300)

    files = fake_hub.files(fake_bucket, "ks/")
    # 2 keys x ceil(500/300)=2 files each = 4
    assert len(files) == 4
    assert all(("g=a/" in f) or ("g=b/" in f) for f in files)
    assert plhf.scan_bucket(base).collect().height == 1000


@both_sinks
def test_partitioned_write_to_bucket_root(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})

    sink(df, f"hf://buckets/{fake_bucket}", partition_by="g")

    assert fake_hub.files(fake_bucket) == [
        "g=a/00000000.parquet",
        "g=b/00000000.parquet",
    ]


@pytest.mark.parametrize("fmt", ["parquet", "csv", "ipc", "ndjson"])
def test_backends_write_the_same_object_names(
    fake_hub: FakeHub, fake_bucket: str, fmt: str
) -> None:
    # Two keys of different types, null keys, values that need encoding, and
    # more than 16 files in one partition (the file index is hexadecimal).
    groups = ["a/b", "x=y", "a b", "é:%", None] + ["many"] * 20
    flags = [True, False, None, True, None] + [True] * 20
    df = pl.DataFrame({"g": groups, "b": flags, "n": range(25)})
    options = {"partition_by": ["g", "b"], "max_rows_per_file": 1, "format": fmt}

    sink_streamed(df, _uri(fake_bucket, "streamed"), **options)
    sink_staged(df, _uri(fake_bucket, "staged"), **options)

    streamed = _relative(fake_hub.files(fake_bucket, "streamed/"), "streamed/")
    staged = _relative(fake_hub.files(fake_bucket, "staged/"), "staged/")
    assert streamed == staged
    extension = "jsonl" if fmt == "ndjson" else fmt
    assert len(streamed) == 25
    assert f"g=many/b=true/00000013.{extension}" in streamed
    assert f"g=a%2Fb/b=true/00000000.{extension}" in streamed
    null = "__HIVE_DEFAULT_PARTITION__"
    assert f"g={null}/b={null}/00000000.{extension}" in streamed
    # Every name holds the same rows with both backends.
    for name in streamed:
        from_streamed = fake_hub.read(fake_bucket, f"streamed/{name}")
        from_staged = fake_hub.read(fake_bucket, f"staged/{name}")
        assert_frame_equal(
            _READERS[fmt](io.BytesIO(from_streamed)),
            _READERS[fmt](io.BytesIO(from_staged)),
        )


@both_sinks
@pytest.mark.parametrize(
    ("key", "directory"),
    [
        (None, "g=__HIVE_DEFAULT_PARTITION__"),
        ("a/b", "g=a%2Fb"),
        ("a b", "g=a%20b"),
        ("x=y", "g=x%3Dy"),
    ],
)
def test_partition_names_are_native(
    fake_hub: FakeHub, fake_bucket: str, sink, key: str | None, directory: str
) -> None:
    # Bug c: the streamed write used the raw key ("g=None", "g=a/b").
    df = pl.DataFrame({"g": [key], "n": [1]}, schema={"g": pl.String, "n": pl.Int64})

    sink(df, _uri(fake_bucket, "parts"), partition_by="g")

    assert fake_hub.files(fake_bucket) == [f"parts/{directory}/00000000.parquet"]


@both_sinks
def test_large_write_is_chunked(fake_hub: FakeHub, fake_bucket: str, sink) -> None:
    # At most 1,000 operations per batch, so 1,001 files need two batches.
    df = pl.DataFrame({"n": range(1001)})

    sink(df, _uri(fake_bucket, "many"), max_rows_per_file=1)

    assert [len(call.added) for call in fake_hub.batch_calls] == [1000, 1]
    assert len(fake_hub.files(fake_bucket, "many/")) == 1001


@both_sinks
@pytest.mark.parametrize("partition", [{"partition_by": "g"}, {"max_rows_per_file": 2}])
def test_empty_partitioned_write_is_one_empty_file(
    fake_hub: FakeHub, fake_bucket: str, sink, partition: dict
) -> None:
    # Bug e: an empty frame with partition_by wrote no file and raised nothing,
    # so a later scan of the prefix failed with FileNotFoundError.
    schema = {"g": pl.String, "n": pl.Int64}
    df = pl.DataFrame({"g": [], "n": []}, schema=schema)
    base = _uri(fake_bucket, "empty")

    sink(df, base, **partition)

    assert fake_hub.files(fake_bucket) == ["empty/00000000.parquet"]
    assert_frame_equal(plhf.scan_bucket(base).collect(), df)


# ---- failures --------------------------------------------------------------


@both_sinks
def test_failed_single_file_sink_keeps_existing_object(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # Bug a: the failed sink replaced the object with an empty file.
    fake_hub.put_parquet(fake_bucket, "keep.parquet", pl.DataFrame({"a": [1, 2, 3]}))
    before = _snapshot(fake_hub, fake_bucket)

    with pytest.raises(RuntimeError, match="scripted sink failure"):
        sink(_failing_frame(), _uri(fake_bucket, "keep.parquet"), mode="overwrite")

    assert _snapshot(fake_hub, fake_bucket) == before
    assert fake_hub.batch_calls == []


@both_sinks
@pytest.mark.parametrize("mode", ["append", "overwrite"])
def test_failed_partitioned_sink_changes_nothing(
    fake_hub: FakeHub, fake_bucket: str, sink, mode: str
) -> None:
    plhf.sink_bucket(
        pl.DataFrame({"g": ["a", "z"], "n": [1, 2]}),
        _uri(fake_bucket, "parts"),
        partition_by="g",
    )
    before = _snapshot(fake_hub, fake_bucket)
    fake_hub.batch_calls.clear()

    with pytest.raises(RuntimeError, match="scripted sink failure"):
        sink(_failing_frame(), _uri(fake_bucket, "parts"), partition_by="g", mode=mode)

    assert _snapshot(fake_hub, fake_bucket) == before
    assert fake_hub.batch_calls == []


@both_sinks
def test_lazy_sink_is_rejected(fake_hub: FakeHub, fake_bucket: str, sink) -> None:
    # Bug b: lazy=True reached polars, the plan was dropped and an empty
    # object was committed.
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="lazy=True is not supported"):
        sink(df, _uri(fake_bucket, "lazy.parquet"), lazy=True)
    with pytest.raises(ValueError, match="lazy=True is not supported"):
        sink(df, _uri(fake_bucket, "lazy"), partition_by="a", lazy=True)

    assert fake_hub.files(fake_bucket) == []
    assert fake_hub.requests == []


def test_lazy_false_is_accepted(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    plhf.sink_bucket(df, _uri(fake_bucket, "eager.parquet"), lazy=False)

    assert fake_hub.files(fake_bucket) == ["eager.parquet"]


def test_atomic_argument_is_rejected(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(TypeError, match="no longer accepts atomic="):
        plhf.sink_bucket(df, _uri(fake_bucket, "p"), partition_by="g", atomic=False)

    assert fake_hub.requests == []


@both_sinks
def test_upload_error_propagates_and_registers_nothing(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # Bug d: the streamed write uploaded in __del__, so the error was
    # swallowed and the files uploaded before it stayed in the bucket.
    df = pl.DataFrame({"g": ["a", "b", "c"], "n": [1, 2, 3]})
    # The staged backend uploads one batch; the stream backend uploads three files.
    if sink is sink_staged:
        fail_batch(fake_hub, 1)
    else:
        fail_stream_upload(fake_hub, 2)

    with pytest.raises(ScriptedUploadError):
        sink(df, _uri(fake_bucket, "parts"), partition_by="g")

    assert fake_hub.files(fake_bucket) == []


def test_failed_stream_write_raises_the_upload_error(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # Polars reports a failed write() as its own ComputeError; the stream backend
    # raises the error of the upload stream instead.
    fake_hub.put_parquet(fake_bucket, "keep.parquet", pl.DataFrame({"a": [1]}))
    before = _snapshot(fake_hub, fake_bucket)
    fake_hub.fail_stream_write_on_call = 1

    with pytest.raises(ScriptedUploadError) as error:
        sink_streamed(
            pl.DataFrame({"a": [1, 2]}),
            _uri(fake_bucket, "keep.parquet"),
            mode="overwrite",
        )

    assert isinstance(error.value.__cause__, pl.exceptions.PolarsError)
    assert _snapshot(fake_hub, fake_bucket) == before
    assert fake_hub.batch_calls == []
    assert [commit.aborted for commit in fake_hub.commits] == [True]


def test_streamed_files_are_stored_before_they_are_registered(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})

    sink_streamed(df, _uri(fake_bucket, "parts"), partition_by="g")

    # The fake refuses a hash it has not received: a registration sent before
    # wait_to_finish() would have failed.
    (commit,) = fake_hub.commits
    assert commit.finished and not commit.aborted
    assert [stream.finished for stream in commit.streams] == [True, True]
    assert [call.via for call in fake_hub.batch_calls] == ["http"]


def test_keyboard_interrupt_stops_the_xet_session(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    backend = _sinks.StreamBackend(fake_bucket, token=None)

    def interrupted(target: object) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        backend.write_file(interrupted, "out.parquet")

    # The commit is cancelled and the shared session is stopped.
    (commit,) = fake_hub.commits
    assert commit.aborted
    assert fake_hub.session_aborts == 1
    assert fake_hub.batch_calls == []


def test_keyboard_interrupt_survives_a_failing_commit_abort(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakehub import MemoryCommit

    def fails(self: MemoryCommit) -> None:
        raise RuntimeError("abort failed")

    monkeypatch.setattr(MemoryCommit, "abort", fails)
    backend = _sinks.StreamBackend(fake_bucket, token=None)

    def interrupted(target: object) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        backend.write_file(interrupted, "out.parquet")

    # The session was still stopped.
    assert fake_hub.session_aborts == 1


def test_keyboard_interrupt_survives_a_failing_session_abort(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fails() -> None:
        raise RuntimeError("session abort failed")

    monkeypatch.setattr(_sinks, "abort_xet_session", fails)
    backend = _sinks.StreamBackend(fake_bucket, token=None)

    def interrupted(target: object) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        backend.write_file(interrupted, "out.parquet")

    assert [commit.aborted for commit in fake_hub.commits] == [True]


def test_rejected_registration_raises_with_the_failed_paths(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    df = pl.DataFrame({"g": ["a", "b", "c"], "n": [1, 2, 3]})
    fake_hub.reject_paths = {"parts/g=b/00000000.parquet"}

    with pytest.raises(BucketRegistrationError) as error:
        sink_streamed(df, _uri(fake_bucket, "parts"), partition_by="g")

    assert [failure["path"] for failure in error.value.failures] == [
        "parts/g=b/00000000.parquet"
    ]
    assert "parts/g=b/00000000.parquet" in str(error.value)
    assert "rejected 1 of 3 operation(s)" in str(error.value)
    assert "the other 2 were applied" in str(error.value)
    # Not transactional: the bucket applied the other operations.
    assert fake_hub.files(fake_bucket) == [
        "parts/g=a/00000000.parquet",
        "parts/g=c/00000000.parquet",
    ]


def test_registration_with_all_paths_rejected(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The Hub answers 422 when every operation of a request fails.
    fake_hub.reject_paths = {"out.parquet"}

    with pytest.raises(BucketRegistrationError, match="out.parquet") as error:
        sink_streamed(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "out.parquet"))

    assert "rejected 1 of 1 operation(s)" in str(error.value)
    assert "applied" not in str(error.value)
    assert fake_hub.files(fake_bucket) == []


@both_sinks
def test_write_of_more_than_one_batch_is_not_transactional(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # Documented limitation: the second batch fails, the first one stays.
    df = pl.DataFrame({"n": range(1001)})
    fail_batch(fake_hub, 2)

    with pytest.raises((ScriptedUploadError, BucketRegistrationError)):
        sink(df, _uri(fake_bucket, "many"), max_rows_per_file=1)

    assert len(fake_hub.files(fake_bucket, "many/")) == 1000


# ---- mode ------------------------------------------------------------------


def test_unknown_mode_is_rejected(fake_hub: FakeHub, fake_bucket: str) -> None:
    with pytest.raises(ValueError, match="unknown mode 'replace'"):
        plhf.sink_bucket(
            pl.DataFrame({"a": [1]}), _uri(fake_bucket, "a.parquet"), mode="replace"
        )

    assert fake_hub.requests == []


def _run_ids(monkeypatch: pytest.MonkeyPatch, *tokens: str) -> None:
    """Make the next ``mode="append"`` calls use these run ids, in order."""
    remaining = list(tokens)
    monkeypatch.setattr(_sinks, "new_run_id", lambda: remaining.pop(0))


@both_sinks
def test_default_mode_raises_if_the_destination_exists(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    df = pl.DataFrame({"g": ["a"], "n": [1]})
    file_uri = _uri(fake_bucket, "out.parquet")
    base = _uri(fake_bucket, "parts")
    sink(df, file_uri)
    sink(df, base, partition_by="g")
    before = _snapshot(fake_hub, fake_bucket)
    fake_hub.batch_calls.clear()
    fake_hub.commits.clear()

    with pytest.raises(FileExistsError) as file_error:
        sink(df, file_uri)
    with pytest.raises(FileExistsError) as prefix_error:
        sink(df, base, partition_by="g")

    assert file_uri in str(file_error.value)
    assert "mode='overwrite'" in str(file_error.value)
    assert base in str(prefix_error.value)
    assert "mode='append'" in str(prefix_error.value)
    assert "mode='overwrite'" in str(prefix_error.value)
    assert _snapshot(fake_hub, fake_bucket) == before
    assert fake_hub.batch_calls == []
    assert fake_hub.commits == []


@both_sinks
def test_append_to_a_single_file_is_refused(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    df = pl.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="cannot append to the file") as error:
        sink(df, _uri(fake_bucket, "out.parquet"), mode="append")

    assert "mode='overwrite'" in str(error.value)
    assert fake_hub.requests == []


@both_sinks
def test_append_twice_keeps_all_rows_and_files(
    fake_hub: FakeHub, fake_bucket: str, sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same keys in both calls: before, the second call replaced
    # g=a/00000000.parquet and the rows of the first call were lost.
    _run_ids(monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    base = _uri(fake_bucket, "parts")
    first = pl.DataFrame({"g": ["a", "a", "b"], "n": [1, 2, 3]})
    second = pl.DataFrame({"g": ["a", "b", "b"], "n": [4, 5, 6]})

    sink(first, base, partition_by="g", max_rows_per_file=1, mode="append")
    after_first = _snapshot(fake_hub, fake_bucket)
    sink(second, base, partition_by="g", max_rows_per_file=1, mode="append")

    assert sorted(after_first) == [
        "parts/g=a/00000000-aaaaaaaaaaaa.parquet",
        "parts/g=a/00000001-aaaaaaaaaaaa.parquet",
        "parts/g=b/00000000-aaaaaaaaaaaa.parquet",
    ]
    after_second = _snapshot(fake_hub, fake_bucket)
    # The files of the first call are still there, byte for byte.
    for path, data in after_first.items():
        assert after_second[path] == data
    assert sorted(set(after_second) - set(after_first)) == [
        "parts/g=a/00000000-bbbbbbbbbbbb.parquet",
        "parts/g=b/00000000-bbbbbbbbbbbb.parquet",
        "parts/g=b/00000001-bbbbbbbbbbbb.parquet",
    ]
    back = plhf.scan_bucket(base).collect().sort("n")
    assert_frame_equal(back, pl.concat([first, second]))
    assert all(call.deleted == [] for call in fake_hub.batch_calls)


def test_append_run_id_is_random_per_call(fake_hub: FakeHub, fake_bucket: str) -> None:
    import re

    base = _uri(fake_bucket, "shards")
    df = pl.DataFrame({"n": [1]})

    for _ in range(3):
        sink_default(df, base, max_rows_per_file=10, mode="append")

    names = fake_hub.files(fake_bucket)
    assert len(names) == 3
    for name in names:
        assert re.fullmatch(r"shards/00000000-[0-9a-f]{12}\.parquet", name)
    assert plhf.scan_bucket(base).collect().height == 3


@pytest.mark.parametrize("fmt", ["parquet", "ndjson"])
def test_append_names_are_the_same_with_both_backends(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch, fmt: str
) -> None:
    _run_ids(monkeypatch, "0123456789ab", "0123456789ab")
    df = pl.DataFrame({"g": ["a/b", None, "c"] + ["many"] * 17, "n": range(20)})
    options = {
        "partition_by": "g",
        "max_rows_per_file": 1,
        "mode": "append",
        "format": fmt,
    }

    sink_streamed(df, _uri(fake_bucket, "streamed"), **options)
    sink_staged(df, _uri(fake_bucket, "staged"), **options)

    streamed = _relative(fake_hub.files(fake_bucket, "streamed/"), "streamed/")
    staged = _relative(fake_hub.files(fake_bucket, "staged/"), "staged/")
    assert streamed == staged
    extension = "jsonl" if fmt == "ndjson" else fmt
    assert f"g=many/00000010-0123456789ab.{extension}" in streamed
    assert f"g=a%2Fb/00000000-0123456789ab.{extension}" in streamed
    # The names differ from the native ones only by the suffix.
    native = sorted(name.replace("-0123456789ab", "") for name in streamed)
    sink_streamed(df, _uri(fake_bucket, "native"), **{**options, "mode": "error"})
    assert _relative(fake_hub.files(fake_bucket, "native/"), "native/") == native


@both_sinks
def test_append_with_an_empty_result_adds_one_empty_file(
    fake_hub: FakeHub, fake_bucket: str, sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ids(monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc")
    base = _uri(fake_bucket, "parts")
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    # A first write with no rows keeps the prefix scannable.
    sink(df.clear(), base, partition_by="g", mode="append")
    assert fake_hub.files(fake_bucket) == ["parts/00000000-aaaaaaaaaaaa.parquet"]
    assert_frame_equal(plhf.scan_bucket(base).collect(), df.clear())

    sink(df, base, partition_by="g", mode="append")
    sink(df.clear(), base, partition_by="g", mode="append")

    assert fake_hub.files(fake_bucket) == [
        "parts/00000000-aaaaaaaaaaaa.parquet",
        "parts/00000000-cccccccccccc.parquet",
        "parts/g=a/00000000-bbbbbbbbbbbb.parquet",
    ]
    assert_frame_equal(plhf.scan_bucket(base).collect(), df)


@both_sinks
def test_append_next_to_files_with_native_names(
    fake_hub: FakeHub, fake_bucket: str, sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ids(monkeypatch, "aaaaaaaaaaaa")
    base = _uri(fake_bucket, "shards")
    first = pl.DataFrame({"n": range(400)})
    sink(first, base, max_rows_per_file=100)
    before = _snapshot(fake_hub, fake_bucket)

    more = pl.DataFrame({"n": range(1000, 1200)})
    sink(more, base, max_rows_per_file=100, mode="append")

    after = _snapshot(fake_hub, fake_bucket)
    assert len(after) == 6
    for path, data in before.items():
        assert after[path] == data
    back = plhf.scan_bucket(base).collect().sort("n")
    assert_frame_equal(back, pl.concat([first, more]))


@both_sinks
def test_overwrite_mode_removes_stale_files(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # Bug o: there was no mode=, and a rewrite with fewer files left the
    # files of the previous write.
    base = _uri(fake_bucket, "shards")
    sink(pl.DataFrame({"n": range(400)}), base, max_rows_per_file=100)
    assert len(fake_hub.files(fake_bucket, "shards/")) == 4

    smaller = pl.DataFrame({"n": range(200)})
    sink(smaller, base, max_rows_per_file=100, mode="overwrite")

    assert len(fake_hub.files(fake_bucket, "shards/")) == 2
    assert_frame_equal(plhf.scan_bucket(base).collect().sort("n"), smaller)


@both_sinks
def test_overwrite_only_deletes_below_the_prefix(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # Names that share the string prefix "out" but are not in the directory.
    for path in ["out2/x.parquet", "out.parquet", "outer.txt", "other/out/y.txt"]:
        fake_hub.put(fake_bucket, path, b"keep")
    fake_hub.put(fake_bucket, "out/stale.txt", b"stale")
    fake_hub.put(fake_bucket, "out/g=old/00000000.parquet", b"stale")

    df = pl.DataFrame({"g": ["a"], "n": [1]})
    sink(df, _uri(fake_bucket, "out"), partition_by="g", mode="overwrite")

    assert fake_hub.files(fake_bucket) == [
        "other/out/y.txt",
        "out.parquet",
        "out/g=a/00000000.parquet",
        "out2/x.parquet",
        "outer.txt",
    ]


@both_sinks
def test_overwrite_of_a_single_file_deletes_nothing(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    fake_hub.put(fake_bucket, "dir/other.txt", b"keep")
    fake_hub.put(fake_bucket, "dir/out.parquet", b"old")
    df = pl.DataFrame({"a": [1]})

    sink(df, _uri(fake_bucket, "dir/out.parquet"), mode="overwrite")

    assert fake_hub.files(fake_bucket) == ["dir/other.txt", "dir/out.parquet"]
    assert_frame_equal(
        plhf.scan_bucket(_uri(fake_bucket, "dir/out.parquet")).collect(), df
    )
    assert [call.deleted for call in fake_hub.batch_calls] == [[]]


@both_sinks
def test_overwrite_with_an_empty_result_leaves_one_empty_file(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    base = _uri(fake_bucket, "parts")
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})
    sink(df, base, partition_by="g")

    sink(df.clear(), base, partition_by="g", mode="overwrite")

    assert fake_hub.files(fake_bucket) == ["parts/00000000.parquet"]
    assert_frame_equal(plhf.scan_bucket(base).collect(), df.clear())


@both_sinks
def test_overwrite_deletes_only_after_the_new_files_are_registered(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    base = _uri(fake_bucket, "shards")
    sink(pl.DataFrame({"n": range(400)}), base, max_rows_per_file=100)
    before = _snapshot(fake_hub, fake_bucket)
    fake_hub.batch_calls.clear()
    fail_batch(fake_hub, 1)

    with pytest.raises((ScriptedUploadError, BucketRegistrationError)):
        sink(pl.DataFrame({"n": [1]}), base, max_rows_per_file=100, mode="overwrite")

    # The registration failed, so no delete request was sent.
    assert _snapshot(fake_hub, fake_bucket) == before
    assert len(fake_hub.batch_calls) == 1


@both_sinks
def test_overwrite_with_a_failed_delete_keeps_new_and_stale_files(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # Documented limitation: the delete request is separate from the
    # registration.
    base = _uri(fake_bucket, "shards")
    sink(pl.DataFrame({"n": range(400)}), base, max_rows_per_file=100)
    fake_hub.batch_calls.clear()
    fail_batch(fake_hub, 2)

    with pytest.raises((ScriptedUploadError, BucketRegistrationError)):
        sink(pl.DataFrame({"n": [-1]}), base, max_rows_per_file=100, mode="overwrite")

    assert len(fake_hub.files(fake_bucket, "shards/")) == 4
    assert plhf.scan_bucket(base).collect()["n"].min() == -1


@both_sinks
def test_overwrite_deletes_are_chunked(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    for number in range(1001):
        fake_hub.put(fake_bucket, f"out/stale-{number}.txt", b"stale")

    df = pl.DataFrame({"n": [1]})
    sink(df, _uri(fake_bucket, "out"), max_rows_per_file=10, mode="overwrite")

    assert fake_hub.files(fake_bucket) == ["out/00000000.parquet"]
    deleted = [len(call.deleted) for call in fake_hub.batch_calls if call.deleted]
    assert deleted == [1000, 1]


@both_sinks
def test_error_mode_raises_if_the_file_exists(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    fake_hub.put(fake_bucket, "out.parquet", b"existing")
    before = _snapshot(fake_hub, fake_bucket)

    with pytest.raises(FileExistsError, match="already exists"):
        sink(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "out.parquet"), mode="error")

    assert _snapshot(fake_hub, fake_bucket) == before
    assert fake_hub.batch_calls == []
    assert fake_hub.commits == []


@both_sinks
def test_error_mode_raises_if_the_prefix_holds_a_file(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    fake_hub.put(fake_bucket, "parts/deep/down/file.txt", b"existing")
    before = _snapshot(fake_hub, fake_bucket)
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(FileExistsError, match="already holds files"):
        sink(df, _uri(fake_bucket, "parts"), partition_by="g", mode="error")

    assert _snapshot(fake_hub, fake_bucket) == before
    assert fake_hub.batch_calls == []
    assert fake_hub.commits == []


@both_sinks
def test_error_mode_writes_to_a_free_destination(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # Siblings that share the string prefix do not count as "existing".
    for path in ["out.parquet.bak", "out.parquetX/child.txt", "parts2/x", "parts.txt"]:
        fake_hub.put(fake_bucket, path, b"sibling")
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    sink(df, _uri(fake_bucket, "out.parquet"), mode="error")
    sink(df, _uri(fake_bucket, "parts"), partition_by="g", mode="error")

    assert "out.parquet" in fake_hub.files(fake_bucket)
    assert fake_hub.files(fake_bucket, "parts/") == ["parts/g=a/00000000.parquet"]


# ---- local disk ------------------------------------------------------------


def _directory_size(directory: str) -> int:
    total = 0
    for root, _, names in os.walk(directory):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass  # removed between the listing and the stat
    return total


def _peak_staging_during_write(
    fake_hub: FakeHub, fake_bucket: str, sink, staging: str
) -> tuple[int, int]:
    """Write 40 files of about 100 kB; return (peak staging bytes, output bytes)."""
    peak = 0
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        peak = max(peak, _directory_size(staging))

    def sample_until_stopped() -> None:
        while not stop.is_set():
            sample()
            stop.wait(0.002)

    # Sample at every batch (deterministic) and in the background.
    fake_hub.before_batch = sample
    sampler = threading.Thread(target=sample_until_stopped, daemon=True)

    rows = 200_000
    ids = pl.int_range(0, rows, eager=True)
    text = (ids * 2654435761 % 1000003).cast(pl.String) + "-" + ids.cast(pl.String)
    df = pl.DataFrame({"id": ids, "text": text})

    sampler.start()
    try:
        sink(
            df,
            _uri(fake_bucket, "many"),
            max_rows_per_file=5_000,
            compression="uncompressed",
        )
    finally:
        stop.set()
        sampler.join()

    written = fake_hub.files(fake_bucket, "many/")
    assert len(written) == 40
    return peak, sum(len(fake_hub.read(fake_bucket, path)) for path in written)


def test_streamed_write_does_not_stage_on_local_disk(
    fake_hub: FakeHub, fake_bucket: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug n: the partitioned write staged the complete output in a local temp
    # dir. The stream backend hands Polars file objects, so nothing is staged.
    # (Offline the upload stream is the in-memory stand-in; the staging test
    # test_local_disk_use measures the real hf_xet upload.)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))

    peak, _ = _peak_staging_during_write(
        fake_hub, fake_bucket, sink_streamed, str(staging)
    )

    assert peak == 0


def test_staged_write_uses_as_much_local_disk_as_the_output(
    fake_hub: FakeHub, fake_bucket: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The documented limitation of the staged backend (bug n does not apply).
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))

    peak, total = _peak_staging_during_write(
        fake_hub, fake_bucket, sink_staged, str(staging)
    )

    assert peak == total
    assert os.listdir(staging) == []  # removed after the upload


def test_staged_write_uses_the_staging_directory_of_the_environment(
    fake_hub: FakeHub, fake_bucket: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "default"
    chosen = tmp_path / "chosen"
    default.mkdir()
    chosen.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(default))
    seen: list[list[str]] = []
    fake_hub.before_batch = lambda: seen.append(
        [os.listdir(default), os.listdir(chosen)]
    )
    monkeypatch.setenv("POLARS_HF_STAGING_DIR", str(chosen))

    sink_staged(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "a.parquet"))
    sink_staged(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "parts"), partition_by="a")

    assert len(seen) == 2
    for in_default, in_chosen in seen:
        assert in_default == []
        assert len(in_chosen) == 1
    assert os.listdir(chosen) == []


def test_staging_directory_is_removed_after_a_failed_sink(
    fake_hub: FakeHub, fake_bucket: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLARS_HF_STAGING_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="scripted sink failure"):
        sink_staged(_failing_frame(), _uri(fake_bucket, "parts"), partition_by="g")

    assert os.listdir(tmp_path) == []


def test_default_sink_writes(fake_hub: FakeHub, fake_bucket: str) -> None:
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})

    sink_default(df, _uri(fake_bucket, "parts"), partition_by="g")

    assert len(fake_hub.files(fake_bucket, "parts/")) == 2


def test_staging_dir_argument_is_not_accepted(
    fake_hub: FakeHub, fake_bucket: str, tmp_path
) -> None:
    # Not a parameter of sink_bucket: it reaches the Polars sink, which
    # refuses it. Nothing is written.
    with pytest.raises(TypeError, match="staging_dir"):
        sink_staged(
            pl.DataFrame({"a": [1]}),
            _uri(fake_bucket, "a.parquet"),
            staging_dir=tmp_path,
        )

    assert fake_hub.files(fake_bucket) == []


# ---- review round 1 --------------------------------------------------------


@both_sinks
@pytest.mark.parametrize("suffix", ["", "/", "//"])
def test_overwrite_of_the_bucket_root_is_refused(
    fake_hub: FakeHub, fake_bucket: str, sink, suffix: str
) -> None:
    fake_hub.put(fake_bucket, "precious.txt", b"keep")
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(ValueError):
        sink(
            df,
            f"hf://buckets/{fake_bucket}{suffix}",
            partition_by="g",
            mode="overwrite",
        )

    assert fake_hub.files(fake_bucket) == ["precious.txt"]
    assert fake_hub.requests == []


def test_overwrite_of_the_bucket_root_names_the_reason(fake_bucket: str) -> None:
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(ValueError, match="needs a directory below the bucket root"):
        plhf.sink_bucket(
            df, f"hf://buckets/{fake_bucket}/", partition_by="g", mode="overwrite"
        )


@both_sinks
@pytest.mark.parametrize(
    ("path", "message"),
    [
        # Refused by parse_bucket_uri, for reads and writes alike.
        ("/", "an empty path segment"),
        ("//", "an empty path segment"),
        ("a//b", "an empty path segment"),
        ("/a", "an empty path segment"),
        ("a/../b", "a '..' path segment"),
        # Refused by the write path.
        ("./a", "the path segment '.'"),
        ("a\\b", "a backslash"),
    ],
)
@pytest.mark.parametrize("mode", ["append", "overwrite", "error"])
def test_invalid_prefix_is_rejected_before_any_request(
    fake_hub: FakeHub, fake_bucket: str, sink, path: str, message: str, mode: str
) -> None:
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(ValueError, match=message):
        sink(df, f"hf://buckets/{fake_bucket}/{path}", partition_by="g", mode=mode)

    assert fake_hub.requests == []
    assert fake_hub.commits == []


@both_sinks
@pytest.mark.parametrize("path", ["a//b.parquet", "../b.parquet", "a\\b.parquet"])
def test_invalid_file_path_is_rejected_before_any_request(
    fake_hub: FakeHub, fake_bucket: str, sink, path: str
) -> None:
    # '//' and '..' are refused by parse_bucket_uri, the backslash by the
    # write path.
    with pytest.raises(ValueError, match="path segment"):
        sink(pl.DataFrame({"a": [1]}), f"hf://buckets/{fake_bucket}/{path}")

    assert fake_hub.requests == []
    assert fake_hub.commits == []


@both_sinks
def test_partition_value_the_hub_refuses_is_rejected_before_registration(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # A backslash is not percent-encoded by the hive layout and the Hub
    # refuses it. Without the check the bucket would apply the other files of
    # the request and reject this one.
    df = pl.DataFrame({"g": ["ok", "a\\b", "fine"], "n": [1, 2, 3]})

    with pytest.raises(ValueError) as error:
        sink(df, _uri(fake_bucket, "parts"), partition_by="g")

    assert "'g=a\\\\b'" in str(error.value)
    assert fake_hub.files(fake_bucket) == []
    assert fake_hub.batch_calls == []
    # The stream backend may have opened streams for other partitions: the
    # commit is aborted, so nothing is stored.
    assert all(commit.aborted for commit in fake_hub.commits)


@both_sinks
def test_overwrite_keeps_files_added_by_another_writer(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    base = _uri(fake_bucket, "shards")
    sink(pl.DataFrame({"n": range(400)}), base, max_rows_per_file=100)
    fake_hub.batch_calls.clear()

    def other_writer() -> None:
        # Runs when the first batch of the overwrite arrives: after the
        # listing, before the deletion.
        if len(fake_hub.batch_calls) == 0:
            fake_hub.put(fake_bucket, "shards/from-another-writer.parquet", b"other")

    fake_hub.before_batch = other_writer

    sink(pl.DataFrame({"n": [1]}), base, max_rows_per_file=100, mode="overwrite")

    assert fake_hub.files(fake_bucket) == [
        "shards/00000000.parquet",
        "shards/from-another-writer.parquet",
    ]


@both_sinks
def test_error_mode_raises_if_a_file_is_at_the_prefix(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    fake_hub.put(fake_bucket, "parts", b"a file, not a directory")
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(FileExistsError, match="'parts' is a file in the bucket"):
        sink(df, _uri(fake_bucket, "parts"), partition_by="g", mode="error")

    assert fake_hub.files(fake_bucket) == ["parts"]
    assert fake_hub.batch_calls == []


@both_sinks
def test_format_is_case_insensitive(fake_hub: FakeHub, fake_bucket: str, sink) -> None:
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    sink(df, _uri(fake_bucket, "out.data"), format="Parquet")
    sink(df, _uri(fake_bucket, "parts"), partition_by="g", format="NDJSON")

    assert fake_hub.files(fake_bucket) == ["out.data", "parts/g=a/00000000.jsonl"]
    with pytest.raises(ValueError, match="unsupported format 'Excel'"):
        sink(df, _uri(fake_bucket, "x"), format="Excel")


def test_write_error_is_raised_even_if_the_sink_swallows_it(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The backend does not rely on Polars to raise for a failed write().
    backend = _sinks.StreamBackend(fake_bucket, token=None)
    fake_hub.fail_stream_write_on_call = 2

    def sink_that_hides_errors(target) -> None:
        target.write(b"first")
        try:
            target.write(b"second")
        except ScriptedUploadError:
            pass

    with pytest.raises(ScriptedUploadError):
        backend.write_file(sink_that_hides_errors, "out.bin")

    assert fake_hub.files(fake_bucket) == []
    assert fake_hub.batch_calls == []
    assert [commit.aborted for commit in fake_hub.commits] == [True]


@both_sinks
def test_missing_bucket_raises_the_registration_error(fake_hub: FakeHub, sink) -> None:
    df = pl.DataFrame({"a": [1]})

    # The existence check is the first request to the missing bucket.
    for mode in ("error", "overwrite"):
        with pytest.raises(FileNotFoundError, match="fake-user/no-such-bucket"):
            sink(df, "hf://buckets/fake-user/no-such-bucket/a.parquet", mode=mode)

    assert fake_hub.batch_calls == []
    assert fake_hub.commits == []


def test_registration_in_a_missing_bucket_raises_the_registration_error(
    fake_hub: FakeHub,
) -> None:
    # The backends themselves: without the checks of sink_bucket in front.
    for backend_class in (_sinks.StreamBackend, _sinks.StagedBackend):
        backend = backend_class("fake-user/no-such-bucket", token=None)

        with pytest.raises(BucketRegistrationError) as error:
            backend.delete(["a.parquet"])

        assert isinstance(error.value.__cause__, HfHubHTTPError)
        assert error.value.__cause__.response.status_code == 404
        assert error.value.failures == []


def test_registration_reply_that_is_not_json_is_an_error(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # For example a proxy that answers 200 with an HTML page.
    fake_hub.add_fault(HUB, "POST", r"/batch$", 200, body=b"<html>ok</html>")

    with pytest.raises(BucketRegistrationError, match="not the expected confirmation"):
        sink_streamed(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "a.parquet"))

    assert fake_hub.files(fake_bucket) == []


@pytest.mark.parametrize("status", [429, 503])
def test_transient_registration_error_is_retried(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    import huggingface_hub.utils._http

    df = pl.DataFrame({"a": [1, 2, 3]})
    uri = _uri(fake_bucket, "a.parquet")
    fake_hub.add_fault(HUB, "POST", r"/batch$", status, headers={"Retry-After": "0"})
    # http_backoff waits between two attempts: record the wait, do not sleep.
    waits: list[float] = []
    monkeypatch.setattr(huggingface_hub.utils._http.time, "sleep", waits.append)

    sink_streamed(df, uri)

    assert len(waits) == 1

    posts = fake_hub.matching(origin=HUB, method="POST", path_contains="/batch")
    assert [request.status for request in posts] == [status, 200]
    assert_frame_equal(plhf.scan_bucket(uri).collect(), df)
    # One upload, registered once.
    assert len(fake_hub.commits) == 1
    assert len(fake_hub.batch_calls) == 1


def _assert_incompatible(error: pytest.ExceptionInfo) -> None:
    import huggingface_hub

    message = str(error.value)
    assert "not compatible with the installed" in message
    assert f"huggingface_hub {huggingface_hub.__version__}" in message
    assert "hf_xet " in message
    assert "backend='staged'" in message
    assert isinstance(error.value.__cause__, TypeError)


def test_changed_xet_commit_signature_is_reported(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def changed(endpoint: str, bucket_id: str, headers: dict) -> None:
        raise TypeError("new_upload_commit() got an unexpected keyword argument")

    monkeypatch.setattr(_sinks, "open_xet_commit", changed)

    with pytest.raises(RuntimeError) as error:
        sink_streamed(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "a.parquet"))

    _assert_incompatible(error)
    assert fake_hub.files(fake_bucket) == []


# ---- review round 2 --------------------------------------------------------


@both_sinks
@pytest.mark.parametrize(
    ("frame", "key", "message"),
    [
        (
            pl.DataFrame({"g": ["ok", "x" * 300], "n": [1, 2]}),
            "g",
            "a path segment of more than 255 bytes",
        ),
        (
            pl.DataFrame({"a\nb": ["v"], "n": [1]}),
            "a\nb",
            "a control character in the path segment 'a\\nb=v'",
        ),
        (
            pl.DataFrame({"a\x00b": ["v"], "n": [1]}),
            "a\x00b",
            "a control character in the path segment",
        ),
    ],
)
def test_partition_names_no_backend_can_write_are_rejected(
    fake_hub: FakeHub, fake_bucket: str, sink, frame: pl.DataFrame, key: str, message
) -> None:
    # The same error from both backends, raised when Polars asks for the file.
    with pytest.raises(ValueError) as error:
        sink(frame, _uri(fake_bucket, "parts"), partition_by=key)

    assert message in str(error.value)
    assert fake_hub.files(fake_bucket) == []
    assert fake_hub.batch_calls == []
    assert all(commit.aborted for commit in fake_hub.commits)


@both_sinks
@pytest.mark.parametrize("path", ["a\tb/c", "x" * 256, "nul\x00/c"])
def test_destination_no_backend_can_write_is_rejected(
    fake_hub: FakeHub, fake_bucket: str, sink, path: str
) -> None:
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(ValueError, match="invalid bucket path"):
        sink(df, f"hf://buckets/{fake_bucket}/{path}", partition_by="g")
    with pytest.raises(ValueError, match="invalid bucket path"):
        sink(df, f"hf://buckets/{fake_bucket}/{path}.parquet")

    assert fake_hub.requests == []


def test_segment_of_255_bytes_is_accepted(fake_hub: FakeHub, fake_bucket: str) -> None:
    value = "x" * (255 - len("g="))
    df = pl.DataFrame({"g": [value], "n": [1]})

    for name, sink in (("stream", sink_streamed), ("staged", sink_staged)):
        sink(df, _uri(fake_bucket, name), partition_by="g")

    assert fake_hub.files(fake_bucket) == [
        f"staged/g={value}/00000000.parquet",
        f"stream/g={value}/00000000.parquet",
    ]


def test_same_output_file_asked_twice_is_refused() -> None:
    from types import SimpleNamespace

    spec = _sinks.PartitionSpec(
        key="g", max_rows_per_file=None, max_bytes_per_file=None, extension="parquet"
    )
    destinations = _sinks._Destinations("parts", spec)
    args = SimpleNamespace(
        partition_keys=pl.DataFrame({"g": ["a"]}), index_in_partition=0
    )

    assert destinations.claim(args) == (
        "g=a/00000000.parquet",
        "parts/g=a/00000000.parquet",
    )
    with pytest.raises(RuntimeError, match="asked twice for the output file"):
        destinations.claim(args)
    assert destinations.paths == ["parts/g=a/00000000.parquet"]


@both_sinks
def test_overwrite_delete_is_a_checked_request(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    # Both backends delete through the package's own /batch request, whose
    # answer is checked with every supported huggingface_hub.
    fake_hub.put(fake_bucket, "out/stale.txt", b"stale")
    df = pl.DataFrame({"n": [1]})

    sink(df, _uri(fake_bucket, "out"), max_rows_per_file=10, mode="overwrite")

    deletes = [call for call in fake_hub.batch_calls if call.deleted]
    assert [(call.via, call.deleted) for call in deletes] == [
        ("http", ["out/stale.txt"])
    ]


# -- staged backend with a client that does not report rejected files (hub 1.x) --


def test_hub_backend_detects_a_file_the_client_dropped_silently(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_sinks, "hub_reports_rejected_files", lambda: False)
    fake_hub.drop_paths = {"parts/g=b/00000000.parquet"}
    df = pl.DataFrame({"g": ["a", "b", "c"], "n": [1, 2, 3]})

    with pytest.raises(BucketRegistrationError) as error:
        sink_staged(df, _uri(fake_bucket, "parts"), partition_by="g")

    assert error.value.failures == [
        {
            "path": "parts/g=b/00000000.parquet",
            "error": "not in the bucket after the upload",
        }
    ]
    assert "rejected 1 of 3 operation(s)" in str(error.value)


def test_hub_backend_detects_a_dropped_single_file(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_sinks, "hub_reports_rejected_files", lambda: False)
    # A sibling with the same string prefix must not hide the missing file.
    fake_hub.put(fake_bucket, "out.parquet.bak", b"sibling")
    fake_hub.drop_paths = {"out.parquet"}

    with pytest.raises(BucketRegistrationError, match="not in the bucket"):
        sink_staged(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "out.parquet"))


def test_hub_backend_verification_passes_and_costs_one_listing(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})
    listings = {}
    for reports in (True, False):
        monkeypatch.setattr(_sinks, "hub_reports_rejected_files", lambda r=reports: r)
        fake_hub.reset_log()
        sink_staged(df, _uri(fake_bucket, f"parts-{reports}"), partition_by="g")
        listings[reports] = len(
            fake_hub.matching(origin=HUB, method="GET", path_contains="/tree")
        )

    # One listing for the default mode="error", one more for the check.
    assert listings == {True: 1, False: 2}


def test_verification_reports_a_wrong_size(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put(fake_bucket, "dir/a.bin", b"12345")
    fake_hub.put(fake_bucket, "dir/b.bin", b"123")
    backend = _sinks.StagedBackend(fake_bucket, token=None)

    backend._verify_added({"dir/a.bin": 5, "dir/b.bin": 3}, "dir")
    with pytest.raises(BucketRegistrationError) as error:
        backend._verify_added({"dir/a.bin": 5, "dir/b.bin": 4, "dir/c.bin": 1}, "dir")

    assert error.value.failures == [
        {"path": "dir/b.bin", "error": "has 3 bytes in the bucket, expected 4"},
        {"path": "dir/c.bin", "error": "not in the bucket after the upload"},
    ]


def test_installed_hub_reports_rejected_files_from_2_0() -> None:
    import huggingface_hub

    major = int(huggingface_hub.__version__.split(".")[0])

    assert _sinks.hub_reports_rejected_files() == (major >= 2)


# -- network failures --------------------------------------------------------


def _assert_unknown_state(error: pytest.ExceptionInfo, cause: type) -> None:
    assert isinstance(error.value.__cause__, cause)
    message = str(error.value)
    assert "state of the destination is unknown" in message
    assert "hf buckets ls" in message


def test_network_error_of_the_registration_request_is_wrapped(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    def no_answer(*args: object, **kwargs: object) -> None:
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(_sinks, "http_backoff", no_answer)

    with pytest.raises(BucketRegistrationError) as error:
        sink_streamed(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "a.parquet"))

    _assert_unknown_state(error, httpx.ReadTimeout)


@pytest.mark.parametrize("backend_sink", ALL_SINKS)
def test_network_error_of_the_delete_request_is_wrapped(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch, backend_sink
) -> None:
    fake_hub.put(fake_bucket, "out/stale.txt", b"stale")
    real = _sinks.http_backoff

    def no_answer_for_deletes(method: str, url: str, **kwargs: object):
        if b"deleteFile" in kwargs["content"]:
            raise ConnectionResetError("connection reset")
        return real(method, url, **kwargs)

    monkeypatch.setattr(_sinks, "http_backoff", no_answer_for_deletes)
    df = pl.DataFrame({"n": [1]})

    with pytest.raises(BucketRegistrationError) as error:
        backend_sink(
            df, _uri(fake_bucket, "out"), max_rows_per_file=10, mode="overwrite"
        )

    _assert_unknown_state(error, ConnectionResetError)
    assert fake_hub.files(fake_bucket) == ["out/00000000.parquet", "out/stale.txt"]


def test_network_error_of_the_hub_upload_is_wrapped(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from huggingface_hub import HfApi

    def no_answer(self: HfApi, bucket_id: str, **kwargs: object) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(HfApi, "batch_bucket_files", no_answer)

    with pytest.raises(BucketRegistrationError) as error:
        sink_staged(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "a.parquet"))

    _assert_unknown_state(error, httpx.ConnectError)


# ---- review round 3 --------------------------------------------------------


@both_sinks
@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        b'{"success": true}',
        b'{"success": true, "processed": 1, "succeeded": 1}',
        b'{"success": false, "processed": 1, "succeeded": 0, "failed": []}',
        b'{"success": true, "processed": 0, "succeeded": 0, "failed": []}',
        b"[]",
        b"null",
    ],
)
def test_registration_answer_that_does_not_confirm_is_an_error(
    fake_hub: FakeHub, fake_bucket: str, sink, body: bytes
) -> None:
    # Both backends send the overwrite delete through the checked request; a
    # 200 that does not confirm the operation is not a success.
    fake_hub.put(fake_bucket, "out/stale.txt", b"stale")
    headers = {"Content-Type": "application/json"}
    fake_hub.add_fault(HUB, "POST", r"/batch$", 200, headers=headers, body=body)
    df = pl.DataFrame({"n": [1]})

    with pytest.raises(BucketRegistrationError) as error:
        if sink is sink_staged:
            # The first /batch request of the staged backend is the delete.
            sink(df, _uri(fake_bucket, "out"), max_rows_per_file=10, mode="overwrite")
        else:
            sink(df, _uri(fake_bucket, "out/a.parquet"))

    assert "not the expected confirmation of 1 operation(s)" in str(error.value)
    assert "state of the destination is unknown" in str(error.value)


def test_changed_http_helper_is_not_reported_as_an_xet_problem(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # http_backoff is public API and serves both backends: its errors are not
    # relabelled as "the stream backend is not compatible".
    fake_hub.put(fake_bucket, "out/stale.txt", b"stale")

    def changed(*args: object, **kwargs: object) -> None:
        raise TypeError("http_backoff() got an unexpected keyword argument")

    monkeypatch.setattr(_sinks, "http_backoff", changed)
    df = pl.DataFrame({"n": [1]})

    with pytest.raises(TypeError, match="http_backoff"):
        sink_staged(
            df, _uri(fake_bucket, "out"), max_rows_per_file=10, mode="overwrite"
        )


# ---- review round 4 --------------------------------------------------------


def _checks(fake_hub: FakeHub) -> list[tuple[str, str]]:
    """``(method, target)`` of the resolve and listing requests, as sent.

    The target is the percent-encoded end of the request path:
    ``"resolve/<path>"`` or ``"tree/<prefix>"`` (``"tree"`` for the bucket
    root).
    """
    targets = []
    for request in fake_hub.requests:
        if "/resolve/" in request.raw_path:
            path = request.raw_path.split("/resolve/", 1)[1]
            targets.append((request.method, f"resolve/{path}"))
        elif "/tree" in request.raw_path:
            prefix = request.raw_path.split("/tree", 1)[1]
            targets.append((request.method, f"tree{prefix}"))
    return targets


@both_sinks
def test_existence_checks_cost_a_bounded_number_of_requests(
    fake_hub: FakeHub, fake_bucket: str, sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Many siblings that share the string prefix "out" ("out2/...",
    # "out.parquet.NNNN"), a large destination "big/", and a listing that
    # returns 100 entries per page. The listing must name the directory with
    # its trailing slash ("out%2F" as the client encodes it) and the HEAD the
    # exact path: without the slash, the siblings would count as existing.
    monkeypatch.setattr(_sinks, "hub_reports_rejected_files", lambda: True)
    fake_hub.tree_page_size = 100
    for number in range(1500):
        fake_hub.put(fake_bucket, f"out2/{number:04}.txt", b"x")
        fake_hub.put(fake_bucket, f"out.parquet.{number:04}", b"x")
        fake_hub.put(fake_bucket, f"big/{number:04}.txt", b"x")
    fake_hub.put(fake_bucket, "out.parquet2/x.txt", b"x")
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    # Directory, mode="error", free: one HEAD and one listing page.
    fake_hub.reset_log()
    sink(df, _uri(fake_bucket, "out"), partition_by="g")
    assert _checks(fake_hub) == [("HEAD", "resolve/out"), ("GET", "tree/out%2F")]

    # Directory, mode="error", 1,500 files there: the same two requests.
    fake_hub.reset_log()
    with pytest.raises(FileExistsError):
        sink(df, _uri(fake_bucket, "big"), partition_by="g")
    assert _checks(fake_hub) == [("HEAD", "resolve/big"), ("GET", "tree/big%2F")]

    # Directory, mode="append": one HEAD, no listing, whatever is there.
    fake_hub.reset_log()
    sink(df, _uri(fake_bucket, "big"), partition_by="g", mode="append")
    assert _checks(fake_hub) == [("HEAD", "resolve/big")]

    # A nested directory: the whole prefix is one encoded segment.
    fake_hub.reset_log()
    sink(df, _uri(fake_bucket, "deep/er/out"), partition_by="g")
    assert _checks(fake_hub) == [
        ("HEAD", "resolve/deep%2Fer%2Fout"),
        ("GET", "tree/deep%2Fer%2Fout%2F"),
    ]

    # Bucket root, mode="error": one listing page.
    fake_hub.reset_log()
    with pytest.raises(FileExistsError):
        sink(df, f"hf://buckets/{fake_bucket}", partition_by="g")
    assert _checks(fake_hub) == [("GET", "tree")]

    # Single file, mode="error": one HEAD and one listing page of "path/".
    fake_hub.reset_log()
    sink(df, _uri(fake_bucket, "out.parquet"))
    assert _checks(fake_hub) == [
        ("HEAD", "resolve/out.parquet"),
        ("GET", "tree/out.parquet%2F"),
    ]

    # Single file, mode="overwrite": one listing page of "path/".
    fake_hub.reset_log()
    sink(df, _uri(fake_bucket, "out.parquet"), mode="overwrite")
    assert _checks(fake_hub) == [("GET", "tree/out.parquet%2F")]


@both_sinks
def test_overwrite_lists_every_page_of_the_destination(
    fake_hub: FakeHub, fake_bucket: str, sink
) -> None:
    fake_hub.tree_page_size = 100
    for number in range(250):
        fake_hub.put(fake_bucket, f"out/{number:04}.txt", b"stale")
    df = pl.DataFrame({"n": [1]})

    sink(df, _uri(fake_bucket, "out"), max_rows_per_file=10, mode="overwrite")

    assert fake_hub.files(fake_bucket) == ["out/00000000.parquet"]


@both_sinks
@pytest.mark.parametrize("mode", ["error", "overwrite"])
def test_file_write_over_a_directory_is_refused(
    fake_hub: FakeHub, fake_bucket: str, sink, mode: str
) -> None:
    fake_hub.put(fake_bucket, "out.parquet/child.txt", b"in a directory")
    before = _snapshot(fake_hub, fake_bucket)

    with pytest.raises(FileExistsError) as error:
        sink(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "out.parquet"), mode=mode)

    message = str(error.value)
    assert "cannot write the file 'out.parquet'" in message
    assert "'out.parquet/' is a directory" in message
    assert "'out.parquet/child.txt'" in message
    assert _snapshot(fake_hub, fake_bucket) == before
    assert fake_hub.batch_calls == []
    assert fake_hub.commits == []


@both_sinks
@pytest.mark.parametrize("mode", ["error", "append", "overwrite"])
def test_directory_write_below_a_file_is_refused(
    fake_hub: FakeHub, fake_bucket: str, sink, mode: str
) -> None:
    fake_hub.put(fake_bucket, "out", b"a file")
    before = _snapshot(fake_hub, fake_bucket)
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(FileExistsError) as error:
        sink(df, _uri(fake_bucket, "out"), partition_by="g", mode=mode)

    message = str(error.value)
    assert "cannot write below 'out/'" in message
    assert "'out' is a file in the bucket" in message
    assert _snapshot(fake_hub, fake_bucket) == before
    assert fake_hub.batch_calls == []
    assert fake_hub.commits == []


def test_append_does_not_list_the_destination(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_sinks, "hub_reports_rejected_files", lambda: True)
    base = _uri(fake_bucket, "parts")
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    for sink in ALL_SINKS:
        sink(df, base, partition_by="g", mode="append")

    assert fake_hub.matching(origin=HUB, method="GET", path_contains="/tree") == []
    assert len(fake_hub.files(fake_bucket)) == 2


def test_messages_of_a_write_name_sink_bucket(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    # The listing and its retry limits are shared with scan_bucket.
    headers = {"Retry-After": "99999"}
    fake_hub.add_fault(HUB, "GET", r"/tree", 429, headers=headers)
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(HfHubHTTPError) as error:
        plhf.sink_bucket(df, f"hf://buckets/{fake_bucket}", partition_by="g")

    assert "allowed for one sink_bucket call" in str(error.value)
    assert "scan_bucket" not in str(error.value)


@pytest.mark.parametrize("status", [403, 503])
def test_failed_verification_listing_is_an_unknown_state(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    # huggingface_hub 1.x: the staged backend lists the destination after the
    # upload. If that listing fails, the files may be registered.
    monkeypatch.setattr(_sinks, "hub_reports_rejected_files", lambda: False)
    fake_hub.add_fault(HUB, "GET", r"/tree", status, times=20)
    df = pl.DataFrame({"g": ["a"], "n": [1]})

    with pytest.raises(BucketRegistrationError) as error:
        # mode="append": no listing before the write.
        sink_staged(df, _uri(fake_bucket, "parts"), partition_by="g", mode="append")

    expected = PermissionError if status == 403 else HfHubHTTPError
    assert isinstance(error.value.__cause__, expected)
    assert "The files may be registered" in str(error.value)
    assert "state of the destination is unknown" in str(error.value)
    assert len(fake_hub.files(fake_bucket, "parts/")) == 1


@pytest.mark.parametrize(
    "local_error", [PermissionError("staged file"), OSError(28, "No space left")]
)
def test_local_error_of_the_staged_upload_is_not_a_network_error(
    fake_hub: FakeHub,
    fake_bucket: str,
    monkeypatch: pytest.MonkeyPatch,
    local_error: OSError,
) -> None:
    from huggingface_hub import HfApi

    def fails(self: HfApi, bucket_id: str, **kwargs: object) -> None:
        raise local_error

    monkeypatch.setattr(HfApi, "batch_bucket_files", fails)

    with pytest.raises(OSError) as error:
        sink_staged(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "a.parquet"))

    assert error.value is local_error


def test_timeout_of_the_staged_upload_is_wrapped(
    fake_hub: FakeHub, fake_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import HfApi

    def fails(self: HfApi, bucket_id: str, **kwargs: object) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr(HfApi, "batch_bucket_files", fails)

    with pytest.raises(BucketRegistrationError) as error:
        sink_staged(pl.DataFrame({"a": [1]}), _uri(fake_bucket, "a.parquet"))

    _assert_unknown_state(error, TimeoutError)
