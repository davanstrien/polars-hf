"""Staging tests of the sink backends: the real uploads, same scenarios for both.

Selected with ``pytest -m staging``. Each test writes to its own new bucket
(the ``staging_bucket`` fixture). The ``xet`` parameter is skipped when the
installed ``huggingface_hub`` cannot run that backend (see ``sinks.py``).
"""

from __future__ import annotations

import os
import threading

import polars as pl
import pytest
from conftest import STAGING_ENDPOINT, STAGING_TOKEN
from huggingface_hub import HfApi, HfFileSystem
from polars.testing import assert_frame_equal
from sinks import sink_default, sink_staged, sink_streamed, staging_sinks

import polars_hf as plhf
from polars_hf import _sinks

pytestmark = pytest.mark.staging

both_sinks = pytest.mark.parametrize("sink", staging_sinks())


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _files(api: HfApi, bucket_id: str, prefix: str = "") -> dict[str, tuple[int, str]]:
    """``{path: (size, xet hash)}`` of the files whose path starts with ``prefix``."""
    files = {}
    for item in api.list_bucket_tree(bucket_id, prefix=prefix or None, recursive=True):
        if item.type == "file":
            files[item.path] = (item.size, item.xet_hash)
    return files


def _read_bytes(bucket_id: str, path: str) -> bytes:
    fs = HfFileSystem(endpoint=STAGING_ENDPOINT, token=STAGING_TOKEN)
    with fs.open(f"buckets/{bucket_id}/{path}", "rb") as f:
        return f.read()


def _failing_frame() -> pl.LazyFrame:
    """A LazyFrame that writes some row groups, then raises ``RuntimeError``."""

    def explode(batch: pl.DataFrame) -> pl.DataFrame:
        if batch["n"].max() >= 150_000:
            raise RuntimeError("scripted sink failure")
        return batch

    frame = pl.LazyFrame({"n": range(200_000)}).with_columns(g=pl.col("n") % 3)
    return frame.map_batches(explode, streamable=True)


def test_default_backend_on_staging() -> None:
    # Nothing is patched here: the default is the xet backend exactly when the
    # installed packages can run it.
    expected = "xet" if _sinks.xet_unavailable_reason() is None else "hub"

    assert _sinks.resolve_backend_name(None) == expected


@both_sinks
@pytest.mark.parametrize("ext", ["parquet", "csv", "ipc", "ndjson"])
def test_single_file_round_trip(staging_bucket: str, sink, ext: str) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})
    readers = {
        "parquet": pl.read_parquet,
        "csv": pl.read_csv,
        "ipc": pl.read_ipc,
        "ndjson": pl.read_ndjson,
    }

    sink(df.lazy(), _uri(staging_bucket, f"single/out.{ext}"))

    data = _read_bytes(staging_bucket, f"single/out.{ext}")
    assert_frame_equal(readers[ext](data), df)


@both_sinks
def test_failed_sink_leaves_existing_object_byte_identical(
    staging_api: HfApi, staging_bucket: str, sink
) -> None:
    uri = _uri(staging_bucket, "keep/data.parquet")
    sink(pl.DataFrame({"n": range(1000)}), uri)
    files_before = _files(staging_api, staging_bucket)
    bytes_before = _read_bytes(staging_bucket, "keep/data.parquet")
    assert len(bytes_before) > 0

    with pytest.raises(RuntimeError, match="scripted sink failure"):
        sink(_failing_frame(), uri, row_group_size=10_000)

    assert _files(staging_api, staging_bucket) == files_before
    assert _read_bytes(staging_bucket, "keep/data.parquet") == bytes_before


@both_sinks
def test_failed_partitioned_sink_writes_nothing(
    staging_api: HfApi, staging_bucket: str, sink
) -> None:
    base = _uri(staging_bucket, "parts")
    sink(pl.DataFrame({"g": [0, 7], "n": [1, 2]}), base, partition_by="g")
    files_before = _files(staging_api, staging_bucket)

    with pytest.raises(RuntimeError, match="scripted sink failure"):
        sink(_failing_frame(), base, partition_by="g", mode="overwrite")

    assert _files(staging_api, staging_bucket) == files_before


@both_sinks
def test_partitioned_layout_is_native(
    staging_api: HfApi, staging_bucket: str, sink
) -> None:
    groups = ["a/b", "x=y", "a b", "é:%", None] + ["many"] * 12
    flags = [True, False, None, True, None] + [True] * 12
    df = pl.DataFrame({"g": groups, "b": flags, "n": range(17)})
    base = _uri(staging_bucket, "layout")

    sink(df, base, partition_by=["g", "b"], max_rows_per_file=1)

    null = "__HIVE_DEFAULT_PARTITION__"
    expected = [
        "layout/g=%C3%A9%3A%25/b=true/00000000.parquet",
        f"layout/g={null}/b={null}/00000000.parquet",
        "layout/g=a%20b/b=__HIVE_DEFAULT_PARTITION__/00000000.parquet",
        "layout/g=a%2Fb/b=true/00000000.parquet",
    ]
    for index in range(12):
        expected.append(f"layout/g=many/b=true/{index:08x}.parquet")
    expected.append("layout/g=x%3Dy/b=false/00000000.parquet")
    assert sorted(_files(staging_api, staging_bucket)) == sorted(expected)
    back = plhf.scan_bucket(base).collect().sort("n")
    assert_frame_equal(back, df)


@both_sinks
def test_overwrite_removes_stale_files(
    staging_api: HfApi, staging_bucket: str, sink
) -> None:
    base = _uri(staging_bucket, "shards")
    sink(pl.DataFrame({"n": range(400)}), base, max_rows_per_file=100)
    sink(pl.DataFrame({"n": [0]}), _uri(staging_bucket, "shards2/keep.parquet"))
    assert len(_files(staging_api, staging_bucket, "shards/")) == 4

    smaller = pl.DataFrame({"n": range(200)})
    sink(smaller, base, max_rows_per_file=100, mode="overwrite")

    assert sorted(_files(staging_api, staging_bucket)) == [
        "shards/00000000.parquet",
        "shards/00000001.parquet",
        "shards2/keep.parquet",
    ]
    assert_frame_equal(plhf.scan_bucket(base).collect().sort("n"), smaller)


@both_sinks
def test_append_keeps_existing_files(
    staging_api: HfApi, staging_bucket: str, sink
) -> None:
    base = _uri(staging_bucket, "shards")
    sink(pl.DataFrame({"n": range(400)}), base, max_rows_per_file=100)

    sink(pl.DataFrame({"n": range(1000, 1100)}), base, max_rows_per_file=100)

    assert len(_files(staging_api, staging_bucket, "shards/")) == 4
    back = plhf.scan_bucket(base).collect().sort("n")
    assert back["n"].to_list() == [*range(100, 400), *range(1000, 1100)]


@both_sinks
def test_error_mode(staging_api: HfApi, staging_bucket: str, sink) -> None:
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})
    file_uri = _uri(staging_bucket, "single/out.parquet")
    base = _uri(staging_bucket, "parts")

    # A free destination is written.
    sink(df, file_uri, mode="error")
    sink(df, base, partition_by="g", mode="error")
    files_before = _files(staging_api, staging_bucket)
    assert len(files_before) == 3

    with pytest.raises(FileExistsError):
        sink(df, file_uri, mode="error")
    with pytest.raises(FileExistsError):
        sink(df, base, partition_by="g", mode="error")

    assert _files(staging_api, staging_bucket) == files_before


@both_sinks
def test_empty_partitioned_write_scans_back(
    staging_api: HfApi, staging_bucket: str, sink
) -> None:
    df = pl.DataFrame({"g": [], "n": []}, schema={"g": pl.String, "n": pl.Int64})
    base = _uri(staging_bucket, "empty")

    sink(df, base, partition_by="g")

    assert sorted(_files(staging_api, staging_bucket)) == ["empty/00000000.parquet"]
    assert_frame_equal(plhf.scan_bucket(base).collect(), df)


@both_sinks
def test_empty_file(staging_api: HfApi, staging_bucket: str, sink) -> None:
    # An empty ndjson output is a 0-byte object.
    df = pl.DataFrame({"a": []}, schema={"a": pl.Int64})

    sink(df, _uri(staging_bucket, "empty.jsonl"))

    files = _files(staging_api, staging_bucket)
    assert list(files) == ["empty.jsonl"]
    assert files["empty.jsonl"][0] == 0


@both_sinks
def test_write_of_more_than_one_batch(
    staging_api: HfApi, staging_bucket: str, sink
) -> None:
    # 1,001 files: more than the 1,000 operations of one /batch request.
    df = pl.DataFrame({"n": range(1001)})
    base = _uri(staging_bucket, "many")

    sink(df, base, max_rows_per_file=1)

    files = _files(staging_api, staging_bucket, "many/")
    assert len(files) == 1001
    assert "many/000003e8.parquet" in files
    back = plhf.scan_bucket(_uri(staging_bucket, "many/000003e8.parquet")).collect()
    assert back.height == 1

    # mode="overwrite" then deletes 1,000 stale files.
    sink(pl.DataFrame({"n": [1]}), base, max_rows_per_file=1, mode="overwrite")

    assert sorted(_files(staging_api, staging_bucket)) == ["many/00000000.parquet"]


def test_backends_write_identical_names_and_data(
    staging_api: HfApi, staging_bucket: str
) -> None:
    if _sinks.xet_unavailable_reason() is not None:
        pytest.skip("the xet backend is not available")
    df = pl.DataFrame({"g": ["a", "a", "b", None], "n": range(4)})

    sink_streamed(df, _uri(staging_bucket, "xet"), partition_by="g")
    sink_staged(df, _uri(staging_bucket, "hub"), partition_by="g")

    streamed = {}
    for path, info in _files(staging_api, staging_bucket, "xet/").items():
        streamed[path[len("xet/") :]] = info
    staged = {}
    for path, info in _files(staging_api, staging_bucket, "hub/").items():
        staged[path[len("hub/") :]] = info
    # Same names, same sizes, same content hashes.
    assert streamed == staged
    assert len(streamed) == 3


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


@both_sinks
def test_local_disk_use(
    staging_api: HfApi,
    staging_bucket: str,
    sink,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peak growth of the temp directory during a partitioned write.

    The output is about 16 MB by default; set ``POLARS_HF_DISK_TEST_MB`` for a
    larger measurement (the result is printed, see ``pytest -s``). The xet
    backend must not stage the output; the hub backend stages all of it.
    """
    import tempfile

    megabytes = int(os.environ.get("POLARS_HF_DISK_TEST_MB", "16"))
    staging = tmp_path / "staging"
    staging.mkdir()
    # tempfile.mkdtemp (hub backend) and any temp file of the Hub client go
    # below this directory; hf_xet reads TMPDIR.
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    monkeypatch.setenv("TMPDIR", str(staging))

    peak = 0
    stop = threading.Event()

    def sample_until_stopped() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, _directory_size(str(staging)))
            stop.wait(0.005)

    # About 22 bytes per row, uncompressed and incompressible enough.
    rows = megabytes * 1_000_000 // 22
    ids = pl.int_range(0, rows, eager=True)
    text = (ids * 2654435761 % 1000003).cast(pl.String) + "-" + ids.cast(pl.String)
    df = pl.DataFrame({"id": ids, "text": text})

    sampler = threading.Thread(target=sample_until_stopped, daemon=True)
    sampler.start()
    try:
        sink(
            df,
            _uri(staging_bucket, "disk"),
            max_rows_per_file=rows // 8,
            compression="uncompressed",
        )
    finally:
        stop.set()
        sampler.join()

    files = _files(staging_api, staging_bucket, "disk/")
    total = sum(size for size, _ in files.values())
    name = "hub" if sink is sink_staged else "xet"
    print(
        f"\nlocal disk [{name}]: output {total / 1e6:.1f} MB in {len(files)} files, "
        f"peak temp-dir growth {peak / 1e6:.1f} MB"
    )
    assert total > megabytes * 500_000
    if sink is sink_staged:
        assert peak >= total * 0.9
    else:
        assert peak <= total // 10


def test_default_sink_round_trip(staging_bucket: str) -> None:
    df = pl.DataFrame({"g": ["a", "b"], "n": [1, 2]})
    base = _uri(staging_bucket, "default")

    sink_default(df, base, partition_by="g")

    assert_frame_equal(plhf.scan_bucket(base).collect().sort("n"), df)


@both_sinks
def test_keyboard_interrupt_leaves_destination_unchanged(
    staging_api: HfApi, staging_bucket: str, sink
) -> None:
    # Deterministic: the interrupt is raised by the query itself, after some
    # row groups were written (Polars re-raises it unchanged).
    def interrupt(batch: pl.DataFrame) -> pl.DataFrame:
        if batch["n"].max() >= 150_000:
            raise KeyboardInterrupt
        return batch

    uri = _uri(staging_bucket, "keep/data.parquet")
    sink(pl.DataFrame({"n": range(1000)}), uri)
    files_before = _files(staging_api, staging_bucket)
    interrupted = pl.LazyFrame({"n": range(200_000)}).map_batches(
        interrupt, streamable=True
    )

    with pytest.raises(KeyboardInterrupt):
        sink(interrupted, uri, row_group_size=10_000)

    assert _files(staging_api, staging_bucket) == files_before
    # The xet backend aborted the process-wide Xet session; the next write
    # gets a new one.
    sink(pl.DataFrame({"n": [1]}), _uri(staging_bucket, "after.parquet"))
    assert "after.parquet" in _files(staging_api, staging_bucket)


@both_sinks
def test_path_the_hub_rejects_raises(
    staging_api: HfApi, staging_bucket: str, sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file the Hub refuses must raise with every supported huggingface_hub.

    The package's own path check is switched off, so the backslash reaches the
    Hub, which rejects that file and applies the others. huggingface_hub 1.x
    does not report this; the hub backend then finds the missing file with a
    listing.
    """
    monkeypatch.setattr(_sinks, "validate_destination", lambda path: None)
    df = pl.DataFrame({"g": ["ok", "a\\b", "fine"], "n": [1, 2, 3]})

    with pytest.raises(plhf.BucketRegistrationError) as error:
        sink(df, _uri(staging_bucket, "parts"), partition_by="g")

    rejected = [failure["path"] for failure in error.value.failures]
    assert rejected == ["parts/g=a\\b/00000000.parquet"]
    assert sorted(_files(staging_api, staging_bucket)) == [
        "parts/g=fine/00000000.parquet",
        "parts/g=ok/00000000.parquet",
    ]


@pytest.mark.skipif(
    os.environ.get("POLARS_HF_BIG_STAGING") != "1",
    reason="large upload: set POLARS_HF_BIG_STAGING=1 (the weekly CI run does)",
)
def test_big_write_is_identical_with_both_backends(
    staging_api: HfApi, staging_bucket: str
) -> None:
    """A few hundred MB over several partitions: same objects from both backends.

    ``POLARS_HF_BIG_STAGING_MB`` sets the uncompressed size (default 300).
    """
    if _sinks.xet_unavailable_reason() is not None:
        pytest.skip("the xet backend is not available")
    megabytes = int(os.environ.get("POLARS_HF_BIG_STAGING_MB", "300"))
    rows = megabytes * 1_000_000 // 30
    ids = pl.int_range(0, rows, eager=False)
    frame = pl.select(id=ids, eager=False).with_columns(
        g=pl.col("id") % 5,
        text=(pl.col("id") * 2654435761 % 1000003).cast(pl.String)
        + "-"
        + pl.col("id").cast(pl.String),
    )
    options = {
        "partition_by": "g",
        "max_rows_per_file": rows // 20,
        "compression": "uncompressed",
        # Ordered output: the same rows reach the same file in both runs.
        "engine": "streaming",
        "maintain_order": True,
    }

    sink_streamed(frame, _uri(staging_bucket, "xet"), **options)
    sink_staged(frame, _uri(staging_bucket, "hub"), **options)

    streamed = {}
    for path, info in _files(staging_api, staging_bucket, "xet/").items():
        streamed[path[len("xet/") :]] = info
    staged = {}
    for path, info in _files(staging_api, staging_bucket, "hub/").items():
        staged[path[len("hub/") :]] = info
    total = sum(size for size, _ in streamed.values())
    print(f"\nbig write: {len(streamed)} files, {total / 1e6:.0f} MB per backend")
    assert len(streamed) >= 20
    assert total > megabytes * 500_000
    # Same names, same sizes, same xet hashes.
    assert streamed == staged
    for name in ("xet", "hub"):
        count = plhf.scan_bucket(_uri(staging_bucket, name)).select(pl.len()).collect()
        assert count.item() == rows
