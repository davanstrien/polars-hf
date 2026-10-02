# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars-hf @ git+https://github.com/davanstrien/polars-hf@main",
#     "polars>=1.40",
#     "huggingface_hub>=2.1",
#     "hf_xet>=1.6.0",
# ]
# ///
"""Compare three ways to read the parquet files of a bucket directory.

Arms (``--arms``):

* ``collect``: ``scan_bucket(uri)``, the default. The URLs are resolved when
  the query runs, group by group, behind an IO-plugin node.
* ``now``: ``scan_bucket(uri, resolve="now")``. Every presigned URL is
  resolved first; the query runs on the native ``scan_parquet`` node. The arm
  sets ``POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1`` for its own process.
* ``download``: the files are downloaded with ``hf_xet``
  (``HfApi.download_bucket_files``) into a staging directory, a few files at a
  time, and each batch is scanned from the local disk and deleted. The next
  batch is downloaded while the current one is scanned.

Queries (``--queries``):

* ``full``: one aggregate over ``--columns`` (all columns by default), so
  every value of these columns is decoded;
* ``selective``: the same aggregate over ``--selective-columns`` (the first
  column by default).

Both count the rows, sum the numeric columns and sum the byte length of the
string columns. Other column types are counted. ``--filter-column`` and
``--filter-min`` add a ``column >= value`` predicate to both.

Every run (one arm, one query) is a new Python process, so peak memory is per
run. The script prints one JSON object per run and appends it to ``--output``
(a local file, or an ``hf://buckets/...`` file that is written at the end).

Every option has an environment variable (``BENCH_INPUT``, ``BENCH_ARMS``,
...; see ``--help``), so a job needs no script arguments::

    hf jobs uv run --flavor cpu-performance --secrets HF_TOKEN \\
        -e BENCH_INPUT=hf://buckets/<namespace>/<bucket>/<directory> \\
        -e BENCH_WARMUP=1 \\
        -e BENCH_OUTPUT=hf://buckets/<namespace>/<bucket>/results/read.jsonl \\
        benchmarks/read_paths.py

``benchmarks/README.md`` explains the numbers and the cold/warm CDN caveat.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ARMS = ("collect", "now", "download")
QUERIES = ("full", "selective")
RESULT_PREFIX = "RESULT "


# ---- measurements ----------------------------------------------------------


def network_bytes_received() -> int | None:
    """Bytes received on all interfaces but loopback; ``None`` off Linux."""
    try:
        with open("/proc/net/dev") as file:
            lines = file.readlines()[2:]
    except OSError:
        return None
    total = 0
    for line in lines:
        name, counters = line.split(":", 1)
        if name.strip() != "lo":
            total += int(counters.split()[0])
    return total


def peak_rss_bytes() -> int:
    """Peak resident set size of this process."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    if sys.platform == "darwin":
        return peak
    return peak * 1024


class DiskWatch:
    """Samples the used bytes of the file system of ``directory``.

    The peak is the largest growth over the value at the start. It counts
    everything written to that file system during the run, not only the
    staging files.
    """

    def __init__(self, directory: str, interval: float = 0.25) -> None:
        self.directory = directory
        self.interval = interval
        self.base = shutil.disk_usage(directory).used
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            grown = shutil.disk_usage(self.directory).used - self.base
            self.peak = max(self.peak, grown)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return self.peak


# ---- input -----------------------------------------------------------------


def list_parquet_files(uri: str) -> tuple[str, list]:
    """The bucket id and the parquet files below the directory ``uri``."""
    from huggingface_hub import HfApi

    import polars_hf as plhf

    bucket_path = plhf.parse_bucket_uri(uri)
    prefix = bucket_path.path.rstrip("/")
    if prefix:
        prefix += "/"
    listed = HfApi().list_bucket_tree(
        bucket_path.bucket_id, prefix=prefix or None, recursive=True
    )
    files = []
    for entry in listed:
        if entry.type != "file" or not entry.path.startswith(prefix):
            continue
        if entry.path.lower().endswith((".parquet", ".pq")):
            files.append(entry)
    files.sort(key=lambda entry: entry.path)
    if not files:
        raise SystemExit(f"no parquet files below {uri!r}")
    return bucket_path.bucket_id, files


def aggregate(lf, columns: list[str] | None, filter_column: str | None, filter_min):
    """Decode ``columns`` of ``lf``: one row of sums. Returns the row."""
    import polars as pl

    schema = lf.collect_schema()
    if columns is None:
        columns = schema.names()
    expressions = [pl.len()]
    for name in columns:
        dtype = schema[name]
        if dtype == pl.String:
            expressions.append(pl.col(name).str.len_bytes().cast(pl.UInt64).sum())
        elif dtype.is_numeric():
            expressions.append(pl.col(name).cast(pl.Float64).sum().round(1))
        else:
            expressions.append(pl.col(name).count())
    if filter_column is not None:
        lf = lf.filter(pl.col(filter_column) >= filter_min)
    return lf.select(expressions).collect(engine="streaming").row(0)


# ---- the download arm ------------------------------------------------------


def rolling_download_scan(
    bucket_id: str, files: list, root: str, group_files: int, schema
):
    """A LazyFrame that downloads and scans ``files`` a group at a time.

    The frame is an IO-plugin source. For every group it downloads the files
    into ``root`` with ``hf_xet``, scans them with a native ``scan_parquet``
    (projection and predicate applied) and deletes them. The download of the
    next group runs while the current group is scanned, so at most two groups
    are on disk. ``schema`` is the schema of the files.
    """
    import polars as pl
    from huggingface_hub import HfApi
    from polars.io.plugins import register_io_source

    api = HfApi()
    groups = []
    for start in range(0, len(files), group_files):
        groups.append(files[start : start + group_files])

    def directory(index: int) -> str:
        return os.path.join(root, f"group-{index:05d}")

    def download(index: int) -> list[str]:
        os.makedirs(directory(index), exist_ok=True)
        pairs = []
        for position, entry in enumerate(groups[index]):
            local = os.path.join(directory(index), f"{position:04d}.parquet")
            pairs.append((entry, local))
        api.download_bucket_files(bucket_id, pairs)
        return [local for _, local in pairs]

    def source(with_columns, predicate, n_rows, batch_size):
        remaining = n_rows
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(download, 0)
            for index in range(len(groups)):
                paths = pending.result()
                if index + 1 < len(groups):
                    pending = pool.submit(download, index + 1)
                lf = pl.scan_parquet(paths)
                if remaining is not None:
                    lf = lf.head(remaining)
                if predicate is not None:
                    lf = lf.filter(predicate)
                if with_columns is not None:
                    lf = lf.select(with_columns)
                batches = lf.collect_batches(chunk_size=batch_size, engine="streaming")
                for batch in batches:
                    if remaining is not None:
                        remaining -= batch.height
                    yield batch
                shutil.rmtree(directory(index), ignore_errors=True)
                if remaining is not None and remaining <= 0:
                    pending.cancel()
                    break

    return register_io_source(source, schema=schema)


# ---- one run ---------------------------------------------------------------


def run_child(spec: dict) -> None:
    """One arm and one query in this process; prints the result line."""
    import polars as pl

    import polars_hf as plhf

    uri = spec["input"]
    bucket_id, files = list_parquet_files(uri)
    input_bytes = sum(entry.size for entry in files)
    directory_uri = uri.rstrip("/") + "/"

    staging_root = os.path.join(spec["staging_dir"], "polars-hf-bench-read")
    shutil.rmtree(staging_root, ignore_errors=True)
    os.makedirs(staging_root)
    disk = DiskWatch(spec["staging_dir"])
    disk.start()

    received_before = network_bytes_received()
    started = time.perf_counter()
    if spec["arm"] == "collect":
        lf = plhf.scan_bucket(directory_uri, resolve="collect")
    elif spec["arm"] == "now":
        # The arm measures the scan of presigned URLs that are in the plan.
        # polars-hf refuses that mode without this acknowledgement; the plan
        # of this process is not printed, serialized or logged.
        os.environ["POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN"] = "1"
        lf = plhf.scan_bucket(directory_uri, resolve="now")
    else:
        # The schema: one resolve request and the footer of the first file.
        first_file = f"hf://buckets/{bucket_id}/{files[0].path}"
        schema = plhf.scan_bucket(first_file).collect_schema()
        lf = rolling_download_scan(
            bucket_id, files, staging_root, spec["download_group"], schema
        )
    row = aggregate(lf, spec["columns"], spec["filter_column"], spec["filter_min"])
    seconds = time.perf_counter() - started

    peak_staging = disk.stop()
    shutil.rmtree(staging_root, ignore_errors=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = usage.ru_utime + usage.ru_stime
    received = None
    if received_before is not None:
        received = network_bytes_received() - received_before
    result = {
        "arm": spec["arm"],
        "query": spec["query"],
        "warmup": spec["warmup"],
        "input": uri,
        "files": len(files),
        "input_bytes": input_bytes,
        "columns": spec["columns"],
        "seconds": round(seconds, 2),
        "cpu_seconds": round(cpu_seconds, 1),
        "cores_avg": round(cpu_seconds / seconds, 2),
        "network_bytes": received,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_staging_bytes": peak_staging,
        "rows": row[0],
        "sums": list(row[1:]),
        "polars": pl.__version__,
        "polars_hf": plhf.__version__,
        "threads": pl.thread_pool_size(),
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def run_in_subprocess(spec: dict) -> dict:
    """Run ``spec`` in a new process and return its result (or the failure)."""
    command = [sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec)]
    done = subprocess.run(command, capture_output=True, text=True)
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    tail = (done.stderr or done.stdout).strip().splitlines()[-8:]
    return {
        "arm": spec["arm"],
        "query": spec["query"],
        "input": spec["input"],
        "failed": True,
        "returncode": done.returncode,
        "error": " | ".join(tail)[-1500:],
    }


# ---- command line ----------------------------------------------------------


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_results(lines: list[str], output: str) -> None:
    """Append ``lines`` to a local file, or write them to a bucket file."""
    text = "".join(line + "\n" for line in lines)
    if not output.startswith("hf://buckets/"):
        with open(output, "a") as file:
            file.write(text)
        return
    from huggingface_hub import HfApi

    import polars_hf as plhf

    bucket_path = plhf.parse_bucket_uri(output)
    HfApi().batch_bucket_files(
        bucket_path.bucket_id, add=[(text.encode(), bucket_path.path)]
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        default=os.environ.get("BENCH_INPUT"),
        help="hf://buckets/... URI of a directory of parquet files. A "
        "comma-separated list with one directory per arm gives every arm its "
        "own files. Default: $BENCH_INPUT.",
    )
    parser.add_argument(
        "--arms",
        default=os.environ.get("BENCH_ARMS", ",".join(ARMS)),
        help=f"comma-separated, from {ARMS}; they run in this order",
    )
    parser.add_argument(
        "--queries",
        default=os.environ.get("BENCH_QUERIES", ",".join(QUERIES)),
        help=f"comma-separated, from {QUERIES}",
    )
    parser.add_argument(
        "--columns",
        default=os.environ.get("BENCH_COLUMNS"),
        help="columns of the full query (default: all)",
    )
    parser.add_argument(
        "--selective-columns",
        default=os.environ.get("BENCH_SELECTIVE_COLUMNS"),
        help="columns of the selective query (default: the first column)",
    )
    parser.add_argument(
        "--filter-column",
        default=os.environ.get("BENCH_FILTER_COLUMN"),
        help="column of a >= predicate",
    )
    parser.add_argument(
        "--filter-min",
        type=float,
        default=float(os.environ.get("BENCH_FILTER_MIN", "0")),
    )
    parser.add_argument(
        "--download-group",
        type=int,
        default=int(os.environ.get("BENCH_DOWNLOAD_GROUP", "4")),
        help="files per batch of the download arm (default 4)",
    )
    parser.add_argument(
        "--staging-dir",
        default=os.environ.get("BENCH_STAGING_DIR", tempfile.gettempdir()),
        help="where the download arm puts its files, e.g. /dev/shm",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=os.environ.get("BENCH_WARMUP") == "1",
        help="read every input once before the arms, so that all arms read "
        "CDN-warm data",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("BENCH_OUTPUT", "read_paths.jsonl"),
        help="JSON lines file to append the results to; an hf://buckets/... "
        "URI is written once, at the end",
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.child is not None:
        run_child(json.loads(arguments.child))
        return
    if not arguments.input:
        raise SystemExit("give --input (or set BENCH_INPUT)")

    arms = comma_list(arguments.arms)
    queries = comma_list(arguments.queries)
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {ARMS}")
    for query in queries:
        if query not in QUERIES:
            raise SystemExit(f"unknown query {query!r}; choose from {QUERIES}")
    inputs = comma_list(arguments.input)
    if len(inputs) not in (1, len(arms)):
        raise SystemExit("--input needs one directory, or one directory per arm")

    columns = None
    if arguments.columns:
        columns = comma_list(arguments.columns)
    selective_columns = None
    if arguments.selective_columns:
        selective_columns = comma_list(arguments.selective_columns)

    def spec_for(arm: str, query: str, uri: str, *, warmup: bool) -> dict:
        query_columns = columns
        if query == "selective":
            query_columns = selective_columns
            if query_columns is None:
                import polars_hf as plhf

                first = plhf.scan_bucket(uri.rstrip("/") + "/").collect_schema()
                query_columns = first.names()[:1]
        return {
            "arm": arm,
            "query": query,
            "input": uri,
            "columns": query_columns,
            "filter_column": arguments.filter_column,
            "filter_min": arguments.filter_min,
            "download_group": arguments.download_group,
            "staging_dir": arguments.staging_dir,
            "warmup": warmup,
        }

    specs = []
    if arguments.warmup:
        for uri in dict.fromkeys(inputs):
            # Every column and no predicate: every byte of the files.
            warmup = spec_for("collect", "full", uri, warmup=True)
            warmup["columns"] = None
            warmup["filter_column"] = None
            specs.append(warmup)
    for query in queries:
        for position, arm in enumerate(arms):
            uri = inputs[position] if len(inputs) > 1 else inputs[0]
            specs.append(spec_for(arm, query, uri, warmup=False))

    lines = []
    for spec in specs:
        line = json.dumps(run_in_subprocess(spec))
        print(line, flush=True)
        lines.append(line)
        if not arguments.output.startswith("hf://buckets/"):
            write_results([line], arguments.output)
    if arguments.output.startswith("hf://buckets/"):
        write_results(lines, arguments.output)


if __name__ == "__main__":
    main()
