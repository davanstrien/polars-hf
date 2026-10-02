# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars-hf @ git+https://github.com/davanstrien/polars-hf@main",
#     "polars>=1.40",
#     "huggingface_hub>=2.1",
#     "hf_xet>=1.6.0",
# ]
# ///
"""Compare the two backends of ``sink_bucket`` on a bucket-to-bucket copy.

Every run reads the parquet files of ``--input`` with ``scan_bucket`` and
writes them with ``sink_bucket`` below ``--output-prefix``.

Arms (``--arms``), named ``{backend}-{shape}``:

* backend ``stream``: each output file is streamed into Xet storage while
  Polars writes it; backend ``staged``: Polars writes to a temporary directory
  that is uploaded afterwards;
* shape ``single``: one output file; shape ``parts``: a partitioned write
  with ``max_bytes_per_file`` (``--max-bytes-per-file``).

Every run is a new Python process. It records the wall time of the
``sink_bucket`` call, the peak resident memory, the peak growth of the
temporary directory's file system, the files and bytes written, and the row
count read back from the output. The script prints one JSON object per run
and appends it to ``--output``.

The output of a run stays in the bucket, at
``{output-prefix}/{backend}-{shape}``; a second run replaces it
(``mode="overwrite"``).

Every option has an environment variable (see ``--help``)::

    hf jobs uv run --flavor cpu-performance --secrets HF_TOKEN \\
        -e BENCH_INPUT=hf://buckets/<namespace>/<bucket>/<directory> \\
        -e BENCH_OUTPUT_PREFIX=hf://buckets/<namespace>/<bucket>/bench-out \\
        benchmarks/sink_backends.py

``benchmarks/README.md`` explains the numbers.
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

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

BACKENDS = ("stream", "staged")
SHAPES = ("single", "parts")
ARMS = ("stream-single", "stream-parts", "staged-parts")
RESULT_PREFIX = "RESULT "


# ---- measurements ----------------------------------------------------------


def current_rss_bytes() -> int | None:
    """Resident set size of this process now; ``None`` off Linux."""
    try:
        with open("/proc/self/status") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def peak_rss_bytes() -> int:
    """Peak resident set size of this process."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    if sys.platform == "darwin":
        return peak
    return peak * 1024


class Watch:
    """Samples the memory of the process and the used bytes of a file system.

    ``peak_disk`` is the largest growth of the file system of ``directory``
    over its value at the start: everything written there during the run.
    """

    def __init__(self, directory: str, interval: float = 0.5) -> None:
        self.directory = directory
        self.interval = interval
        self.base = shutil.disk_usage(directory).used
        self.peak_disk = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            grown = shutil.disk_usage(self.directory).used - self.base
            self.peak_disk = max(self.peak_disk, grown)
            rss = current_rss_bytes()
            if rss is not None:
                self.peak_rss = max(self.peak_rss, rss)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


# ---- one run ---------------------------------------------------------------


def list_files(uri: str) -> list:
    """The files below the directory ``uri``."""
    from huggingface_hub import HfApi

    import polars_hf as plhf

    bucket_path = plhf.parse_bucket_uri(uri)
    prefix = bucket_path.path.rstrip("/") + "/"
    listed = HfApi().list_bucket_tree(
        bucket_path.bucket_id, prefix=prefix, recursive=True
    )
    files = []
    for entry in listed:
        if entry.type == "file" and entry.path.startswith(prefix):
            files.append(entry)
    return files


def run_child(spec: dict) -> None:
    """One backend and one shape in this process; prints the result line."""
    import polars as pl

    import polars_hf as plhf

    backend, shape = spec["arm"].split("-")
    lf = plhf.scan_bucket(spec["input"].rstrip("/") + "/")
    if spec["columns"] is not None:
        lf = lf.select(spec["columns"])
    destination = f"{spec['output_prefix'].rstrip('/')}/{spec['arm']}"

    watch = Watch(tempfile.gettempdir())
    watch.start()
    started = time.perf_counter()
    options = {
        "mode": "overwrite",
        "backend": backend,
        "compression": spec["compression"],
    }
    if shape == "single":
        written = f"{destination}/all.parquet"
        plhf.sink_bucket(lf, written, **options)
    else:
        written = f"{destination}/"
        plhf.sink_bucket(
            lf, destination, max_bytes_per_file=spec["max_bytes_per_file"], **options
        )
    seconds = time.perf_counter() - started
    watch.stop()

    files = list_files(destination)
    output_bytes = sum(entry.size for entry in files)
    rows = plhf.scan_bucket(written).select(pl.len()).collect().item()
    result = {
        "arm": spec["arm"],
        "backend": backend,
        "shape": shape,
        "input": spec["input"],
        "destination": destination,
        "seconds": round(seconds, 2),
        "output_files": len(files),
        "output_bytes": output_bytes,
        "peak_rss_bytes": max(watch.peak_rss, peak_rss_bytes()),
        "peak_disk_bytes": watch.peak_disk,
        "rows_read_back": rows,
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
        "input": spec["input"],
        "failed": True,
        "returncode": done.returncode,
        "error": " | ".join(tail)[-1500:],
    }


# ---- command line ----------------------------------------------------------


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        default=os.environ.get("BENCH_INPUT"),
        help="hf://buckets/... URI of a directory of parquet files to copy. "
        "Default: $BENCH_INPUT.",
    )
    parser.add_argument(
        "--output-prefix",
        default=os.environ.get("BENCH_OUTPUT_PREFIX"),
        help="hf://buckets/... directory to write below; every arm writes to "
        "its own sub-directory. Default: $BENCH_OUTPUT_PREFIX.",
    )
    parser.add_argument(
        "--arms",
        default=os.environ.get("BENCH_ARMS", ",".join(ARMS)),
        help="comma-separated {backend}-{shape} names; backends "
        f"{BACKENDS}, shapes {SHAPES}. Default: {','.join(ARMS)}",
    )
    parser.add_argument(
        "--columns",
        default=os.environ.get("BENCH_COLUMNS"),
        help="columns to copy (default: all)",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=int(os.environ.get("BENCH_MAX_BYTES_PER_FILE", "1000000000")),
        help="target file size of the parts shape (default 1e9)",
    )
    parser.add_argument(
        "--compression",
        default=os.environ.get("BENCH_COMPRESSION", "zstd"),
        help="parquet compression of the output (default zstd)",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("BENCH_OUTPUT", "sink_backends.jsonl"),
        help="local JSON lines file to append the results to",
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.child is not None:
        run_child(json.loads(arguments.child))
        return
    if not arguments.input or not arguments.output_prefix:
        raise SystemExit(
            "give --input and --output-prefix (or set BENCH_INPUT and "
            "BENCH_OUTPUT_PREFIX)"
        )

    arms = comma_list(arguments.arms)
    for arm in arms:
        backend, _, shape = arm.partition("-")
        if backend not in BACKENDS or shape not in SHAPES:
            raise SystemExit(
                f"unknown arm {arm!r}; use {{backend}}-{{shape}} with backends "
                f"{BACKENDS} and shapes {SHAPES}"
            )
    columns = None
    if arguments.columns:
        columns = comma_list(arguments.columns)

    with open(arguments.output, "a") as output:
        for arm in arms:
            spec = {
                "arm": arm,
                "input": arguments.input,
                "output_prefix": arguments.output_prefix,
                "columns": columns,
                "max_bytes_per_file": arguments.max_bytes_per_file,
                "compression": arguments.compression,
            }
            line = json.dumps(run_in_subprocess(spec))
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()


if __name__ == "__main__":
    main()
