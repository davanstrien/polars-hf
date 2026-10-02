# Benchmarks

Two scripts that measure the read and the write path of `polars-hf` on real buckets. They are
[PEP 723](https://peps.python.org/pep-0723/) scripts: `uv` installs their dependencies, so they run
with `uv run` locally and with `hf jobs uv run` on Hugging Face Jobs. They need a token with read
access to the input bucket (and write access to the output bucket for `sink_backends.py`), in
`HF_TOKEN` or from `hf auth login`.

Neither script names a bucket: you give the input and the output. Neither creates or deletes a
bucket. `sink_backends.py` writes below the output prefix you give and leaves the files there.

The scripts install `polars-hf` from the `main` branch of the repository. To measure another
revision, change the `polars-hf @ git+...@main` line at the top of the script. To pin Polars, add
for example `--with "polars==2.0.0rc2" --with "polars-runtime-32==2.0.0rc2"` to the `uv run`
options.

## `read_paths.py`

Compares four ways to run the same query over the parquet files of one bucket directory:

| Arm | What it does |
| --- | --- |
| `redirect` | `scan_bucket(uri)`, the default: the native `scan_parquet` node over URLs of a local server that redirects to the presigned URLs |
| `collect` | `scan_bucket(uri, resolve="collect")`: the URLs are resolved when the query runs, group by group, behind an IO-plugin node |
| `now` | `scan_bucket(uri, resolve="now")`: every presigned URL is resolved first, then the native node reads them directly. The arm sets `POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN=1` for its own process |
| `download` | `HfApi.download_bucket_files` (`hf_xet`) downloads a few files at a time into a staging directory; each batch is scanned from local disk and deleted, while the next batch is downloaded |

and two queries: `full` (an aggregate that decodes every value of `--columns`; all columns by
default) and `selective` (the same aggregate over `--selective-columns`; the first column by
default). `--filter-column` / `--filter-min` add a `column >= value` predicate.

```bash
hf jobs uv run --flavor cpu-performance --secrets HF_TOKEN \
    -e BENCH_INPUT=hf://buckets/<namespace>/<bucket>/<directory> \
    -e BENCH_WARMUP=1 \
    -e BENCH_OUTPUT=hf://buckets/<namespace>/<bucket>/results/read_paths.jsonl \
    benchmarks/read_paths.py

# locally, with arguments:
uv run benchmarks/read_paths.py --input hf://buckets/<namespace>/<bucket>/<directory> \
    --arms redirect,now --queries full --columns text,url --warmup
```

Options (each has a `BENCH_...` environment variable; `--help` lists them): `--input`, `--arms`,
`--queries`, `--columns`, `--selective-columns`, `--filter-column`, `--filter-min`,
`--download-group` (files per batch of the `download` arm, default 4), `--staging-dir` (where the
`download` arm puts its files; `/dev/shm` stages in memory), `--warmup`, `--output`.

Each run (one arm, one query) is a new Python process. It prints one JSON object, which is also
appended to `--output`:

| Field | Meaning |
| --- | --- |
| `seconds` | wall time from the `scan_bucket` call (or the first download) to the result of the query; the listing and the `resolve` requests are included |
| `cpu_seconds`, `cores_avg` | user + system CPU time of the process, and that time divided by `seconds` |
| `network_bytes` | bytes received on all network interfaces but loopback during the run (`/proc/net/dev`); `null` where that file does not exist (macOS) |
| `peak_rss_bytes` | peak resident memory of the process (`ru_maxrss`). For a staging directory in `/dev/shm` the staged files are not part of this number |
| `peak_staging_bytes` | largest growth of the used bytes of the staging directory's file system, sampled every 0.25 s. It counts everything written to that file system during the run, so use a file system that nothing else writes to |
| `files`, `input_bytes` | the parquet files below `--input` and their total size |
| `rows`, `sums` | the result of the query; equal for all arms of one query, which is the check that the arms read the same data |
| `warmup` | `true` for the warm-up run |

### Cold and warm CDN

The first read of bucket data that was written or copied recently is slower than the next ones: the
bytes are not at the CDN edge yet. An arm that runs first on cold data pays for the arms after it.
Two ways to keep the comparison fair:

- `--warmup` reads every input once (all columns, `redirect` arm) before the arms. All arms then read
  warm data. The warm-up run is in the output with `"warmup": true`.
- Give `--input` one directory per arm, separated by commas (as many as arms). Each arm then reads
  its own files, cold or warm alike. The directories must hold comparable data.

Say which of the two a published number used.

## `sink_backends.py`

Copies the parquet files of `--input` to `{output-prefix}/{arm}` with `sink_bucket`, once per arm.
An arm is `{backend}-{shape}`: backend `stream` or `staged`, shape `single` (one output file) or
`parts` (a partitioned write with `max_bytes_per_file`).

```bash
hf jobs uv run --flavor cpu-performance --secrets HF_TOKEN \
    -e BENCH_INPUT=hf://buckets/<namespace>/<bucket>/<directory> \
    -e BENCH_OUTPUT_PREFIX=hf://buckets/<namespace>/<bucket>/bench-out \
    -e BENCH_ARMS=stream-single,stream-parts,staged-parts \
    benchmarks/sink_backends.py
```

Options: `--input`, `--output-prefix`, `--arms`, `--columns`, `--max-bytes-per-file` (default
1e9), `--compression` (default `zstd`), `--output` (a local JSON lines file; the lines are also
printed).

| Field | Meaning |
| --- | --- |
| `seconds` | wall time of the `sink_bucket` call: read, write, upload and registration |
| `peak_rss_bytes` | peak resident memory of the process: the larger of `ru_maxrss` and of `VmRSS` sampled every 0.5 s |
| `peak_disk_bytes` | largest growth of the used bytes of the temporary directory's file system during the call, sampled every 0.5 s. This is where the `staged` backend stages its output |
| `output_files`, `output_bytes` | the files below the destination after the write |
| `rows_read_back` | `scan_bucket(destination).select(pl.len())` after the write |

The `staged` backend needs free local disk of the size of the output. Every run writes with
`mode="overwrite"`, so a second run replaces the output of the first.

## Results

These numbers are from the ad-hoc scripts that the two scripts above were made from. They were
measured before the collect-time read path was merged: the "IO-plugin wrapper" row is a prototype
that resolved all URLs in one group when the query ran, which is what the `collect` arm does for up
to 64 files, and "native presigned" is what the `now` arm does. The scripts in this directory have
not been run on Jobs yet. Every row is one run, on
HF Jobs `cpu-performance` (32 vCPU). GB is 10^9 bytes.

### Read: full scan of four columns, 54 files, 126.7 GB

Job `6abe5426404719ba3761835c`, Polars 2.0.0rc2. All arms read CDN-warm data.

| Arm | Seconds | Cores (avg) | Peak RSS | Peak staging |
| --- | --- | --- | --- | --- |
| rolling `hf_xet` download to disk, local scan | 114.0 | 10.7 | 7.3 GB | 15.1 GB |
| rolling `hf_xet` download to `/dev/shm`, local scan | 115.4 | not recorded here | not recorded here | not recorded here |
| IO-plugin wrapper over the presigned scan | 244.0 | 4.2 | 3.2 GB | none |
| native presigned scan | 300.8 | 3.3 | 2.5 GB | none |

The wrapper and the native scan returned the same result. One run each does not separate 244 s
from 301 s: read it as "the wrapper showed no overhead for a full scan", not as a speed-up.

On a 3-file fixture, a row count took 2.5 s through the wrapper and 1.8 s on the native node: an IO
source is asked for one column to count rows, the native node answers from the footers.

### Read: redirect mode and presigned URLs read directly, 12 files, 28 GB

Jobs `6abf69f8fbc85ba682369709` and `6abf6c26fbc85ba6823699c8`, Polars 2.0.0rc2, with a prototype
of the redirect server (`asyncio`, backlog 4,096). Selective queries were noisy in both arms.

| Query | `resolve="redirect"` | presigned URLs read directly |
| --- | --- | --- |
| row count, `select(pl.len())` | 0.5–0.6 s | 0.5–0.6 s |
| `head(5)` | 0.6–0.9 s | 0.5 s |
| `tail(5)` | 0.6–0.8 s | 0.6 s |
| full scan of four columns | 48.2–50.6 s | 46.8–48.1 s |
| one small column | 5.9–9.0 s | 8.6–22.1 s |
| filter and two columns | 4.1–13.2 s | 5.5–7.1 s |

With a server from `http.server` (listen backlog 5) the selective queries took 165–264 s.

### Read: collect mode and native node, 12 files, 28 GB

Job `6abec96b404719ba3761a56b`, Polars 2.0.0rc2, with the IO-plugin source of this repository
(`resolve="collect"`) and the native node over presigned URLs (`resolve="now"`).

| Query | `resolve="collect"` | `resolve="now"` (native node) |
| --- | --- | --- |
| row count, `select(pl.len())` | 30–33 s, 27.7 GB downloaded | 1.1 s, 0.07 GB downloaded |
| `tail(5)` | 19 GB peak memory | 2.6 GB peak memory |
| one small column | 3.3–4.8 s | 2.1–2.9 s |
| `head(5)` | 0.8 s | 0.7 s |
| full scan of four columns | 22 s | 24 s |

Polars asks an IO source for one column to count rows; here it was nearly all of the data.
`polars_hf.count_rows` and the default mode read the footers instead. `tail()` is not pushed into an
IO source, so the collect mode scans all files for it.

### `POLARS_CONCURRENCY_BUDGET`

A presigned scan of 14 GB with `POLARS_CONCURRENCY_BUDGET=64` and without it.

| Polars | Job | With the variable | Without |
| --- | --- | --- | --- |
| 2.0.0rc2 | `6abe4f2e404719ba376181f0` | 31.0 s, 32.1 s | 29.1 s, 33.6 s |
| 1.44.2 | `6abe4d4cfbc85ba682361f8b` | about 8% faster | |

### Write: bucket-to-bucket copy

Job `6abe7042404719ba37618a7f`.

| Output | Backend | Seconds | Peak RSS | Peak local disk |
| --- | --- | --- | --- | --- |
| 1 file, 8.8 GB | `stream` | 45 | 17.8 GB | not recorded here |
| 1 file, 35.0 GB | `stream` | 151 | 23.3 GB | not recorded here |
| 1 file, 79.2 GB | `stream` | 308 | 24.0 GB | not recorded here |
| 96 files, 35.0 GB | `stream` | 126 | 21.8 GB | 0.1 GB |
| 96 files, 35.0 GB | `staged` | 124 | 16.7 GB | 35.1 GB |
