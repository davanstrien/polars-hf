# Writing to a bucket: reference

Details behind [`sink_bucket`](../README.md#write). The README has the short version.


`sink_bucket` accepts a `LazyFrame` (streaming) or a `DataFrame` and runs the query before it
returns (`lazy=True` is rejected). Partitioned writes split by key, by size, or both.
`max_rows_per_file` is a limit. `max_bytes_per_file` is a target, not a limit: it is Polars'
`approximate_bytes_per_file`, an estimate made while the rows are written, so a file can be larger.

**Object names.** A partitioned write uses the names Polars writes to a local directory:
`key=value/` directories and files `00000000.parquet`, `00000001.parquet`, ... (the index is
hexadecimal; the extension is `.parquet`, `.csv`, `.ipc` or `.jsonl`). Key values are
percent-encoded like Polars does (`/`, `=`, `%`, `:`, space, control characters and non-ASCII
bytes), and a null key is `__HIVE_DEFAULT_PARTITION__`. With `mode="append"` every file name also
carries a token that is unique to the call: `00000000-1f0c9a52b7e3.parquet` (12 random hex
characters, the same for all files of one call). A partitioned write whose query returns no rows
writes one file with the schema and no rows, `{prefix}/00000000.{extension}` (with the token in
`mode="append"`), so the prefix can be scanned afterwards.

**`mode`** sets what happens if the destination exists. A single-file destination exists if there
is an object at that path. A partitioned destination exists if there is a file anywhere below the
base prefix.

| `mode` | Single file | Partitioned (base prefix) |
| --- | --- | --- |
| `"error"` (default) | `FileExistsError` if the object exists | `FileExistsError` if the destination exists |
| `"append"` | not possible: `ValueError` | new files are added under names that carry a random 48-bit run id per call; nothing is deleted |
| `"overwrite"` | the object is replaced | the new files are registered (a file with the same name is replaced), then the files that were below the prefix before the write, and that this call did not write, are deleted |

**A file and a directory of one name are refused, in every mode.** A single-file write to `out`
raises `FileExistsError` if there are objects below `out/`, and a partitioned write to `out/...`
raises `FileExistsError` if `out` is a file. The bucket could store both; `sink_bucket` does not
create such a pair. Only the destination itself is checked: an *ancestor* that is a file (a write to
`a/b/c.parquet` while `a` is a file) is not detected.

The checks run before anything is uploaded. An existence check costs one listing page of the
destination directory plus one exact check of the path, whatever the destination and its siblings
hold. The checks do not exclude a concurrent writer.

| Write | Requests before the upload |
| --- | --- |
| single file, `"error"` | 2: one `HEAD` of the path, one listing page of `path/` |
| single file, `"overwrite"` | 1: one listing page of `path/` |
| partitioned, `"error"` | 2: one `HEAD` of the prefix, one listing page of `prefix/` (1 at the bucket root) |
| partitioned, `"append"` | 1: one `HEAD` of the prefix (0 at the bucket root); the prefix is not listed |
| partitioned, `"overwrite"` | one `HEAD` of the prefix, then every listing page of `prefix/` |

These requests use the listing and the retry limits of the read path: bounded retries, a deadline
of 10 minutes per call, `PermissionError` for a token without access and `FileNotFoundError` for a
bucket that does not exist.

`"append"` gives the files of each call names with a random 48-bit run id. Two appends with the
same partition keys therefore keep the rows of both, and running the same job twice adds the rows
twice. A name collision with an existing object is negligible but not excluded by a check:
`sink_bucket` does not list the prefix to look for a name that exists already.

`"overwrite"` needs a directory below the bucket root: `hf://buckets/ns/name` and
`hf://buckets/ns/name/` are refused with a `ValueError`. It lists the prefix before the write and
deletes only files from that listing, so a file that another writer adds during the write is kept.
A query that returns no rows still deletes the previous files and leaves one file with the schema
and no rows.
A file that another writer *replaces* during the write is still deleted.

**Paths.** The Hub refuses a path with a backslash, an empty segment (`a//b`) or a `.` / `..`
segment. `sink_bucket` also refuses a control character in a path and a path segment of more than
255 bytes, with both backends, because the `"staged"` backend cannot create such a name on a local
file system. It raises a `ValueError` for such a destination before it sends a request. A partition
column name or value can produce such a path too (a backslash is not percent-encoded, and an
encoded value counts with its `key=` prefix towards the 255 bytes); the error names the `key=value`
segment, and no file of that write is registered.

**`backend`** sets how the output reaches the bucket:

| `backend` | How | Local disk | Needs |
| --- | --- | --- | --- |
| `"stream"` | each output file is streamed into Xet storage with `hf_xet` while Polars writes it, then all files are registered in the bucket | none for the output | `huggingface_hub>=1.19` and the `hf_xet` that it requires (installed with it on x86_64 and arm64) |
| `"staged"` | Polars writes the output to a temporary directory; the public `HfApi.batch_bucket_files` uploads and registers it | the size of the complete output | any supported `huggingface_hub` |

The default (`backend=None`) is the environment variable `POLARS_HF_SINK_BACKEND` if set, else
`"stream"` when the installed packages support it, else `"staged"`. An explicit `"stream"` that cannot run
raises an error; it does not fall back.

With both backends, the deletes of `mode="overwrite"` are sent to the bucket batch endpoint
(`/api/buckets/{id}/batch`) directly, not through `HfApi.batch_bucket_files`: `huggingface_hub` 1.x
does not report a rejected delete, and the direct request lets `sink_bucket` check the answer.

Notes on the `"stream"` backend:

- It uses private helpers of `huggingface_hub.utils._xet` and builds the
  `/api/buckets/{id}/batch` request itself, because the public API has no streaming upload. If a
  later `huggingface_hub` or `hf_xet` changes these, `sink_bucket` raises a `RuntimeError` that
  names the installed versions and points to `backend="staged"`; it does not fall back silently.
- A `KeyboardInterrupt` during a write aborts the process-wide Xet session, as `huggingface_hub`
  does in its own uploads. Other Xet uploads and downloads that run in the same process are
  cancelled.
- All upload streams stay open until the query ends, because Polars does not signal when one file is
  complete.

Notes on the `"staged"` backend:

- It stages in the directory named by `POLARS_HF_STAGING_DIR`, else in the system temporary
  directory, and removes the files when the write ends.
- With `huggingface_hub` 1.x, `HfApi.batch_bucket_files` does not report files that the bucket
  rejected. The backend then lists the destination once after the upload (one extra listing of the
  file's string prefix, or of the base prefix, per write) and raises if a file is missing or has
  another size. `huggingface_hub` 2.x reports rejected files itself, and no listing is made.

**Memory.** In laptop runs with outputs up to 4.5 GB, the `"stream"` backend used 0.5–1.8 GB more
memory than Polars alone. A bound on its memory use is not proven. See
[Measurements](#measurements).

**What a failure leaves behind.** With both backends, no file is registered in the bucket before the
Polars sink has finished without an error. If the query fails or is interrupted, the destination is
unchanged: an existing object keeps its content, and no partial or empty file appears. An upload
that fails before the first registration request also leaves the destination unchanged. A failure
of a registration request does not: see the list below.

A failed registration or delete request raises `polars_hf.BucketRegistrationError` (a
`RuntimeError`) with both backends, and so does a failed upload of the `"staged"` backend (an HTTP
error, a timeout or a connection error; the original error is the `__cause__`). Its `failures`
attribute lists the operations the bucket rejected. After a timeout or a connection error the
request may or may not have been applied: the message says so, and a listing of the destination
shows the state. An error of the `hf_xet` upload itself is raised as `hf_xet` reports it.

The write is **not transactional**, because the bucket API has no transactions:

- Files are registered in requests of at most 1,000 operations. If a write of more than 1,000 files
  fails between two requests, the files of the earlier requests stay in the bucket.
- If the bucket rejects single files of a request (a 200 answer with `failed` entries), it still
  applies the other files of that request, also in a write of fewer than 1,000 files. `sink_bucket`
  then raises an error that lists the rejected paths.
- `mode="overwrite"` deletes the stale files only after all new files are registered, in separate
  requests. If the process stops or a request fails between the registration and the end of the
  deletion, the prefix holds the new files and the remaining stale files, and (for a failed
  request) the error is raised.
- A reader can see a part of the new files while the requests are in progress.

## Measurements

Peak RSS of one process, one run each, on one macOS laptop, with uncompressed incompressible data.
"Polars alone" is the same query written to a file object that discards the bytes.

| Output | Polars alone | `"staged"` | `"stream"` |
| --- | --- | --- | --- |
| 1 file, 2.2 GB | 1.05 GB | 1.58 GB | 1.78 GB |
| 1 file, 4.5 GB | 1.56 GB | not measured | 2.99 GB |
| 16 files, 2.2 GB | 1.80 GB | 1.51 GB | 3.56 GB |
| 2,000 files, 2.2 GB | 3.35 GB | 3.34 GB (13 s) | 4.90 GB (62 s) |

These numbers describe these runs only; they are not a bound.
