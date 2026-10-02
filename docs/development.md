# Development

```bash
uv sync
uv run ruff check .
uv run pytest                 # offline suite (the default)
uv run pytest -m staging      # live tests against the Hub CI staging instance
```

`addopts` in `pyproject.toml` deselects the staging tests and the tests marked `slow` by default.
The `slow` tests are offline tests that wait for the retries of Polars' HTTP client (several
seconds each); `uv run pytest -m slow` runs them, and so does the weekly CI run. A command such as
`uv run pytest tests/test_read.py` therefore selects nothing: add `-m staging`.

**Offline tests** need no network and no token. `tests/fakehub.py` runs a local fake Hub: one HTTP
server for the bucket API and the `resolve` redirect, and a second one (another origin) that serves
the "presigned" URLs with range requests. It records every request and the bytes it serves, and can
be scripted to fail (`429`, `503`, `403`, ...), and to refuse the presigned URLs it has made so far
(`expire_signed_urls()`), like the real CDN after an hour, or to break a connection
(`action="reset"`, `action="truncate"`). File uploads do not go over HTTP; the fake has one
seam per sink backend. For the `"staged"` backend, a patched `HfApi._batch_bucket_files` stores the
files in memory, so the client-side chunking of the public method stays real. For the `"stream"`
backend, an in-memory object replaces the `hf_xet` upload commit, and the backend's own
registration request goes to the fake's `POST /api/buckets/{id}/batch` route. The fake also reports
the `"stream"` backend as available, so the offline suite runs both backends with every supported
`huggingface_hub`. The real `hf_xet` upload is covered by the staging tests only. Use the
`fake_hub` / `fake_bucket` fixtures. `tests/sinks.py` is the one place that names the write design
(backends and failure switches); write tests go through its helpers.

**Staging tests** (`-m staging`) do real round-trips against `https://hub-ci.huggingface.co`, the
instance `huggingface_hub` uses for its own tests. `tests/conftest.py` sets `HF_ENDPOINT` and
`HF_TOKEN` to the staging endpoint and its public CI token *before* `huggingface_hub` is imported,
so the test process never talks to `huggingface.co` and never reads your own token. Each test
creates a uniquely named bucket and deletes it afterwards. `tests/test_staging_sinks.py` runs the
same write scenarios for both sink backends; the `stream` cases are skipped when the installed
`huggingface_hub` is older than 1.19. `test_local_disk_use` measures the growth of the temporary
directory during a write (16 MB by default; `POLARS_HF_DISK_TEST_MB=300 uv run pytest -m staging -k
test_local_disk_use -s` prints a larger measurement).
`test_big_write_is_identical_with_both_backends` uploads about 300 MB with each backend and compares
the sizes and Xet hashes of all objects; it runs only with `POLARS_HF_BIG_STAGING=1`, which the weekly
CI run sets. Staging can answer
`409`/`502`/`503`/`504` or time out; a test that fails with one of these is rerun automatically
(`pytest-rerunfailures`). Other failures are not rerun.

**Property tests** (`tests/test_properties.py`) use [Hypothesis](https://hypothesis.readthedocs.io)
with a derandomized profile, so every run executes the same examples. For a randomized pass, select
the `random` profile (`--hypothesis-seed` has an effect only with this profile):

```bash
HYPOTHESIS_PROFILE=random uv run pytest tests/test_properties.py
HYPOTHESIS_PROFILE=random uv run pytest tests/test_properties.py --hypothesis-seed=1234
```

**Known bugs** go in `tests/test_known_bugs.py` as `@pytest.mark.xfail(strict=True)` tests (none is
open at the moment; the tests there are regression tests of fixed bugs). Such a test
asserts the behaviour we *want* and fails today for the reason in its `reason=`, so it is reported
as `xfailed`. Because the marker is strict, a test that starts to pass turns the suite red
(`XPASS(strict)`): the pull request that fixes a bug must remove the marker in the same change. To
see the current failure of such a test, run it with `--runxfail`.

CI runs the offline suite on Python 3.10–3.14 with the locked dependencies, and again with the
lowest supported direct dependencies, the latest releases, and the newest Polars 2 pre-release
(that last job is allowed to fail). The staging suite runs on pull requests and on pushes to `main`. A weekly
scheduled run repeats all jobs with the `random` Hypothesis profile.
