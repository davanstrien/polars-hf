"""Shared fixtures: staging isolation, the offline fake Hub, staging buckets.

Endpoint isolation
------------------
``huggingface_hub`` reads ``HF_ENDPOINT`` once, at import time. This module is
imported by pytest before any test module, so the three assignments below run
before ``huggingface_hub`` (or ``polars_hf``) is imported and force the whole
test process onto the Hub CI staging instance with its public CI token:

* the production Hub is never contacted, so tests cannot write to it;
* ``HF_TOKEN`` in the environment has priority over a cached login, and
  ``HF_TOKEN_PATH`` points at a file that does not exist, so a developer's
  real token is never read and never sent to staging.

The values are assigned, not ``setdefault``-ed: a developer's own
``HF_ENDPOINT`` / ``HF_TOKEN`` must not leak into a test run. The check after
the import stops the session if another plugin imported ``huggingface_hub``
first (the environment would then be too late to have an effect).

Offline tests use the ``fake_hub`` fixture, which points the endpoint at a
local server for the duration of one test.
"""

from __future__ import annotations

import os

STAGING_ENDPOINT = "https://hub-ci.huggingface.co"
# Public token of the staging CI user; only valid on hub-ci.huggingface.co.
STAGING_TOKEN = "hf_HubCITokenXXXXXXXXXXXXXXXXXXXXX"

os.environ["HF_ENDPOINT"] = STAGING_ENDPOINT
os.environ["HF_TOKEN"] = STAGING_TOKEN
os.environ["HF_TOKEN_PATH"] = os.path.join(os.sep, "nonexistent", "polars-hf-tests")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import time  # noqa: E402
import uuid  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from typing import TypeVar  # noqa: E402

import polars as pl  # noqa: E402
import pytest  # noqa: E402
from fakehub import FakeHub  # noqa: E402
from huggingface_hub import HfApi, HfFileSystem, constants  # noqa: E402
from hypothesis import settings  # noqa: E402

if constants.ENDPOINT != STAGING_ENDPOINT:  # pragma: no cover - safety stop
    pytest.exit(
        "huggingface_hub was imported before tests/conftest.py could force the "
        f"staging endpoint (ENDPOINT={constants.ENDPOINT!r}); refusing to run.",
        returncode=3,
    )

# Derandomized: every run executes the same examples, so the suite (and its
# strict xfail properties) cannot flake. No deadline: the first example of a
# test pays one-off costs (imports, a new HTTP client).
settings.register_profile("polars-hf", derandomize=True, deadline=None, max_examples=60)
settings.load_profile("polars-hf")

T = TypeVar("T")

# Same transient-error pattern as huggingface_hub's own staging test runs.
_STAGING_RERUN_PATTERNS = [
    "OSError",
    "FileNotFoundError",
    "Timeout",
    "HTTPError.*409",
    "HTTPError.*502",
    "HTTPError.*503",
    "HTTPError.*504",
    "HTTPStatusError.*50[234]",
    "ComputeError",
]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Rerun staging tests on transient errors; never rerun offline tests."""
    for item in items:
        if item.get_closest_marker("staging") is None:
            continue
        item.add_marker(
            pytest.mark.flaky(
                reruns=5, reruns_delay=2, only_rerun=_STAGING_RERUN_PATTERNS
            )
        )


# ---- offline fake ----------------------------------------------------------

FAKE_BUCKET = "fake-user/fake-bucket"


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHub]:
    """A local fake Hub; ``huggingface_hub`` talks to it for one test.

    ``HfApi`` and ``HfFileSystem`` read ``constants.ENDPOINT`` when they are
    instantiated, so patching the constant is enough. ``HfFileSystem``
    instances are cached by fsspec: the cache is cleared on both sides of the
    test so no instance (and no directory listing cache) leaks between tests.
    Uploads are stored in the fake bucket (see ``FakeHub.patch_uploads``).
    """
    with FakeHub() as hub:
        hub.create_bucket(FAKE_BUCKET)
        monkeypatch.setattr(constants, "ENDPOINT", hub.endpoint)
        hub.patch_uploads(monkeypatch)
        HfFileSystem.clear_instance_cache()
        yield hub
        HfFileSystem.clear_instance_cache()


@pytest.fixture
def fake_bucket(fake_hub: FakeHub) -> str:
    """The id of an empty bucket on the fake Hub."""
    return FAKE_BUCKET


# ---- staging ---------------------------------------------------------------


def _staging_retry(action: Callable[[], T], attempts: int = 4) -> T:
    """Run ``action``, retrying the transient errors staging is known for."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - re-raised below
            last_error = error
            time.sleep(1 + attempt)
    assert last_error is not None
    raise last_error


def _create_staging_bucket(api: HfApi) -> str:
    name = f"polars-hf-test-{uuid.uuid4().hex[:12]}"
    url = _staging_retry(lambda: api.create_bucket(name, private=True, exist_ok=True))
    return url.bucket_id


def _delete_staging_bucket(api: HfApi, bucket_id: str) -> None:
    try:
        _staging_retry(lambda: api.delete_bucket(bucket_id, missing_ok=True))
    except Exception:  # noqa: BLE001 - staging buckets are disposable
        pass


@pytest.fixture(scope="session")
def staging_api() -> HfApi:
    """An ``HfApi`` bound explicitly to the staging endpoint and CI token."""
    return HfApi(endpoint=STAGING_ENDPOINT, token=STAGING_TOKEN)


@pytest.fixture
def staging_bucket(staging_api: HfApi) -> Iterator[str]:
    """A new, uniquely named, empty bucket on staging; deleted afterwards."""
    bucket_id = _create_staging_bucket(staging_api)
    yield bucket_id
    _delete_staging_bucket(staging_api, bucket_id)


def _parquet_bytes(df: pl.DataFrame) -> bytes:
    import io

    buffer = io.BytesIO()
    df.write_parquet(buffer)
    return buffer.getvalue()


def _bench_frame(run: int, rows: int = 100_000) -> pl.DataFrame:
    start = run * rows
    ids = pl.int_range(start, start + rows, eager=True)
    return pl.DataFrame(
        {
            "id": ids,
            "category": (ids % 7).cast(pl.String),
            "value": (ids % 1000).cast(pl.Float64) / 10.0,
            "text": "row " + ids.cast(pl.String),
        }
    )


def staging_seed_files() -> dict[str, pl.DataFrame]:
    """Synthetic contents of the read-only staging bucket, keyed by path.

    * ``smoke/filtered.parquet`` — one 500 x 4 file;
    * ``smoke/bench/run_{0,1,2}.parquet`` — three files with the same schema
      (``id``, ``category``, ``value``, ``text``), 100 000 rows each;
    * ``smoke/extra.parquet`` — a schema that differs from ``filtered``, so
      ``smoke/*.parquet`` is a mixed-schema glob.
    """
    filtered = pl.DataFrame(
        {
            "id": range(500),
            "label": [i % 5 for i in range(500)],
            "score": [i / 500 for i in range(500)],
            "name": [f"item-{i}" for i in range(500)],
        }
    )
    extra = pl.DataFrame({"id": range(20), "note": [f"n{i}" for i in range(20)]})
    files = {"smoke/filtered.parquet": filtered, "smoke/extra.parquet": extra}
    for run in range(3):
        files[f"smoke/bench/run_{run}.parquet"] = _bench_frame(run)
    return files


@pytest.fixture(scope="module")
def staging_read_bucket(staging_api: HfApi) -> Iterator[str]:
    """A staging bucket seeded once per module; tests must not write to it."""
    bucket_id = _create_staging_bucket(staging_api)
    add = []
    for path, frame in staging_seed_files().items():
        add.append((_parquet_bytes(frame), path))
    try:
        _staging_retry(lambda: staging_api.batch_bucket_files(bucket_id, add=add))
        yield bucket_id
    finally:
        _delete_staging_bucket(staging_api, bucket_id)
