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
# A developer's HF_HUB_OFFLINE=1 would make every Hub call fail before it
# reaches the fake or staging.
os.environ["HF_HUB_OFFLINE"] = "0"
# The fake Hub listens on 127.0.0.1: httpx and the Polars HTTP client must not
# send those requests to a proxy configured in the environment.
for _name in ("NO_PROXY", "no_proxy"):
    _hosts = ["127.0.0.1", "localhost"]
    if os.environ.get(_name):
        _hosts.append(os.environ[_name])
    os.environ[_name] = ",".join(_hosts)

import re  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from typing import TypeVar  # noqa: E402

import httpx  # noqa: E402
import huggingface_hub  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402
from fakehub import FakeHub  # noqa: E402
from huggingface_hub import HfApi, HfFileSystem, constants  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402
from hypothesis import settings  # noqa: E402

# If a pytest plugin imported huggingface_hub before this module, the constant
# (and any HfApi / HfFileSystem object the plugin already created) keeps the
# production endpoint, whatever the environment says now. Abort the session
# instead of running tests against production.
if constants.ENDPOINT != STAGING_ENDPOINT:  # pragma: no cover - safety stop
    pytest.exit(
        "huggingface_hub was imported before tests/conftest.py could force the "
        f"staging endpoint (ENDPOINT={constants.ENDPOINT!r}); refusing to run.",
        returncode=3,
    )

# Default profile, derandomized: every run executes the same examples, so the
# suite (and its strict xfail properties) cannot flake. No deadline: the first
# example of a test pays one-off costs (imports, a new HTTP client).
# HYPOTHESIS_PROFILE=random selects new examples on every run (and makes
# --hypothesis-seed effective); the weekly CI run uses it.
settings.register_profile("polars-hf", derandomize=True, deadline=None, max_examples=60)
settings.register_profile("random", derandomize=False, deadline=None, max_examples=60)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "polars-hf"))

T = TypeVar("T")

HUB_MAJOR = int(huggingface_hub.__version__.split(".")[0])
# The HTTP library behind the huggingface_hub session: httpx2 for
# huggingface_hub 2.x, httpx for 1.x. Mock transports, requests and responses
# of that session come from this module.
if HUB_MAJOR >= 2:
    import httpx2 as hub_httpx
else:
    hub_httpx = httpx

POLARS_MAJOR = int(pl.__version__.split(".")[0])

# A query parameter that holds a signature or a credential of a presigned URL
# (CloudFront, S3 and Xet forms), plain or percent-encoded.
_SIGNED_PARAMETER = re.compile(
    r"(signature|policy|key-pair-id|x-amz-[a-z0-9-]+|x-xet-[a-z0-9-]+)(=|%3d)",
    re.IGNORECASE,
)


def assert_no_signed_url(text: str | bytes, fake_hub: FakeHub) -> None:
    """``text`` holds no presigned URL of the fake Hub and no part of one.

    The check does not depend on the names the fake uses: the host of the cdn
    server must not appear, and no signature-like query parameter.
    """
    if isinstance(text, bytes):
        text = text.decode("latin-1")
    cdn_host = fake_hub.cdn_endpoint.partition("://")[2]
    assert cdn_host not in text
    assert _SIGNED_PARAMETER.search(text) is None


def exception_chain(error: BaseException) -> list[BaseException]:
    """``error`` and every exception linked to it as cause or context."""
    chain: list[BaseException] = []
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or any(current is seen for seen in chain):
            continue
        chain.append(current)
        pending.append(current.__cause__)
        pending.append(current.__context__)
    return chain


@contextmanager
def raises_at_collect(
    expected: type[BaseException],
    match: str | None = None,
    *,
    schema_step: bool = False,
) -> Iterator[pytest.ExceptionInfo]:
    """Expect ``expected`` from a query of a collect-time ``scan_bucket``.

    Polars 1.x wraps an exception that an IO source raises in a
    ``ComputeError`` whose message holds the type name and the message;
    Polars 2 raises the exception itself. Every version wraps an exception of
    the schema read (``schema_step=True``). ``match`` is searched in the
    message in both cases.
    """
    wrapped = schema_step or POLARS_MAJOR < 2
    raised = pl.exceptions.ComputeError if wrapped else expected
    with pytest.raises(raised) as error:
        yield error
    message = str(error.value)
    if wrapped:
        assert f"{expected.__name__}: " in message
    if match is not None:
        assert re.search(match, message), message


# Transient staging errors only: HTTP 409/502/503/504 and timeouts. Anything
# else (a missing file, a read error) can be a real read-after-write bug and
# must fail the test.
_STAGING_TRANSIENT_STATUS = (409, 502, 503, 504)
_STAGING_RERUN_PATTERNS = [
    "Timeout",
    "HTTPError.*(409|502|503|504)",
    "HTTPStatusError.*(409|502|503|504)",
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
# Module constants of polars_hf.read that set the backoff without server hint.
RETRY_BACKOFF_CONSTANTS = ("_RETRY_BASE_WAIT", "_RETRY_MAX_BACKOFF")
# On staging the CI account shares its rate limits with other projects: a
# rate-limited test must fail within seconds, not sleep until the reset. A
# wait is made only if it ends before this deadline.
STAGING_SCAN_DEADLINE = 30.0


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHub]:
    """A local fake Hub; ``huggingface_hub`` talks to it for one test.

    ``HfApi`` and ``HfFileSystem`` read ``constants.ENDPOINT`` when they are
    instantiated, so patching the constant is enough. ``HfFileSystem``
    instances are cached by fsspec: the cache is cleared on both sides of the
    test so no instance (and no directory listing cache) leaks between tests.
    Uploads are stored in the fake bucket (see ``FakeHub.patch_uploads``).
    """
    with FakeHub(token=STAGING_TOKEN) as hub:
        hub.create_bucket(FAKE_BUCKET)
        monkeypatch.setattr(constants, "ENDPOINT", hub.endpoint)
        hub.patch_uploads(monkeypatch)
        fast_resolve_retries(monkeypatch)
        HfFileSystem.clear_instance_cache()
        yield hub
        HfFileSystem.clear_instance_cache()


@pytest.fixture
def fake_bucket(fake_hub: FakeHub) -> str:
    """The id of an empty bucket on the fake Hub."""
    return FAKE_BUCKET


def fast_resolve_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the backoff between two attempts of a Hub request negligible.

    The number of attempts is unchanged, and a wait that the server asks for
    (rate-limit reset, ``Retry-After``) is not shortened.
    """
    from polars_hf import read

    # raising=False: a renamed constant must not break every test that uses
    # the fake Hub. test_retry.py checks that these names exist.
    for name in RETRY_BACKOFF_CONSTANTS:
        monkeypatch.setattr(read, name, 0.001, raising=False)


@pytest.fixture(autouse=True)
def _no_mode_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mode or an acknowledgement of the developer's shell must not reach a test."""
    monkeypatch.delenv("POLARS_HF_RESOLVE", raising=False)
    monkeypatch.delenv("POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN", raising=False)


@pytest.fixture
def allow_signed_urls_in_plan(
    monkeypatch: pytest.MonkeyPatch, _no_mode_from_the_environment: None
) -> None:
    """Acknowledge ``resolve="now"`` for one test.

    Tests use that mode as the reference (the plain native scan) and must ask
    for this fixture. A test without it cannot pass through the gated mode.
    """
    monkeypatch.setenv("POLARS_HF_ALLOW_SIGNED_URLS_IN_PLAN", "1")


@pytest.fixture(autouse=True)
def _no_real_sleep(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline tests: a retry wait of the read path must not sleep for real.

    Waits of the shortened backoff pass; a longer one (a server hint, or a
    backoff that is not shortened) fails the test at once. Tests of the wait
    policy replace the clock (``test_retry.py``).
    """
    if request.node.get_closest_marker("staging") is not None:
        return
    import types

    from polars_hf import read

    def sleep(seconds: float) -> None:
        if seconds > 0.01:
            raise AssertionError(f"an offline test would sleep for {seconds} s")
        time.sleep(seconds)

    clock = types.SimpleNamespace(monotonic=time.monotonic, sleep=sleep)
    monkeypatch.setattr(read, "time", clock)


@contextmanager
def hub_session(handler: Callable) -> Iterator[None]:
    """Answer every request of the ``huggingface_hub`` session with ``handler``.

    ``handler`` receives the request (``hub_httpx.Request``) and returns a
    ``hub_httpx.Response``. The client follows redirects by default, like the
    real session, so a test sees it when the code under test forgets to turn
    that off. The default session is restored on exit.
    """
    # default_client_factory is private; set_client_factory and close_session
    # are public. Verified on huggingface_hub 1.12.0, 1.17.0 and 2.0.0.
    from huggingface_hub.utils._http import default_client_factory

    def factory():
        transport = hub_httpx.MockTransport(handler)
        return hub_httpx.Client(transport=transport, follow_redirects=True)

    huggingface_hub.set_client_factory(factory)
    try:
        yield
    finally:
        huggingface_hub.set_client_factory(default_client_factory)


# ---- staging ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _short_staging_waits(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging tests: a short limit on the retry waits of one ``scan_bucket``."""
    if request.node.get_closest_marker("staging") is None:
        return
    from polars_hf import read

    monkeypatch.setattr(read, "_SCAN_DEADLINE", STAGING_SCAN_DEADLINE, raising=False)


# Collected with getattr: the exception must not be required of every httpx
# release that a resolver can pick for the dev group.
_TIMEOUT_ERRORS: tuple[type[Exception], ...] = ()
for _module in (httpx, hub_httpx):
    _timeout_error = getattr(_module, "TimeoutException", None)
    if _timeout_error is not None and _timeout_error not in _TIMEOUT_ERRORS:
        _TIMEOUT_ERRORS = (*_TIMEOUT_ERRORS, _timeout_error)


def _is_transient_staging_error(error: Exception) -> bool:
    if isinstance(error, _TIMEOUT_ERRORS):
        return True
    if isinstance(error, HfHubHTTPError):
        return error.response.status_code in _STAGING_TRANSIENT_STATUS
    return False


def _staging_retry(action: Callable[[], T], attempts: int = 4) -> T:
    """Run ``action``; retry transient staging errors, raise everything else."""
    for attempt in range(attempts):
        try:
            return action()
        except Exception as error:
            is_last_attempt = attempt == attempts - 1
            if is_last_attempt or not _is_transient_staging_error(error):
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def _create_staging_bucket(api: HfApi) -> str:
    name = f"polars-hf-test-{uuid.uuid4().hex[:12]}"
    url = _staging_retry(lambda: api.create_bucket(name, private=True, exist_ok=True))
    return url.bucket_id


def _delete_staging_bucket(api: HfApi, bucket_id: str) -> None:
    # missing_ok swallows "not found" only; any other error fails the teardown.
    _staging_retry(lambda: api.delete_bucket(bucket_id, missing_ok=True))


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
    try:
        # Inside the try: an error while the seed data is built must still
        # delete the bucket.
        add = []
        for path, frame in staging_seed_files().items():
            add.append((_parquet_bytes(frame), path))
        _staging_retry(lambda: staging_api.batch_bucket_files(bucket_id, add=add))
        yield bucket_id
    finally:
        _delete_staging_bucket(staging_api, bucket_id)
