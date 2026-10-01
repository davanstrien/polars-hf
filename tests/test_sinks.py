"""Unit tests of the sink backends: selection and object names (no network)."""

from __future__ import annotations

import datetime
import os
import tempfile

import huggingface_hub
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import polars_hf as plhf
from polars_hf import _sinks

# ---- backend selection -----------------------------------------------------


def _hub_version() -> tuple[int, int]:
    major, minor = huggingface_hub.__version__.split(".")[:2]
    return int(major), int(minor)


def test_xet_backend_availability_matches_the_installed_versions() -> None:
    # The Xet session helpers the backend needs exist from huggingface_hub
    # 1.19.0, which also requires an hf_xet with streaming uploads.
    pytest.importorskip("hf_xet")

    reason = _sinks.xet_unavailable_reason()

    if _hub_version() >= (1, 19):
        assert reason is None
    else:
        assert reason is not None
        assert huggingface_hub.__version__ in reason


@pytest.fixture
def no_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_sinks.BACKEND_ENV_VAR, raising=False)


def _set_xet(monkeypatch: pytest.MonkeyPatch, reason: str | None) -> None:
    monkeypatch.setattr(_sinks, "xet_unavailable_reason", lambda: reason)


def test_default_backend_is_xet_when_available(
    monkeypatch: pytest.MonkeyPatch, no_backend_env: None
) -> None:
    _set_xet(monkeypatch, None)

    assert _sinks.resolve_backend_name(None) == "xet"


def test_default_backend_falls_back_to_hub(
    monkeypatch: pytest.MonkeyPatch, no_backend_env: None
) -> None:
    _set_xet(monkeypatch, "hf_xet is not installed")

    assert _sinks.resolve_backend_name(None) == "hub"
    assert _sinks.resolve_backend_name("hub") == "hub"


@pytest.mark.parametrize("how", ["argument", "environment"])
def test_explicit_xet_backend_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, no_backend_env: None, how: str
) -> None:
    _set_xet(monkeypatch, "huggingface_hub 1.12.0 has no Xet session helpers")
    requested = "xet" if how == "argument" else None
    if how == "environment":
        monkeypatch.setenv(_sinks.BACKEND_ENV_VAR, "xet")

    with pytest.raises(RuntimeError) as error:
        _sinks.resolve_backend_name(requested)

    message = str(error.value)
    assert "huggingface_hub>=1.19" in message
    assert "hf_xet>=1.5.1" in message
    assert "huggingface_hub 1.12.0 has no Xet session helpers" in message
    assert "backend='hub'" in message


def test_environment_variable_selects_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_xet(monkeypatch, None)
    monkeypatch.setenv(_sinks.BACKEND_ENV_VAR, "hub")

    assert _sinks.resolve_backend_name(None) == "hub"
    # The argument has priority over the environment.
    assert _sinks.resolve_backend_name("xet") == "xet"


def test_empty_environment_variable_means_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_xet(monkeypatch, None)
    monkeypatch.setenv(_sinks.BACKEND_ENV_VAR, "")

    assert _sinks.resolve_backend_name(None) == "xet"


def test_unknown_backend_is_rejected(no_backend_env: None) -> None:
    with pytest.raises(ValueError, match="unknown sink backend 's3'"):
        plhf.sink_bucket(
            pl.DataFrame({"a": [1]}), "hf://buckets/ns/name/a.parquet", backend="s3"
        )


# ---- object names ----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (None, "__HIVE_DEFAULT_PARTITION__"),
        ("", ""),
        ("plain-._~", "plain-._~"),
        ("a/b", "a%2Fb"),
        ("a b", "a%20b"),
        ("x=y", "x%3Dy"),
        ("50%", "50%25"),
        ("12:30", "12%3A30"),
        ("tab\there", "tab%09here"),
        ("del\x7f", "del%7F"),
        ("é", "%C3%A9"),
        ("日本", "%E6%97%A5%E6%9C%AC"),
        ("a\\b?c#d&e+f", "a\\b?c#d&e+f"),
    ],
)
def test_hive_encode(value: str | None, encoded: str) -> None:
    assert _sinks.hive_encode(value) == encoded


class _Discard:
    """A write-only file object that drops the bytes."""

    def write(self, data: bytes) -> int:
        return len(data)

    def flush(self) -> None:
        pass


def _provider_names(df: pl.DataFrame, **partition: object) -> set[str]:
    """The names the xet backend's provider builds for a partitioned sink."""
    names = set()

    def provider(args: object) -> _Discard:
        names.add(
            _sinks.partition_file_name(
                args.partition_keys, args.index_in_partition, "parquet"
            )
        )
        return _Discard()

    target = pl.PartitionBy("unused", file_path_provider=provider, **partition)
    df.lazy().sink_parquet(target)
    return names


def _native_names(df: pl.DataFrame, **partition: object) -> set[str]:
    """The names Polars itself writes for the same sink to a local directory."""
    names = set()
    with tempfile.TemporaryDirectory() as directory:
        df.lazy().sink_parquet(pl.PartitionBy(directory, **partition))
        for root, _, files in os.walk(directory):
            for name in files:
                relative = os.path.relpath(os.path.join(root, name), directory)
                names.add(relative.replace(os.sep, "/"))
    return names


# Any text: control characters, "/", "%", non-ASCII. Short, so the encoded
# directory name stays below the 255-byte limit of local file systems.
_text_keys = st.one_of(st.none(), st.text(max_size=8))
_key_columns = st.one_of(
    st.lists(_text_keys, min_size=1, max_size=5).map(
        lambda values: pl.Series(values, dtype=pl.String)
    ),
    st.lists(st.one_of(st.none(), st.booleans()), min_size=1, max_size=5).map(
        lambda values: pl.Series(values, dtype=pl.Boolean)
    ),
    st.lists(
        st.one_of(st.none(), st.integers(-(2**63), 2**63 - 1)), min_size=1, max_size=5
    ).map(lambda values: pl.Series(values, dtype=pl.Int64)),
    st.lists(
        st.one_of(st.none(), st.floats(allow_nan=True, width=64)),
        min_size=1,
        max_size=5,
    ).map(lambda values: pl.Series(values, dtype=pl.Float64)),
    st.lists(
        st.one_of(
            st.none(),
            st.dates(datetime.date(1, 1, 1), datetime.date(9999, 12, 31)),
        ),
        min_size=1,
        max_size=5,
    ).map(lambda values: pl.Series(values, dtype=pl.Date)),
    st.lists(
        st.one_of(
            st.none(),
            st.datetimes(
                datetime.datetime(1900, 1, 1), datetime.datetime(2200, 1, 1)
            ),
        ),
        min_size=1,
        max_size=5,
    ).map(lambda values: pl.Series(values, dtype=pl.Datetime("us"))),
)


@settings(max_examples=150)
@given(first=_key_columns, second=_key_columns, max_rows=st.sampled_from([None, 1]))
def test_partition_file_names_equal_the_native_layout(
    first: pl.Series, second: pl.Series, max_rows: int | None
) -> None:
    height = min(len(first), len(second))
    df = pl.DataFrame(
        {
            "k1": first.head(height),
            "k 2": second.head(height),
            "n": range(height),
        }
    )
    partition: dict[str, object] = {"key": ["k1", "k 2"]}
    if max_rows is not None:
        partition["max_rows_per_file"] = max_rows

    assert _provider_names(df, **partition) == _native_names(df, **partition)


def test_partition_file_index_is_hexadecimal() -> None:
    df = pl.DataFrame({"g": ["a"] * 300, "n": range(300)})
    partition = {"key": "g", "max_rows_per_file": 1}

    names = _provider_names(df, **partition)

    assert names == _native_names(df, **partition)
    assert "g=a/0000012b.parquet" in names


def test_partition_file_names_without_a_key() -> None:
    df = pl.DataFrame({"n": range(5)})

    names = _provider_names(df, max_rows_per_file=2)

    assert names == _native_names(df, max_rows_per_file=2)
    assert names == {"00000000.parquet", "00000001.parquet", "00000002.parquet"}
