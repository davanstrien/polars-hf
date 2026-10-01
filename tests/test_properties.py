"""Hypothesis property tests (offline).

The ``polars-hf`` Hypothesis profile (``conftest.py``) is derandomized, so every
run executes the same examples and the ``xfail(strict=True)`` properties fail
for the same reason every time.
"""

from __future__ import annotations

import string
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
import pytest
from conftest import hub_httpx, hub_session
from fakehub import FakeHub
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes
from sinks import ALL_SINKS, sink_streamed

import polars_hf as plhf
from polars_hf._uri import BucketPath, parse_bucket_uri
from polars_hf.read import _MAX_REDIRECT_HOPS, _Budget, _signed_url

# ---- URI parsing -----------------------------------------------------------

_ID_ALPHABET = string.ascii_letters + string.digits + "-_."
# One path segment: any text without "/" (which ends the segment). "@" is a
# normal character below the bucket name. ".." is a rejected segment and is
# generated separately.
_segment = st.text(
    alphabet=st.characters(
        exclude_characters="/", exclude_categories=("Cs",), max_codepoint=0x2FFF
    ),
    min_size=1,
    max_size=12,
).filter(lambda segment: segment != "..")
_id_part = st.text(alphabet=_ID_ALPHABET, min_size=1, max_size=12)
# The parser rejects a URI that ends with whitespace, so a valid path does not.
_path = (
    st.lists(_segment, min_size=0, max_size=4)
    .map("/".join)
    .filter(lambda path: path == path.rstrip())
)


@given(namespace=_id_part, name=_id_part, path=_path)
def test_valid_uri_round_trips(namespace: str, name: str, path: str) -> None:
    uri = f"hf://buckets/{namespace}/{name}"
    if path:
        uri = f"{uri}/{path}"

    bp = parse_bucket_uri(uri)

    assert bp == BucketPath(bucket_id=f"{namespace}/{name}", path=path)
    assert f"hf://{bp.fs_path}" == uri
    assert bp.is_glob == any(c in path for c in "*?[")


@given(namespace=_id_part, name=_id_part, path=_path.filter(lambda path: path != ""))
def test_trailing_slash_names_the_same_path(
    namespace: str, name: str, path: str
) -> None:
    bp = parse_bucket_uri(f"hf://buckets/{namespace}/{name}/{path}/")

    assert bp == BucketPath(bucket_id=f"{namespace}/{name}", path=f"{path}/")


@given(namespace=_id_part, name=_id_part, before=_path, stem=_segment)
def test_uri_with_at_sign_in_path_round_trips(
    namespace: str, name: str, before: str, stem: str
) -> None:
    file_name = f"{stem}@example.com.parquet"
    path = f"{before}/{file_name}" if before else file_name

    bp = parse_bucket_uri(f"hf://buckets/{namespace}/{name}/{path}")

    assert bp == BucketPath(bucket_id=f"{namespace}/{name}", path=path)


@given(
    namespace=_id_part,
    name=_id_part,
    revision=_id_part,
    path=_path,
    in_namespace=st.booleans(),
)
def test_at_sign_in_the_bucket_id_is_rejected(
    namespace: str, name: str, revision: str, path: str, in_namespace: bool
) -> None:
    if in_namespace:
        uri = f"hf://buckets/{namespace}@{revision}/{name}/{path}"
    else:
        uri = f"hf://buckets/{namespace}/{name}@{revision}/{path}"

    with pytest.raises(ValueError, match="do not support @revision"):
        parse_bucket_uri(uri)


# A path that the parser must reject: (segments, text at the end of the URI).
_bad_segment = st.sampled_from(["", ".."])
_trailing_whitespace = st.text(alphabet=" \t\n\r\u00a0\u2003", min_size=1, max_size=3)


@given(
    namespace=_id_part,
    name=_id_part,
    before=st.lists(_segment, max_size=2),
    bad=_bad_segment,
    after=st.lists(_segment, min_size=1, max_size=2),
)
def test_empty_and_dot_dot_segments_are_rejected(
    namespace: str, name: str, before: list[str], bad: str, after: list[str]
) -> None:
    # The bad segment is never the last one: a trailing "" is the allowed
    # trailing slash. A trailing ".." is covered below.
    path = "/".join([*before, bad, *after])

    with pytest.raises(ValueError, match="path segment"):
        parse_bucket_uri(f"hf://buckets/{namespace}/{name}/{path}".rstrip())


@given(namespace=_id_part, name=_id_part, before=st.lists(_segment, max_size=2))
def test_dot_dot_as_last_segment_is_rejected(
    namespace: str, name: str, before: list[str]
) -> None:
    path = "/".join([*before, ".."])

    for suffix in ("", "/"):
        with pytest.raises(ValueError, match="path segment"):
            parse_bucket_uri(f"hf://buckets/{namespace}/{name}/{path}{suffix}")


@given(namespace=_id_part, name=_id_part, path=_path, whitespace=_trailing_whitespace)
def test_trailing_whitespace_is_rejected(
    namespace: str, name: str, path: str, whitespace: str
) -> None:
    uri = f"hf://buckets/{namespace}/{name}"
    if path:
        uri = f"{uri}/{path}"

    with pytest.raises(ValueError, match="ends with whitespace"):
        parse_bucket_uri(uri + whitespace)


@given(
    text=st.text(max_size=60), prefix=st.sampled_from(["", "hf://", "hf://buckets/"])
)
def test_arbitrary_text_only_raises_value_error(text: str, prefix: str) -> None:
    try:
        bp = parse_bucket_uri(prefix + text)
    except ValueError:
        return

    # Accepted: the result is self-consistent.
    namespace, _, name = bp.bucket_id.partition("/")
    assert namespace and name and "/" not in name
    assert "@" not in bp.bucket_id
    expected = (
        f"buckets/{bp.bucket_id}/{bp.path}" if bp.path else f"buckets/{bp.bucket_id}"
    )
    assert bp.fs_path == expected
    # No empty or ".." segment; at most one trailing slash.
    segments = bp.path.split("/") if bp.path else []
    if segments and segments[-1] == "":
        segments = segments[:-1]
    assert "" not in segments and ".." not in segments
    assert (prefix + text) == (prefix + text).rstrip()


# ---- redirect resolver -----------------------------------------------------

HUB_ORIGIN = "https://huggingface.co"
RESOLVE = f"{HUB_ORIGIN}/buckets/ns/name/resolve/data.parquet"
AUTH = {"authorization": "Bearer hf_secret"}

# One redirect hop: (kind, Location header). ``same`` hops stay on the Hub
# origin and are followed; ``other`` hops leave it and end the resolution.
_same_origin_hops = st.sampled_from(
    [
        ("same", "/buckets/ns/name/resolve2/data.parquet"),
        ("same", "next/data.parquet"),
        ("same", f"{HUB_ORIGIN}/buckets/ns/renamed/resolve/data.parquet"),
        ("same", "https://HuggingFace.co/buckets/ns/name/resolve3/data.parquet"),
        ("same", "//huggingface.co/protocol-relative/data.parquet"),
        ("same", "https://huggingface.co:443/default-port/data.parquet"),
    ]
)
_other_origin_hops = st.sampled_from(
    [
        ("other", "https://us.aws.cdn.hf.co/xet-bridge-us/abc?Expires=1&Signature=s"),
        ("other", "//us.aws.cdn.hf.co/xet-bridge-us/abc"),
        ("other", "http://huggingface.co/buckets/ns/name/resolve/data.parquet"),
        ("other", "https://huggingface.co:8443/buckets/ns/name/resolve/data.parquet"),
        ("other", "https://huggingface.co:80/buckets/ns/name/resolve/data.parquet"),
        ("other", "https://huggingface.co.evil.example/steal"),
        ("other", "https://evil.example/huggingface.co/steal"),
    ]
)
_hops = st.lists(st.one_of(_same_origin_hops, _other_origin_hops), max_size=8)
_redirect_codes = st.sampled_from([301, 302, 303, 307, 308])


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlparse(url)
    port = parsed.port
    if port is None:
        port = {"http": 80, "https": 443}.get(parsed.scheme)
    return (parsed.scheme, parsed.hostname, port)


@given(hops=_hops, code=_redirect_codes)
def test_signed_url_never_sends_auth_off_origin(
    hops: list[tuple[str, str]], code: int
) -> None:
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) <= len(hops):
            return hub_httpx.Response(
                code, headers={"location": hops[len(seen) - 1][1]}
            )
        return hub_httpx.Response(200, headers={"content-length": "512"})

    result: str | None = None
    error: RuntimeError | None = None
    # The mock session follows redirects by default, like the real one: a
    # request to another origin shows up in ``seen``.
    with hub_session(handler):
        try:
            result = _signed_url(
                RESOLVE, AUTH, uri="hf://buckets/ns/name/x", budget=_Budget("ns/name")
            )
        except RuntimeError as raised:
            error = raised

    # Terminates within the hop limit.
    assert 1 <= len(seen) <= _MAX_REDIRECT_HOPS
    # Only HEAD requests, and every one of them went to the Hub origin: the
    # Authorization header never reaches another origin.
    for request in seen:
        assert request.method == "HEAD"
        assert _origin(str(request.url)) == _origin(RESOLVE)
        assert request.headers["authorization"] == AUTH["authorization"]

    kinds = [kind for kind, _ in hops]
    first_other = kinds.index("other") if "other" in kinds else None
    if first_other is not None and first_other < _MAX_REDIRECT_HOPS:
        # The first off-origin location is returned and never requested.
        assert result is not None
        assert _origin(result) != _origin(RESOLVE)
        assert len(seen) == first_other + 1
    elif len(hops) < _MAX_REDIRECT_HOPS:
        # Only same-origin hops, then a 200 from the Hub itself: that URL
        # needs the token, so it is never returned.
        assert result is None
        assert error is not None and "did not redirect" in str(error)
    else:
        assert result is None
        assert error is not None and "too many redirects" in str(error)


# ---- offline round trips ---------------------------------------------------

# Dtypes that parquet stores and returns without any change of type or value.
_EXACT_DTYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
    pl.Boolean,
    pl.String,
    pl.Binary,
    pl.Date,
]

_round_trip_settings = settings(
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@_round_trip_settings
@given(
    df=dataframes(
        min_cols=1,
        max_cols=4,
        max_size=20,
        allowed_dtypes=_EXACT_DTYPES,
        allow_null=True,
    )
)
def test_single_file_round_trip(
    fake_hub: FakeHub, fake_bucket: str, df: pl.DataFrame
) -> None:
    # The fixtures are shared by all examples: each one overwrites the file.
    uri = f"hf://buckets/{fake_bucket}/prop/single.parquet"

    plhf.sink_bucket(df, uri, mode="overwrite")
    back = plhf.scan_bucket(uri).collect()

    assert_frame_equal(back, df)


def _clear(fake_hub: FakeHub, bucket_id: str, prefix: str) -> None:
    from huggingface_hub import HfApi

    stale = fake_hub.files(bucket_id, prefix)
    if stale:
        HfApi().batch_bucket_files(bucket_id, delete=stale)


# Characters that need care in a URL or a file name. Glob characters are
# generated by the property below; "/" and "\\" are left out: they make a path
# the Hub rejects.
_risky_fragments = st.sampled_from(
    ["a", "Z", "0", " ", "#", "%20", "%", "+", ";", "&", "=", ":", ",", "'", "~"]
    + ["!", "$", "(", ")", "é", "ü", "日本", "\u2603", ".", "-", "_", "@"]
)
_risky_stems = (
    st.lists(_risky_fragments, min_size=1, max_size=8)
    .map("".join)
    .filter(lambda stem: stem.strip(".") != "")
)


@_round_trip_settings
@given(stem=_risky_stems, directory=_risky_stems)
def test_risky_file_names_round_trip(
    fake_hub: FakeHub, fake_bucket: str, stem: str, directory: str
) -> None:
    _clear(fake_hub, fake_bucket, "")
    df = pl.DataFrame({"name": [stem], "n": [len(stem)]})
    path = f"{directory}/{stem}.parquet"
    uri = f"hf://buckets/{fake_bucket}/{path}"

    plhf.sink_bucket(df, uri)

    assert fake_hub.files(fake_bucket) == [path]
    assert_frame_equal(plhf.scan_bucket(uri).collect(), df)
    # The directory scan finds the same file through the listing. The
    # trailing slash keeps a directory name that ends with a space valid.
    directory_uri = f"hf://buckets/{fake_bucket}/{directory}/"
    assert_frame_equal(plhf.scan_bucket(directory_uri).collect(), df)


# File names made of glob characters and plain ones: when the file exists, the
# URI reads that file, not the files its name matches as a pattern.
_glob_fragments = st.sampled_from(["[", "]", "*", "?", "[1]", "[!a]", "a", "1", "-"])
_glob_stems = (
    st.lists(_glob_fragments, min_size=1, max_size=5)
    .map("".join)
    .filter(lambda stem: any(c in stem for c in "*?["))
)


@_round_trip_settings
@given(stem=_glob_stems, others=st.lists(st.sampled_from("a1-]x"), max_size=3))
def test_file_names_with_glob_characters_read_literally(
    fake_hub: FakeHub, fake_bucket: str, stem: str, others: list[str]
) -> None:
    _clear(fake_hub, fake_bucket, "")
    df = pl.DataFrame({"name": [stem]})
    fake_hub.put_parquet(fake_bucket, f"globby/{stem}.parquet", df)
    # Neighbours that the name can match when it is read as a pattern.
    for other in others:
        if other == stem:
            continue
        neighbour = pl.DataFrame({"name": [f"neighbour {other}"]})
        fake_hub.put_parquet(fake_bucket, f"globby/{other}.parquet", neighbour)

    got = plhf.scan_bucket(f"hf://buckets/{fake_bucket}/globby/{stem}.parquet")

    assert_frame_equal(got.collect(), df)


_safe_keys = st.text(
    alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=4
)


# Each example reads several files back, so this property is the slowest one.
@settings(_round_trip_settings, max_examples=6)
@given(
    keys=st.lists(_safe_keys, min_size=1, max_size=4),
    sink=st.sampled_from(ALL_SINKS),
    max_rows=st.sampled_from([None, 1, 3]),
)
def test_partitioned_round_trip(
    fake_hub: FakeHub,
    fake_bucket: str,
    keys: list[str],
    sink,
    max_rows: int | None,
) -> None:
    _clear(fake_hub, fake_bucket, "prop-parts/")
    df = pl.DataFrame({"g": keys, "n": range(len(keys))})
    base = f"hf://buckets/{fake_bucket}/prop-parts"

    sink(df, base, partition_by="g", max_rows_per_file=max_rows)
    back = plhf.scan_bucket(base).collect()

    assert_frame_equal(back.sort("n"), df)
    directories = {
        path.split("/")[1] for path in fake_hub.files(fake_bucket, "prop-parts/")
    }
    assert directories == {f"g={key}" for key in keys}


# Printable ASCII without "/" and "\\": the raw key would make a path the Hub
# rejects (422), which is a different failure from the directory name itself.
_KEY_ALPHABET = (string.printable.strip() + " ").replace("/", "").replace("\\", "")
_any_keys = st.one_of(
    st.none(), st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=6)
)


def _native_partition_directories(df: pl.DataFrame, key: str) -> set[str]:
    """The directory names Polars itself writes for a hive partition by ``key``."""
    with tempfile.TemporaryDirectory() as tmp:
        df.lazy().sink_parquet(pl.PartitionBy(tmp, key=key))
        return {entry.name for entry in Path(tmp).iterdir()}


@settings(
    max_examples=40,
    phases=[Phase.explicit, Phase.generate],
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(keys=st.lists(_any_keys, min_size=1, max_size=4, unique=True))
def test_streamed_partition_layout_is_native(
    fake_hub: FakeHub, fake_bucket: str, keys: list[str | None]
) -> None:
    _clear(fake_hub, fake_bucket, "layout/")
    df = pl.DataFrame(
        {"g": keys, "n": range(len(keys))}, schema={"g": pl.String, "n": pl.Int64}
    )

    sink_streamed(df, f"hf://buckets/{fake_bucket}/layout", partition_by="g")

    written = fake_hub.files(fake_bucket, "layout/")
    directories = {path[len("layout/") :].split("/")[0] for path in written}
    assert directories == _native_partition_directories(df, "g")
