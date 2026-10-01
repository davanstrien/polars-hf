"""Tests of the fake Hub itself (``fakehub.py``), not of polars-hf."""

from __future__ import annotations

import httpx
import polars as pl
import pytest
from fakehub import CDN, SIGNATURE, FakeHub
from huggingface_hub import HfApi, HfFileSystem

import polars_hf as plhf


def _uri(bucket_id: str, path: str) -> str:
    return f"hf://buckets/{bucket_id}/{path}"


def _numbered_frame(start: int, rows: int) -> pl.DataFrame:
    ids = pl.int_range(start, start + rows, eager=True)
    return pl.DataFrame({"id": ids, "label": (ids % 3).cast(pl.String)})


def test_unknown_token_is_refused(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "data/one.parquet", _numbered_frame(0, 5))
    url = f"{fake_hub.endpoint}/buckets/{fake_bucket}/resolve/data/one.parquet"

    wrong = httpx.head(url, headers={"Authorization": "Bearer hf_wrong"})
    missing = httpx.head(url)

    assert wrong.status_code == 401
    assert missing.status_code == 401
    assert fake_hub.requests[0].authorization == "Bearer hf_wrong"
    assert fake_hub.requests[1].authorization is None


def test_resolve_reply_has_the_xet_headers(fake_hub: FakeHub, fake_bucket: str) -> None:
    from conftest import STAGING_TOKEN

    size = fake_hub.put_parquet(fake_bucket, "data/one.parquet", _numbered_frame(0, 5))
    url = f"{fake_hub.endpoint}/buckets/{fake_bucket}/resolve/data/one.parquet"

    reply = httpx.head(url, headers={"Authorization": f"Bearer {STAGING_TOKEN}"})

    xet_hash = reply.headers["X-Xet-Hash"]
    assert reply.status_code == 302
    assert reply.headers["X-Linked-Size"] == str(size)
    assert len(xet_hash) == 64
    assert reply.headers["X-Linked-Etag"] == '"' + xet_hash + '"'
    assert reply.headers["Location"].startswith(fake_hub.cdn_endpoint)


def test_signed_url_requires_its_signature(fake_hub: FakeHub, fake_bucket: str) -> None:
    fake_hub.put_parquet(fake_bucket, "one.parquet", _numbered_frame(0, 10))
    plhf.scan_bucket(_uri(fake_bucket, "one.parquet")).collect()
    served = fake_hub.matching(origin=CDN, method="GET")[0]

    base = f"{fake_hub.cdn_endpoint}{served.path}"
    good = httpx.get(f"{base}?X-Amz-Signature={SIGNATURE}")
    bad = httpx.get(f"{base}?X-Amz-Signature=tampered")

    assert good.status_code == 200
    assert bad.status_code == 401


def test_patched_batch_supports_bytes_paths_and_delete(
    fake_hub: FakeHub, fake_bucket: str, tmp_path
) -> None:
    local = tmp_path / "local.bin"
    local.write_bytes(b"from a path")
    api = HfApi()

    api.batch_bucket_files(
        fake_bucket, add=[(b"from bytes", "a.bin"), (str(local), "b.bin")]
    )
    api.batch_bucket_files(fake_bucket, delete=["a.bin"])

    assert fake_hub.files(fake_bucket) == ["b.bin"]
    assert fake_hub.read(fake_bucket, "b.bin") == b"from a path"
    assert fake_hub.batch_calls[1].deleted == ["a.bin"]


def test_filesystem_delete_reaches_the_fake(
    fake_hub: FakeHub, fake_bucket: str
) -> None:
    fake_hub.put(fake_bucket, "dir/a.bin", b"a")
    fake_hub.put(fake_bucket, "dir/b.bin", b"b")

    HfFileSystem().rm(f"buckets/{fake_bucket}/dir/a.bin")

    assert fake_hub.files(fake_bucket) == ["dir/b.bin"]


def test_batch_on_missing_bucket_raises_the_client_error(fake_hub: FakeHub) -> None:
    from huggingface_hub import errors

    expected = getattr(errors, "BucketNotFoundError", errors.HfHubHTTPError)

    with pytest.raises(expected) as error:
        HfApi().batch_bucket_files("fake-user/no-such-bucket", add=[(b"x", "a.bin")])

    assert error.value.response.status_code == 404


@pytest.mark.parametrize(
    "destination", ["", "/a.bin", "a//b.bin", "a/", "a/../b.bin", "a\\b.bin"]
)
def test_batch_rejects_invalid_destinations(
    fake_hub: FakeHub, fake_bucket: str, destination: str
) -> None:
    from huggingface_hub.errors import HfHubHTTPError

    with pytest.raises(HfHubHTTPError) as error:
        HfApi().batch_bucket_files(fake_bucket, add=[(b"x", destination)])

    assert error.value.response.status_code == 422
    assert fake_hub.files(fake_bucket) == []
