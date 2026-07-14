"""D4 — S3BlobStore against a mocked S3 bucket (moto). put -> get round-trips bytes + content-type,
has() reflects existence, and data_url returns a signed URL naming the bucket + key. The fixture flips
``deployment`` to ``cloud`` and sets a bucket, so ``get_blob_store()`` itself resolves to ``S3BlobStore``
(store.py's existing selector, untouched by this task) alongside a directly-constructed instance."""

from __future__ import annotations

import pytest

boto3 = pytest.importorskip("boto3")
pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

import mixle_mlops.multimodal.store as store_mod  # noqa: E402
from mixle_mlops.config import get_settings  # noqa: E402
from mixle_mlops.multimodal.store import S3BlobStore, get_blob_store  # noqa: E402

BUCKET = "mixle-test-bucket"


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setenv("MIXLE_DEPLOYMENT", "cloud")
    monkeypatch.setenv("MIXLE_S3_BUCKET", BUCKET)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    get_settings.cache_clear()
    store_mod.reset_blob_store()
    yield
    get_settings.cache_clear()
    store_mod.reset_blob_store()


@mock_aws
def test_put_get_has_and_data_url(cloud):
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)

    # get_blob_store() picks S3BlobStore under deployment=cloud without any change to the selector.
    store = get_blob_store()
    assert isinstance(store, S3BlobStore)

    data = b"\x89PNG-fake-bytes-for-a-test-blob"
    record = store.put(data, filename="pixel.png", content_type="image/png")
    assert record.filename == "pixel.png"
    assert record.content_type == "image/png"
    assert record.size == len(data)

    got_record, got_data = store.get(record.id)
    assert got_data == data
    assert got_record.id == record.id
    assert got_record.filename == "pixel.png"
    assert got_record.content_type == "image/png"
    assert got_record.size == len(data)

    assert store.has(record.id) is True
    assert store.has("file-does-not-exist") is False

    url = store.data_url(record.id)
    assert BUCKET in url
    assert record.id in url


@mock_aws
def test_get_unknown_raises_keyerror(cloud):
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
    store = S3BlobStore(bucket=BUCKET)
    with pytest.raises(KeyError):
        store.get("file-does-not-exist")
