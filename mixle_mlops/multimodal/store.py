"""Blob storage for multimodal uploads. ``BlobStore`` is the abstraction; ``LocalBlobStore`` writes under
``get_settings().data_dir/'blobs'`` (local-first), and ``S3BlobStore`` backs the cloud deployment via ``boto3``.

A blob is addressed by an opaque id and exposes a retrievable URL *path* (``/v1/files/{id}/content``) that the
gateway serves back. The same id is what a chat message references; ``content.resolve_content`` turns that
reference into the ``data:`` URL (or signed URL, in cloud) the vision backends expect."""
from __future__ import annotations

import base64
import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings


def _blob_id() -> str:
    return "file-" + uuid.uuid4().hex


@dataclass
class BlobRecord:
    """Metadata for a stored blob. ``url`` is the gateway path that serves the bytes back."""

    id: str
    filename: str
    content_type: str
    size: int

    @property
    def url(self) -> str:
        return f"/v1/files/{self.id}/content"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "url": self.url,
            "object": "file",
        }


class BlobStore(ABC):
    """Store and retrieve opaque binary blobs (uploaded images/files) by id."""

    @abstractmethod
    def put(self, data: bytes, *, filename: str, content_type: str) -> BlobRecord: ...

    @abstractmethod
    def get(self, blob_id: str) -> tuple[BlobRecord, bytes]:
        """Return ``(record, data)`` for the blob; raise ``KeyError`` if unknown."""
        ...

    @abstractmethod
    def has(self, blob_id: str) -> bool: ...

    def data_url(self, blob_id: str) -> str:
        """Inline ``data:`` URL for the blob — what the OpenAI-compatible image parts carry by value."""
        record, data = self.get(blob_id)
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{record.content_type};base64,{b64}"


class LocalBlobStore(BlobStore):
    """Filesystem-backed store: bytes + a ``.json`` sidecar of metadata, under ``data_dir/'blobs'``."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else (get_settings().data_dir / "blobs")
        self.root.mkdir(parents=True, exist_ok=True)

    def _bin_path(self, blob_id: str) -> Path:
        return self.root / f"{blob_id}.bin"

    def _meta_path(self, blob_id: str) -> Path:
        return self.root / f"{blob_id}.json"

    def put(self, data: bytes, *, filename: str, content_type: str) -> BlobRecord:
        blob_id = _blob_id()
        record = BlobRecord(id=blob_id, filename=filename, content_type=content_type, size=len(data))
        self._bin_path(blob_id).write_bytes(data)
        self._meta_path(blob_id).write_text(
            json.dumps({"id": blob_id, "filename": filename, "content_type": content_type, "size": len(data)})
        )
        return record

    def get(self, blob_id: str) -> tuple[BlobRecord, bytes]:
        meta_path = self._meta_path(blob_id)
        bin_path = self._bin_path(blob_id)
        if not meta_path.exists() or not bin_path.exists():
            raise KeyError(f"blob {blob_id!r} not found")
        meta = json.loads(meta_path.read_text())
        record = BlobRecord(
            id=meta["id"], filename=meta["filename"], content_type=meta["content_type"], size=meta["size"]
        )
        return record, bin_path.read_bytes()

    def has(self, blob_id: str) -> bool:
        return self._meta_path(blob_id).exists() and self._bin_path(blob_id).exists()


class S3BlobStore(BlobStore):
    """Cloud-backed store: writes each blob to S3 (or any S3-compatible endpoint, e.g. MinIO) via ``boto3``,
    keyed off ``get_settings().s3_bucket`` / ``s3_endpoint``. Mirrors ``LocalBlobStore``'s bin+json layout —
    a blob is two objects, ``{id}`` (raw bytes) and ``{id}.json`` (a metadata sidecar) — so ``get`` can
    reconstruct the ``BlobRecord`` without a separate database. ``data_url`` returns a short-lived signed
    GET URL instead of inlining bytes, so large images aren't base64'd into every request in the cloud."""

    _URL_TTL_SECONDS = 3600

    def __init__(self, bucket: str | None = None, endpoint: str | None = None, *, region: str | None = None):
        s = get_settings()
        self.bucket = bucket or s.s3_bucket
        self.endpoint = endpoint or s.s3_endpoint
        self.region = region
        self._client = None

    def _meta_key(self, blob_id: str) -> str:
        return f"{blob_id}.json"

    def _s3(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "S3BlobStore requires boto3; install it with: pip install mixle-mlops[cloud]"
                ) from exc
            if not self.bucket:
                raise RuntimeError(
                    "S3BlobStore requires a bucket; set MIXLE_S3_BUCKET or pass bucket=... explicitly."
                )
            region = (
                self.region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
            )
            self._client = boto3.client("s3", endpoint_url=self.endpoint or None, region_name=region)
        return self._client

    def put(self, data: bytes, *, filename: str, content_type: str) -> BlobRecord:
        blob_id = _blob_id()
        record = BlobRecord(id=blob_id, filename=filename, content_type=content_type, size=len(data))
        client = self._s3()
        client.put_object(Bucket=self.bucket, Key=blob_id, Body=data, ContentType=content_type)
        meta = {"id": blob_id, "filename": filename, "content_type": content_type, "size": len(data)}
        client.put_object(
            Bucket=self.bucket,
            Key=self._meta_key(blob_id),
            Body=json.dumps(meta).encode("utf-8"),
            ContentType="application/json",
        )
        return record

    def get(self, blob_id: str) -> tuple[BlobRecord, bytes]:
        from botocore.exceptions import ClientError

        client = self._s3()
        try:
            meta_obj = client.get_object(Bucket=self.bucket, Key=self._meta_key(blob_id))
            meta = json.loads(meta_obj["Body"].read())
            data_obj = client.get_object(Bucket=self.bucket, Key=blob_id)
            data = data_obj["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                raise KeyError(f"blob {blob_id!r} not found") from exc
            raise
        record = BlobRecord(
            id=meta["id"], filename=meta["filename"], content_type=meta["content_type"], size=meta["size"]
        )
        return record, data

    def has(self, blob_id: str) -> bool:
        from botocore.exceptions import ClientError

        client = self._s3()
        try:
            client.head_object(Bucket=self.bucket, Key=blob_id)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                return False
            raise
        return True

    def data_url(self, blob_id: str) -> str:
        client = self._s3()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": blob_id},
            ExpiresIn=self._URL_TTL_SECONDS,
        )


_store: BlobStore | None = None


def get_blob_store() -> BlobStore:
    """Process-wide blob store, chosen by deployment (local fs vs S3). Cached after first use."""
    global _store
    if _store is None:
        settings = get_settings()
        _store = S3BlobStore() if settings.deployment == "cloud" else LocalBlobStore()
    return _store


def reset_blob_store() -> None:
    """Test hook: drop the cached store so a fresh ``data_dir`` is picked up."""
    global _store
    _store = None
