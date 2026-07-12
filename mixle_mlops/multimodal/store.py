"""Blob storage for multimodal uploads. ``BlobStore`` is the abstraction; ``LocalBlobStore`` writes under
``get_settings().data_dir/'blobs'`` and ``S3BlobStore`` records the cloud deployment contract.

A blob is addressed by an opaque id and exposes a retrievable URL *path* (``/v1/files/{id}/content``) that the
gateway serves back. The same id is what a chat message references; ``content.resolve_content`` turns that
reference into the ``data:`` URL (or signed URL, in cloud) the vision backends expect."""
from __future__ import annotations

import base64
import json
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
    def put(self, data: bytes, *, filename: str, content_type: str) -> BlobRecord:
        ...

    @abstractmethod
    def get(self, blob_id: str) -> tuple[BlobRecord, bytes]:
        """Return ``(record, data)`` for the blob; raise ``KeyError`` if unknown."""
        ...

    @abstractmethod
    def has(self, blob_id: str) -> bool:
        ...

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
            json.dumps({"id": blob_id, "filename": filename,
                        "content_type": content_type, "size": len(data)})
        )
        return record

    def get(self, blob_id: str) -> tuple[BlobRecord, bytes]:
        meta_path = self._meta_path(blob_id)
        bin_path = self._bin_path(blob_id)
        if not meta_path.exists() or not bin_path.exists():
            raise KeyError(f"blob {blob_id!r} not found")
        meta = json.loads(meta_path.read_text())
        record = BlobRecord(id=meta["id"], filename=meta["filename"],
                            content_type=meta["content_type"], size=meta["size"])
        return record, bin_path.read_bytes()

    def has(self, blob_id: str) -> bool:
        return self._meta_path(blob_id).exists() and self._bin_path(blob_id).exists()


class S3BlobStore(BlobStore):
    """Cloud blob-store contract for S3-compatible object storage.

    A cloud deployment should write bytes to object storage and return
    short-lived signed URLs from ``data_url`` so large images are not inlined
    into every request. The local-first build uses :class:`LocalBlobStore`.
    """

    def __init__(self, bucket: str | None = None, endpoint: str | None = None):
        s = get_settings()
        self.bucket = bucket or s.s3_bucket
        self.endpoint = endpoint or s.s3_endpoint

    def _unavailable(self) -> NotImplementedError:
        return NotImplementedError(
            "S3BlobStore requires boto3 and MIXLE_S3_BUCKET configuration."
        )

    def put(self, data: bytes, *, filename: str, content_type: str) -> BlobRecord:
        raise self._unavailable()

    def get(self, blob_id: str) -> tuple[BlobRecord, bytes]:
        raise self._unavailable()

    def has(self, blob_id: str) -> bool:
        raise self._unavailable()


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
