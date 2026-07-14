"""Multimodal support: blob storage for uploads + content-part normalization so image inputs flow through the
OpenAI-compatible chat interface to vision LLMs. Backend-agnostic — images become ``image_url`` parts."""

from __future__ import annotations

from .content import (
    MultimodalError,
    guard_image,
    normalize_messages,
    resolve_content,
)
from .ingest import IngestResult, ingest_extraction
from .store import BlobRecord, BlobStore, LocalBlobStore, S3BlobStore, get_blob_store

__all__ = [
    "BlobRecord",
    "BlobStore",
    "LocalBlobStore",
    "S3BlobStore",
    "get_blob_store",
    "MultimodalError",
    "guard_image",
    "normalize_messages",
    "resolve_content",
    "IngestResult",
    "ingest_extraction",
]
