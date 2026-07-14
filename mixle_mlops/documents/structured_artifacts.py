"""Turn parsed structured data (typed tables today; curve sets, graphs, ... later) into IC-13 knowledge items
plus the content-addressed artifact handles they point back at.

``parse.py`` owns *format* dispatch (PDF/XLSX/LAS extraction); this module owns the two format-agnostic concerns
that sit downstream of it:

  * :func:`store_artifact` -- persist raw bytes into a :class:`~mixle_mlops.multimodal.store.BlobStore` and hand
    back a handle that carries a sha256 content hash alongside the store's opaque id, in the spirit of
    ``mixle_pde.io.artifacts`` (IC-2): a ``*_ref`` should be a verifiable, content-addressed handle, not just an
    opaque pointer. (The concrete :class:`BlobStore` allocator has no content-addressed dedup of its own -- two
    ``put()`` calls with identical bytes get two different ids -- so the hash is threaded alongside for
    verification rather than used as the store key.)
  * :func:`build_typed_table_item` -- build one canonical ``mixle://schema/typed-table/1`` ``KnowledgeItem`` dict
    (IC-13, frozen in ``mixle_knowledge.contracts``) per sheet. ``content_hash`` is derived purely from the
    ``payload`` (columns + rows), never from the ``text_surface`` rendering, so deleting or rewriting the text
    surface never changes a table's identity.

``mixle_knowledge`` is used when importable (it is the frozen source of truth for the ``KnowledgeItem`` shape) but
is not a hard runtime dependency of this package -- ``build_typed_table_item`` falls back to hand-building a plain
dict with the identical field set when it isn't on the path, matching the lazy-import convention the rest of
``documents/parse.py`` already uses for optional parsers.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..multimodal.store import BlobStore

TYPED_TABLE_SCHEMA = "mixle://schema/typed-table/1"


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return s or "sheet"


def content_hash_bytes(data: bytes) -> str:
    """sha256 hex digest of raw bytes -- the IC-2-style content hash for a stored artifact."""
    return hashlib.sha256(data).hexdigest()


def content_hash_payload(payload: dict[str, Any]) -> str:
    """sha256 hex digest of a canonicalised JSON payload, independent of any accompanying text surface."""
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class StoredArtifact:
    """A persisted immutable blob plus the content hash callers thread through as ``artifact_ref``."""

    ref: str
    sha256: str
    media_type: str
    filename: str
    size: int


def store_artifact(data: bytes, *, filename: str, media_type: str, store: BlobStore) -> StoredArtifact:
    """Persist ``data`` into ``store`` and return a handle carrying its content hash."""
    record = store.put(data, filename=filename, content_type=media_type)
    return StoredArtifact(
        ref=record.id,
        sha256=content_hash_bytes(data),
        media_type=media_type,
        filename=filename,
        size=len(data),
    )


def load_artifact_bytes(ref: str, store: BlobStore) -> bytes:
    """Hydrate the original bytes behind an ``artifact_ref`` produced by :func:`store_artifact`."""
    _record, data = store.get(ref)
    return data


def build_typed_table_item(
    *,
    item_id: str,
    sheet_name: str,
    primary_key: list[str],
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    workbook_ref: str,
    workbook_sha256: str,
    workbook_media_type: str,
    source_filename: str,
    text_surface: str | None = None,
) -> dict[str, Any]:
    """One IC-13 ``KnowledgeItem`` dict (``mixle://schema/typed-table/1`` payload) for one workbook sheet.

    Every item carries ``artifact_ref = workbook_ref`` (the raw workbook bytes) plus a ``derived_from`` relation
    whose metadata records ``sheet_name`` -- the durable link a consumer hydrates structure through rather than
    reconstructing it from ``text_surface``.
    """
    payload = {"primary_key": list(primary_key), "columns": list(columns), "rows": list(rows)}
    chash = content_hash_payload(payload)
    metadata = {"sheet_name": sheet_name, "source_filename": source_filename}
    relation_meta = {"sheet_name": sheet_name}

    try:
        from mixle_knowledge.contracts import (
            KnowledgeItem,
            KnowledgeRelation,
            Modality,
            ResourceKind,
            SourceRef,
        )
    except ImportError:
        return {
            "id": item_id,
            "kind": "table",
            "modality": "table",
            "schema_uri": TYPED_TABLE_SCHEMA,
            "schema_version": "1.0.0",
            "media_type": workbook_media_type,
            "content_hash": chash,
            "payload": payload,
            "artifact_ref": workbook_ref,
            "text_surface": text_surface,
            "provenance": [
                {"uri": f"artifact://{workbook_ref}", "sha256": workbook_sha256, "media_type": workbook_media_type}
            ],
            "relations": [
                {"predicate": "derived_from", "target_id": workbook_ref, "provenance": [], "metadata": relation_meta}
            ],
            "uncertainty": None,
            "metadata": metadata,
            "revision": 1,
            "supersedes": [],
        }

    item = KnowledgeItem(
        id=item_id,
        kind=ResourceKind.TABLE,
        modality=Modality.TABLE,
        schema_uri=TYPED_TABLE_SCHEMA,
        media_type=workbook_media_type,
        content_hash=chash,
        payload=payload,
        artifact_ref=workbook_ref,
        text_surface=text_surface,
        provenance=[SourceRef(uri=f"artifact://{workbook_ref}", sha256=workbook_sha256, media_type=workbook_media_type)],
        relations=[KnowledgeRelation(predicate="derived_from", target_id=workbook_ref, metadata=relation_meta)],
        metadata=metadata,
    )
    return item.model_dump(mode="json")
