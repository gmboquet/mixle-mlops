"""Serving-time modality embedding adapters (M6a, work-plan §7-M).

Embeds an IC-13-shaped item (text/document, typed table, property graph, image, or field/time-series)
into its own **native** vector space at serve time -- distinct spaces per modality (``text-v1``,
``table-v1``, ``graph-v1``, ``image-openclip-v1``, ``field-v1``), never a single shared space. M1b's
federated retrieval searches within compatible spaces and normalizes scores across them; nothing here
compares raw vectors from different spaces directly -- that only becomes sound once M6's learned
cross-space alignment is loaded via :meth:`ModalityEmbedder.load_alignment`.

Item shape mirrors IC-13's ``mixle://schema/...`` convention (id/content_hash/modality/ref) without a
hard dependency on ``mixle_knowledge`` -- same "shaped like it, not imported from it" convention D5's
``multimodal/content.py`` already uses for ``StructuredMediaRef``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .embeddings import Embedder

# Native embedding-space ids per modality. Distinct dims/semantics -- never compared raw across spaces.
TEXT_SPACE = "text-v1"
TABLE_SPACE = "table-v1"
GRAPH_SPACE = "graph-v1"
IMAGE_SPACE = "image-openclip-v1"
FIELD_SPACE = "field-v1"

_MAX_ROW_SAMPLE = 32


class IncompatibleSpaceError(Exception):
    """Raised when two vectors from different (unaligned) `space_id`s are compared directly."""


@dataclass(frozen=True)
class EmbeddedItem:
    """A serve-time embedding of one IC-13 item: identity (id/hash) + the native-space vector."""

    item_id: str
    content_hash: str
    modality: str
    space_id: str
    vector: np.ndarray


def _resolver_get_bytes(item: dict[str, Any], resolver: Any) -> bytes:
    """Hydrate an item's raw bytes via the resolver (a `BlobStore`-shaped object: `.get(file_id)`)."""
    ref = item.get("ref")
    if ref is None:
        raise ValueError(f"item {item.get('id')!r} has no `ref` to hydrate bytes from")
    if resolver is None:
        raise ValueError("image items require a resolver to hydrate bytes")
    _record, data = resolver.get(ref)
    return data


def _table_surface(item: dict[str, Any]) -> str:
    """Deterministic semantic surface for a typed table: column names/types/units + a bounded row sample."""
    columns = item.get("columns", [])
    col_text = "; ".join(
        f"{c.get('name')}:{c.get('dtype', 'unknown')}" + (f"[{c['unit']}]" if c.get("unit") else "")
        for c in columns
    )
    rows = item.get("rows", [])[:_MAX_ROW_SAMPLE]
    rows_text = json.dumps(rows, sort_keys=True, default=str)
    return f"table columns: {col_text}\nrows sample: {rows_text}"


def _graph_surface(item: dict[str, Any]) -> str:
    """Deterministic semantic surface for a property graph: node/edge types + selected labels."""
    node_types = sorted({n.get("type", "") for n in item.get("nodes", [])})
    edge_types = sorted({e.get("type", "") for e in item.get("edges", [])})
    labels = sorted(str(n.get("label", "")) for n in item.get("nodes", []) if n.get("label"))[:_MAX_ROW_SAMPLE]
    return f"graph node_types: {node_types}\nedge_types: {edge_types}\nlabels: {labels}"


def _field_surface(item: dict[str, Any]) -> str:
    """Deterministic semantic surface for a field/time series: schema + summary statistics."""
    values = np.asarray(item.get("values", []), dtype=float)
    schema = item.get("schema", {})
    if values.size:
        stats = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    else:
        stats = {"n": 0}
    return f"field schema: {json.dumps(schema, sort_keys=True)}\nstats: {json.dumps(stats, sort_keys=True)}"


def _hash_image_bytes(data: bytes, dim: int = 512) -> np.ndarray:
    """Deterministic fallback image embedding (feature-hashed byte n-grams) used when OpenCLIP is absent.

    Only a fallback -- :func:`_openclip_embed` is preferred whenever ``open_clip`` is importable.
    """
    vec = np.zeros(dim, dtype=np.float64)
    step = max(1, len(data) // 4096)
    for i in range(0, len(data), step):
        chunk = data[i : i + 8]
        h = hashlib.sha1(chunk).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _openclip_embed(data: bytes) -> np.ndarray | None:
    """Try a real OpenCLIP image embedding; return ``None`` if the optional dependency isn't installed."""
    try:
        import io

        import open_clip
        import torch
        from PIL import Image
    except ImportError:
        return None

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    image = preprocess(Image.open(io.BytesIO(data)).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).numpy().astype(np.float64)


class ModalityEmbedder:
    """Embed IC-13 items into per-modality native vector spaces at serve time (M6a).

    Caches vectors by ``(content_hash, space_id, alignment_handle)`` so repeated calls on the same
    hashed item are stable and cheap; embedding never mutates the item's payload/ref.
    """

    def __init__(self, *, embedder: Embedder | None = None):
        self._embedder = embedder or Embedder(allow_remote=False)
        self._cache: dict[tuple[str, str, str | None], np.ndarray] = {}
        self._alignments: dict[str, np.ndarray] = {}
        self._alignment_handle: str | None = None

    def embed_item(self, item: dict[str, Any], *, resolver: Any = None) -> EmbeddedItem:
        """Embed one IC-13-shaped item (dict with at least ``id``/``content_hash``/``modality``)."""
        item_id = item["id"]
        content_hash = item["content_hash"]
        modality = item["modality"]

        cache_key = (content_hash, self._space_id_for(modality), self._alignment_handle)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return EmbeddedItem(
                item_id=item_id, content_hash=content_hash, modality=modality,
                space_id=cache_key[1], vector=cached,
            )

        space_id = self._space_id_for(modality)
        if modality in ("text", "document"):
            text = item.get("text", "")
            vec = self._embedder.embed([text])[0]
        elif modality == "table":
            vec = self._embedder.embed([_table_surface(item)])[0]
        elif modality == "graph":
            vec = self._embedder.embed([_graph_surface(item)])[0]
        elif modality == "field":
            vec = self._embedder.embed([_field_surface(item)])[0]
        elif modality == "image":
            data = _resolver_get_bytes(item, resolver)
            vec = _openclip_embed(data)
            if vec is None:
                vec = _hash_image_bytes(data)
        else:
            raise ValueError(f"unknown modality {modality!r}")

        vec = np.asarray(vec, dtype=np.float64)
        self._cache[cache_key] = vec
        return EmbeddedItem(item_id=item_id, content_hash=content_hash, modality=modality, space_id=space_id, vector=vec)

    def embed_query(self, query: str | bytes, *, modality: str) -> tuple[str, np.ndarray]:
        """Embed a query string/bytes into the native space for ``modality``; returns ``(space_id, vector)``."""
        space_id = self._space_id_for(modality)
        if modality in ("text", "document", "table", "graph", "field"):
            if not isinstance(query, str):
                raise TypeError(f"{modality} queries must be str")
            vec = self._embedder.embed([query])[0]
        elif modality == "image":
            if not isinstance(query, (bytes, bytearray)):
                raise TypeError("image queries must be bytes")
            vec = _openclip_embed(bytes(query))
            if vec is None:
                vec = _hash_image_bytes(bytes(query))
        else:
            raise ValueError(f"unknown modality {modality!r}")
        return space_id, np.asarray(vec, dtype=np.float64)

    def load_alignment(self, handle: str) -> None:
        """Install M6's learned cross-space projections (an ``.npz`` of ``space_id -> projection matrix``).

        Installing an alignment never changes indexed item ids/hashes or invalidates the native-space
        cache entries already computed -- it only changes the ``alignment_handle`` component of the cache
        key, so subsequently-embedded items are looked up (or recomputed) under the new alignment epoch.
        """
        data = np.load(handle, allow_pickle=False)
        self._alignments = {name: data[name] for name in data.files}
        self._alignment_handle = handle

    @staticmethod
    def _space_id_for(modality: str) -> str:
        return {
            "text": TEXT_SPACE,
            "document": TEXT_SPACE,
            "table": TABLE_SPACE,
            "graph": GRAPH_SPACE,
            "image": IMAGE_SPACE,
            "field": FIELD_SPACE,
        }[modality]


def assert_comparable(a: EmbeddedItem, b: EmbeddedItem) -> None:
    """Guard against comparing raw vectors from different (unaligned) `space_id`s directly."""
    if a.space_id != b.space_id:
        raise IncompatibleSpaceError(
            f"cannot directly compare vectors from {a.space_id!r} and {b.space_id!r}; "
            "align via ModalityEmbedder.load_alignment (M6) first"
        )
