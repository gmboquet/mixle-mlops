"""Normalize ``ChatMessage`` content parts before they reach a backend. Image parts can reference a blob three
ways, and we resolve them all to the ``data:``/``https`` ``image_url`` form the OpenAI-compatible vision backends
expect:

  * ``{"url": "data:image/png;base64,..."}``   — already inline, passed through (size/mime guarded)
  * ``{"url": "https://..."}``                  — remote, passed through
  * ``{"url": "/v1/files/file-abc/content"}`` or ``{"file_id": "file-abc"}`` — an uploaded blob; resolved to a
    ``data:`` URL by reading it from the :class:`BlobStore`.

This keeps the gateway backend-agnostic: by the time a request leaves ``normalize_messages`` every image is a
self-contained ``image_url`` part, so :class:`OpenAICompatAdapter` just forwards it to the vision LLM.

D5 adds two more concerns that live here rather than in a new module, because both are extensions of "how an
image part gets resolved":

  * vision-capability routing (``has_vision``/``select_vision_model``) — a request naming a text-only model
    but carrying an image should be rerouted to a vision-capable model rather than silently dropping/breaking.
  * a serializable image sidecar (``GeoRef``/``StructuredMediaRef``) — a large tiled raster (D1's ``RasterTile``)
    should travel through bundles/tool provenance as a small content-addressed reference (blob id + hash +
    spatial frame), not as an inlined ``data:`` blob, and an ``ImagePart`` is only ever materialized ephemerally
    at the adapter boundary via ``to_image_part``. There is deliberately no process-local ``dict[id(part)]``
    registry: everything the sidecar needs to carry survives a ``to_dict()``/``from_dict()`` JSON round trip."""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.adapters import ChatMessage, ContentPart, ImagePart, ModelAdapter, TextPart
from ..core.registry import ModelRegistry
from .store import BlobStore, get_blob_store

# Reasonable defaults; a vision request with a 30 MB image is almost always a mistake.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}

# Mirrors IC-13's frozen ``mixle://schema/spatial-media/1`` schema uri (workstream M / mixle-knowledge). Kept as
# a local constant rather than a hard dependency: mixle-mlops does not yet declare `mixle_knowledge` as a
# dependency, so `StructuredMediaRef` carries data shaped to that schema (crs/extent/pixel_to_crs) without
# importing it — a future knowledge-store integration can wrap `StructuredMediaRef.to_dict()["georef"]`
# straight into a `SpatialMediaPayload`/`KnowledgeItem` with no reshaping.
SPATIAL_MEDIA_SCHEMA = "mixle://schema/spatial-media/1"

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<b64>;base64)?,(?P<payload>.*)$", re.DOTALL)
_FILE_PATH_RE = re.compile(r"^/v1/files/(?P<id>[\w.-]+?)(?:/content)?$")


class MultimodalError(Exception):
    """Bad image content (oversize, unsupported mime, unknown file id). → HTTP 400 at the gateway."""


def _blob_id_from_url(url: str) -> str | None:
    """Extract a blob id from a gateway file URL/path, else ``None``."""
    m = _FILE_PATH_RE.match(url.strip())
    return m.group("id") if m else None


def guard_image(*, content_type: str, size: int) -> None:
    """Reject oversize or unsupported-mime images. Raises :class:`MultimodalError`."""
    if size > MAX_IMAGE_BYTES:
        raise MultimodalError(f"image is {size} bytes; max is {MAX_IMAGE_BYTES}")
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime not in ALLOWED_IMAGE_TYPES:
        raise MultimodalError(f"unsupported image type {mime!r}; allowed: {sorted(ALLOWED_IMAGE_TYPES)}")


def _guard_data_url(url: str) -> None:
    """Validate an inline ``data:`` image URL's mime + decoded size."""
    m = _DATA_URL_RE.match(url)
    if not m:
        return  # not a data URL (e.g. https://) — nothing to guard here
    mime = (m.group("mime") or "").lower()
    payload = m.group("payload") or ""
    if m.group("b64"):
        try:
            size = len(base64.b64decode(payload, validate=False))
        except (binascii.Error, ValueError) as exc:
            raise MultimodalError(f"invalid base64 image payload: {exc}")
    else:
        size = len(payload)
    guard_image(content_type=mime or "image/png", size=size)


def _resolve_image_part(part: ImagePart, store: BlobStore) -> ImagePart:
    """Turn any blob reference into an inline ``data:`` URL; pass data:/https through (guarded)."""
    image_url = dict(part.image_url)
    file_id = image_url.pop("file_id", None)
    url = image_url.get("url", "")

    if file_id is None and isinstance(url, str):
        file_id = _blob_id_from_url(url)

    if file_id is not None:
        if not store.has(file_id):
            raise MultimodalError(f"referenced file {file_id!r} not found")
        record, data = store.get(file_id)
        guard_image(content_type=record.content_type, size=record.size)
        image_url["url"] = store.data_url(file_id)
        return ImagePart(image_url=image_url)

    if isinstance(url, str) and url:
        _guard_data_url(url)
        image_url["url"] = url
        return ImagePart(image_url=image_url)

    raise MultimodalError("image part has neither a file id nor a url")


def resolve_content(
    content: str | list[ContentPart], store: BlobStore | None = None
) -> str | list[ContentPart]:
    """Resolve every image part of one message's content into a self-contained ``image_url`` part."""
    if isinstance(content, str):
        return content
    store = store or get_blob_store()
    out: list[ContentPart] = []
    for part in content:
        if isinstance(part, ImagePart):
            out.append(_resolve_image_part(part, store))
        elif isinstance(part, TextPart):
            out.append(part)
        else:  # pragma: no cover - exhaustive over ContentPart union
            out.append(part)
    return out


def normalize_messages(
    messages: list[ChatMessage], store: BlobStore | None = None
) -> list[ChatMessage]:
    """Return new messages with all image parts resolved to backend-ready ``image_url`` parts."""
    store = store or get_blob_store()
    return [
        m.model_copy(update={"content": resolve_content(m.content, store)})
        for m in messages
    ]


# --- D5: vision-capability routing -----------------------------------------------------------------------------


def has_vision(adapter: ModelAdapter) -> bool:
    """Whether ``adapter`` advertises image understanding, per its ``capabilities()``."""
    caps = adapter.capabilities()
    return "vision" in caps or "image" in caps


def select_vision_model(registry: ModelRegistry, requested: str) -> str:
    """Return ``requested`` if it is registered and vision-capable; else the cheapest registered vision-capable
    model id; else raise :class:`MultimodalError`.

    "Cheapest" honors an optional ``cost_per_1k_tokens`` (or ``cost``) attribute on an adapter when present —
    the registry/adapter surface has no frozen pricing field yet, so this is a best-effort, forward-compatible
    proxy: unpriced adapters sort last and ties break on model id, so the choice is always deterministic.
    """
    if requested and registry.has(requested) and has_vision(registry.get(requested)):
        return requested

    vision_ids = [model_id for model_id in registry.names() if has_vision(registry.get(model_id))]
    if not vision_ids:
        raise MultimodalError("request carries an image but no registered model supports vision")

    def _cost(model_id: str) -> tuple[float, str]:
        adapter = registry.get(model_id)
        cost = getattr(adapter, "cost_per_1k_tokens", None)
        if cost is None:
            cost = getattr(adapter, "cost", None)
        return (float(cost) if cost is not None else float("inf"), model_id)

    return min(vision_ids, key=_cost)


# --- D5: image georeferencing sidecar ---------------------------------------------------------------------------


@dataclass
class GeoRef:
    """The spatial frame a piece of image content carries: CRS, ground ``extent`` (minx, miny, maxx, maxy),
    ground-units-per-pixel ``scale``, and (optionally) the full six-term pixel→CRS affine. All fields are
    plain, JSON-serializable values — a ``GeoRef`` is data, never a handle into process memory."""

    crs: str | None
    extent: tuple[float, float, float, float] | None
    scale: float | None
    pixel_to_crs: tuple[float, float, float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "extent": list(self.extent) if self.extent is not None else None,
            "scale": self.scale,
            "pixel_to_crs": list(self.pixel_to_crs) if self.pixel_to_crs is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeoRef":
        extent = d.get("extent")
        pixel_to_crs = d.get("pixel_to_crs")
        return cls(
            crs=d.get("crs"),
            extent=tuple(extent) if extent is not None else None,
            scale=d.get("scale"),
            pixel_to_crs=tuple(pixel_to_crs) if pixel_to_crs is not None else None,
        )


@dataclass
class StructuredMediaRef:
    """A serializable, content-addressed reference to image media: a blob-store ``artifact_ref`` + its
    ``content_hash`` + an optional ``georef`` spatial frame + free-form ``provenance``. This is what travels in
    bundles/tool provenance across processes — never a live ``ImagePart``/``id(part)`` handle. An ``ImagePart``
    is materialized fresh, only at the adapter boundary, via :meth:`to_image_part`."""

    artifact_ref: str
    content_hash: str
    media_type: str
    georef: GeoRef | None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_ref": self.artifact_ref,
            "content_hash": self.content_hash,
            "media_type": self.media_type,
            "georef": self.georef.to_dict() if self.georef is not None else None,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StructuredMediaRef":
        georef = d.get("georef")
        return cls(
            artifact_ref=d["artifact_ref"],
            content_hash=d["content_hash"],
            media_type=d["media_type"],
            georef=GeoRef.from_dict(georef) if georef is not None else None,
            provenance=dict(d.get("provenance") or {}),
        )

    def to_image_part(self, store: BlobStore) -> ImagePart:
        """Materialize an ephemeral ``ImagePart`` (an inline ``data:`` URL) at the adapter boundary — the one
        place a vision backend actually needs bytes. Never cached/keyed by identity; call again to re-resolve."""
        if not store.has(self.artifact_ref):
            raise MultimodalError(f"structured media ref {self.artifact_ref!r} not found in store")
        return ImagePart(image_url={"url": store.data_url(self.artifact_ref)})


def attach_georef(media: StructuredMediaRef, geo: GeoRef) -> StructuredMediaRef:
    """Return a copy of ``media`` with ``geo`` attached. A plain, serializable update — never keys state by
    ``id(part)``; the returned record is the only handle the caller needs to keep."""
    return replace(media, georef=geo)


def media_ref_from_tile(
    tile: Any,
    store: BlobStore,
    *,
    crs: str | None = None,
    pixel_to_crs: tuple[float, ...] | None = None,
    media_type: str = "image/png",
    filename: str = "tile.png",
    provenance: dict[str, Any] | None = None,
) -> StructuredMediaRef:
    """Persist one tiled-raster tile's encoded bytes as a blob and wrap it in a :class:`StructuredMediaRef`
    carrying its spatial frame (D1 ``RasterTile`` → D5 sidecar). Duck-typed on ``tile.png``/``tile.extent``/
    ``tile.scale`` so it works against D1's ``RasterTile`` (or any tile-shaped object) without importing
    ``mixle_mlops.multimodal.raster`` here."""
    data: bytes = tile.png
    content_hash = hashlib.sha256(data).hexdigest()
    record = store.put(data, filename=filename, content_type=media_type)
    extent = getattr(tile, "extent", None)
    geo = GeoRef(
        crs=crs,
        extent=tuple(extent) if extent is not None else None,
        scale=getattr(tile, "scale", None),
        pixel_to_crs=tuple(pixel_to_crs) if pixel_to_crs is not None else None,
    )
    return StructuredMediaRef(
        artifact_ref=record.id,
        content_hash=content_hash,
        media_type=media_type,
        georef=geo,
        provenance=dict(provenance or {}),
    )
