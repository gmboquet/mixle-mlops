"""I2 — cross-section & log-plot curve/value extraction (the folded D6).

Turns a plotted curve or a set of annotated point values on a raster image (a well-log track, a resistivity/
porosity "log-plot" panel, a geological cross-section) into typed, physically-calibrated arrays, ready for I7's
verification gate and eventually the physics tools (workstream E). Two entry points:

  * :func:`extract_curve`  — one continuous curve per track → a :class:`DepthSeries` (depth/distance + value).
  * :func:`extract_values` — a sparse set of discrete point annotations → ``list[dict]`` with stable ids.

Algorithm (shared by both): 1) resolve the immutable source image bytes via the D5 ``BlobStore``/``StructuredMediaRef``
plumbing (never mutate/re-encode the source — only its hash is carried forward). 2) ask a vision-capable model
for typed pixel-space JSON (a polyline for a curve, point centroids for annotations); if no model is configured,
or the call fails for any reason (no network in a sandbox, no credentials, a malformed reply), fall back to a
deterministic, VLM-free pixel digitizer — a frontier VLM is an accuracy *enhancement* here, never a hard
dependency. 3) map pixels to physical units via :class:`AxisCalib` (linear or logarithmic value axis — the
"log-plot" case). 4) sort by the primary axis, dedupe colocated samples, clip to the declared value range.
5) persist the exact arrays + dtype/unit/axis-calibration + source-image hash as an IC-13-shaped typed-table
``knowledge_item`` (mirrored locally as a plain dict, the same way ``multimodal.content.SPATIAL_MEDIA_SCHEMA``
mirrors IC-13's spatial-media schema uri — mixle-mlops does not depend on ``mixle_knowledge``); only a short
``text_surface`` is a text summary, and it is never required to reconstruct the arrays. 6) return typed objects
ready for I7 to verify before anything reaches physics.

Non-goals (owned elsewhere): LAS parsing (mixle-pde B3), georeferencing (I1/I6), the ingest/verification gate
(I7) — this module only digitizes and types what is already on the page.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .store import BlobStore, get_blob_store

# Mirrors IC-13's frozen ``mixle://schema/typed-table/1`` schema uri (workstream M / mixle-knowledge). Kept as a
# local constant for the same reason ``multimodal.content.SPATIAL_MEDIA_SCHEMA`` is: mixle-mlops does not declare
# ``mixle_knowledge`` as a dependency, so ``knowledge_item`` payloads are shaped to that schema without importing
# it — a future knowledge-store integration can hand this dict straight to a ``TypedTablePayload``/``KnowledgeItem``.
TYPED_TABLE_SCHEMA = "mixle://schema/typed-table/1"
KNOWLEDGE_ITEM_SCHEMA_VERSION = "1.0.0"

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?:;base64)?,(?P<payload>.*)$", re.DOTALL)


class ExtractionError(Exception):
    """Raised for a bad ``image_ref``, an uncalibratable axis, or a track with no digitizable pixels."""


# --- public dataclasses ------------------------------------------------------------------------------------


@dataclass
class AxisCalib:
    """Pixel <-> physical calibration for one track/panel of a cross-section or log-plot image.

    ``depth_px``/``depth_range`` calibrate the plot's primary axis — depth (increasing downward) for a well-log
    track, or along-section distance for a geological cross-section; whichever it is, it becomes
    :attr:`DepthSeries.depth`. ``value_px``/``value_range`` calibrate the plotted curve's own axis: linear by
    default, or logarithmic (``value_log=True``) for the decade-scaled tracks (resistivity, permeability, ...)
    that make "log-plot" extraction distinct from a plain linear chart. ``curve_rgb``/``color_tolerance``
    identify the plotted curve's pixel color for the offline (no-VLM) digitizer.
    """

    depth_px: tuple[float, float]
    depth_range: tuple[float, float]
    value_px: tuple[float, float]
    value_range: tuple[float, float]
    value_log: bool = False
    depth_unit: str = "m"
    value_unit: str = ""
    curve_rgb: tuple[int, int, int] = (0, 0, 0)
    color_tolerance: float = 40.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_px": list(self.depth_px),
            "depth_range": list(self.depth_range),
            "value_px": list(self.value_px),
            "value_range": list(self.value_range),
            "value_log": self.value_log,
            "depth_unit": self.depth_unit,
            "value_unit": self.value_unit,
            "curve_rgb": list(self.curve_rgb),
            "color_tolerance": self.color_tolerance,
        }


@dataclass
class DepthSeries:
    """A digitized curve: parallel ``depth``/``value`` arrays plus provenance and its IC-13 ``knowledge_item``."""

    depth: np.ndarray
    value: np.ndarray
    unit: str
    provenance: dict[str, Any] = field(default_factory=dict)
    knowledge_item: dict[str, Any] = field(default_factory=dict)


# --- image / blob resolution -------------------------------------------------------------------------------


def _decode_data_url(url: str) -> tuple[str, bytes]:
    m = _DATA_URL_RE.match(url)
    if not m:
        raise ExtractionError("unrecognized data: URL for source image")
    mime = m.group("mime") or "application/octet-stream"
    try:
        data = base64.b64decode(m.group("payload"), validate=False)
    except (ValueError, TypeError) as exc:
        raise ExtractionError(f"invalid base64 image payload: {exc}") from exc
    return mime, data


def _resolve_image_bytes(image_ref: str, store: BlobStore) -> tuple[bytes, str, str]:
    """Resolve ``image_ref`` to raw bytes via the blob store's public API. Returns ``(data, media_type,
    sha256_hex)`` — the immutable source triple every downstream artifact is provenanced against. A ``data:``
    URL is also accepted directly (a caller that already has bytes in hand, no store round-trip needed)."""
    if image_ref.startswith("data:"):
        media_type, data = _decode_data_url(image_ref)
    else:
        if not store.has(image_ref):
            raise ExtractionError(f"source image {image_ref!r} not found in the blob store")
        record, data = store.get(image_ref)
        media_type = record.content_type
    return data, media_type, hashlib.sha256(data).hexdigest()


def _load_image_array(data: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without Pillow installed
        raise ExtractionError(
            "cross-section/log-plot curve extraction needs Pillow to decode the source image; "
            "install it with: pip install pillow"
        ) from exc
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


# --- pixel <-> physical mapping ----------------------------------------------------------------------------


def _map_point(x: float, y: float, axis: AxisCalib) -> tuple[float, float]:
    y0, y1 = axis.depth_px
    d0, d1 = axis.depth_range
    depth = d0 + (y - y0) * (d1 - d0) / (y1 - y0)
    x0, x1 = axis.value_px
    v0, v1 = axis.value_range
    t = (x - x0) / (x1 - x0)
    if axis.value_log:
        if v0 <= 0 or v1 <= 0:
            raise ExtractionError("a logarithmic value axis needs strictly positive value_range bounds")
        value = v0 * (v1 / v0) ** t
    else:
        value = v0 + t * (v1 - v0)
    return depth, value


def _pixels_to_series(points: list[dict[str, float]], axis: AxisCalib) -> tuple[np.ndarray, np.ndarray]:
    """Map pixel points to physical (depth, value), then sort/dedupe/clip (I2 algorithm steps 3-4)."""
    mapped = sorted((_map_point(p["x"], p["y"], axis) for p in points), key=lambda dv: dv[0])
    depths: list[float] = []
    values: list[float] = []
    for d, v in mapped:
        if depths and math.isclose(d, depths[-1], rel_tol=0.0, abs_tol=1e-9):
            values[-1] = (values[-1] + v) / 2.0  # dedupe colocated samples by averaging
            continue
        depths.append(d)
        values.append(v)
    depth = np.asarray(depths, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    lo, hi = sorted(axis.value_range)
    value = np.clip(value, lo, hi)
    return depth, value


# --- offline (no-VLM) digitizers ---------------------------------------------------------------------------


def _trace_curve_offline(arr: np.ndarray, axis: AxisCalib) -> list[dict[str, float]]:
    """Deterministic, VLM-free curve digitizer: for each pixel row spanning the calibrated depth window, find
    the column(s) matching the plotted curve color and take their centroid. Runs whenever no VLM is configured
    or the VLM call fails — production-quality for the common case of one cleanly plotted curve, and exactly
    what the offline test fixture exercises."""
    h, w = arr.shape[:2]
    y0, y1 = sorted((int(round(axis.depth_px[0])), int(round(axis.depth_px[1]))))
    y0 = max(y0, 0)
    y1 = min(y1, h - 1)
    x_lo = max(int(math.floor(min(axis.value_px))) - 3, 0)
    x_hi = min(int(math.ceil(max(axis.value_px))) + 3, w - 1)
    curve = np.array(axis.curve_rgb, dtype=np.int16)
    points: list[dict[str, float]] = []
    for y in range(y0, y1 + 1):
        row = arr[y, x_lo : x_hi + 1].astype(np.int16)
        dist = np.abs(row - curve).sum(axis=1)
        mask = dist <= axis.color_tolerance
        if not mask.any():
            continue
        xs = np.nonzero(mask)[0].astype(np.float64) + x_lo
        points.append({"x": float(xs.mean()), "y": float(y)})
    return points


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Plain BFS 4-connectivity labeling (no scipy dependency — these images are small)."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(comp)
    return components


def _detect_marker_centroids(
    arr: np.ndarray, marker_rgb: tuple[int, int, int], tolerance: float
) -> list[tuple[float, float]]:
    marker = np.array(marker_rgb, dtype=np.int16)
    dist = np.abs(arr.astype(np.int16) - marker).sum(axis=2)
    mask = dist <= tolerance
    centroids = []
    for comp in _connected_components(mask):
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        centroids.append((float(np.mean(xs)), float(np.mean(ys))))
    return centroids


# --- VLM prompting (best-effort; never a hard dependency) --------------------------------------------------

_POLYLINE_PROMPT = (
    "You are digitizing the {track!r} track of a cross-section/log-plot image. Trace the single plotted curve "
    "and reply with ONLY a JSON object of the form "
    '{{"points": [{{"x": <pixel_x>, "y": <pixel_y>}}, ...]}}. '
    "List one point per pixel row from y={y0} to y={y1} inclusive, ordered by increasing y. Coordinates are in "
    "image pixel space with the origin at the top-left corner."
)

_VALUES_PROMPT = (
    "You are locating discrete annotated value markers (labeled at {track!r}) in a cross-section/log-plot image. "
    "Reply with ONLY a JSON object of the form "
    '{{"points": [{{"x": <pixel_x>, "y": <pixel_y>}}, ...]}}, one entry per marker, in image pixel space with '
    "the origin at the top-left corner."
)


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model reply did not contain a JSON object")
    return text[start : end + 1]


def _call_vlm_json(image_bytes: bytes, media_type: str, prompt: str, vlm: str | None) -> dict[str, Any] | None:
    """Best-effort call to a configured chat backend asking for pixel-space JSON. Returns ``None`` on any
    failure (no vlm requested, backend not configured, network/parse error) so the caller falls back to the
    deterministic offline digitizer — a frontier VLM is an accuracy enhancement here, never a hard dependency."""
    if not vlm:
        return None
    try:
        import asyncio

        from ..config import get_settings
        from ..core.adapters import ChatMessage, ChatRequest, ImagePart, TextPart
        from ..models import make_adapter

        settings = get_settings()
        backend = dict((settings.llm_backends or {}).get(vlm, {}))
        adapter = make_adapter(
            vlm, backend, default_base_url=settings.llm_base_url, default_api_key=settings.llm_api_key
        )
        b64 = base64.b64encode(image_bytes).decode("ascii")
        req = ChatRequest(
            model=vlm,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        TextPart(text=prompt),
                        ImagePart(image_url={"url": f"data:{media_type};base64,{b64}"}),
                    ],
                )
            ],
            temperature=0.0,
        )
        completion = asyncio.run(adapter.chat(req))
        text = completion.choices[0].message.text()
        return json.loads(_extract_json_object(text))
    except Exception:
        return None


# --- IC-13 knowledge-item persistence -----------------------------------------------------------------------


def _sha256_of_arrays(arrays: dict[str, np.ndarray]) -> str:
    """Mirrors IC-2's frozen hashing rule (``mixle_pde.io.artifacts.sha256_of_arrays``): sha256 over
    ``arrays[k].tobytes()`` for ``k`` in ``sorted(arrays)``. Reimplemented locally (mixle-mlops does not depend
    on mixle-pde) so a knowledge item's ``content_hash`` is derived the same way a pde field-posterior's is."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


def _build_knowledge_item(
    depth: np.ndarray,
    value: np.ndarray,
    axis: AxisCalib,
    track: str,
    provenance: dict[str, Any],
    source_hash: str,
) -> dict[str, Any]:
    content_hash = _sha256_of_arrays({"depth": depth, "value": value})
    rows = [{"depth": float(d), "value": float(v)} for d, v in zip(depth.tolist(), value.tolist())]
    payload = {
        "primary_key": ["depth"],
        "columns": [
            {"name": "depth", "type": "float", "unit": axis.depth_unit, "dtype": str(depth.dtype)},
            {"name": "value", "type": "float", "unit": axis.value_unit, "dtype": str(value.dtype)},
        ],
        "rows": rows,
        "axis_calibration": axis.to_dict(),
        "track": track,
    }
    n = int(depth.shape[0])
    lo = float(depth.min()) if n else float("nan")
    hi = float(depth.max()) if n else float("nan")
    text_surface = (
        f"{n} {track or 'curve'} point(s) ({axis.value_unit or 'unitless'}) over "
        f"{lo:.2f}-{hi:.2f} {axis.depth_unit} digitized from image {source_hash[:12]}."
    )
    return {
        "schema_uri": TYPED_TABLE_SCHEMA,
        "schema_version": KNOWLEDGE_ITEM_SCHEMA_VERSION,
        "content_hash": content_hash,
        "payload": payload,
        "relations": [{"predicate": "derived_from", "target_id": source_hash}],
        "provenance": dict(provenance),
        "text_surface": text_surface,
    }


def load_depth_series(knowledge_item: dict[str, Any]) -> DepthSeries:
    """Reload a :class:`DepthSeries` from an I2 ``knowledge_item`` dict (e.g. after a JSON round trip). Uses
    only ``payload``/``provenance`` — deleting ``text_surface`` has no effect on the reconstructed arrays/hash."""
    payload = knowledge_item["payload"]
    columns = {c["name"]: c for c in payload["columns"]}
    rows = payload["rows"]
    depth_dtype = columns.get("depth", {}).get("dtype", "float64")
    value_dtype = columns.get("value", {}).get("dtype", "float64")
    depth = np.array([r["depth"] for r in rows], dtype=depth_dtype)
    value = np.array([r["value"] for r in rows], dtype=value_dtype)
    unit = columns.get("value", {}).get("unit") or ""
    provenance = dict(knowledge_item.get("provenance") or {})
    return DepthSeries(depth=depth, value=value, unit=unit, provenance=provenance, knowledge_item=knowledge_item)


# --- public API ----------------------------------------------------------------------------------------------


def extract_curve(
    image_ref: str,
    *,
    axis: AxisCalib,
    track: str,
    vlm: str | None = None,
    store: BlobStore | None = None,
) -> DepthSeries:
    """Digitize one continuous curve (a well-log track, a log-plot panel, a cross-section boundary) into a
    :class:`DepthSeries`. See the module docstring for the full algorithm; ready for I7 to verify."""
    store = store or get_blob_store()
    data, media_type, source_hash = _resolve_image_bytes(image_ref, store)

    y0, y1 = sorted((int(round(axis.depth_px[0])), int(round(axis.depth_px[1]))))
    prompt = _POLYLINE_PROMPT.format(track=track, y0=y0, y1=y1)
    reply = _call_vlm_json(data, media_type, prompt, vlm)

    method = "offline-pixel-trace"
    points: list[dict[str, float]] | None = None
    if reply is not None and isinstance(reply.get("points"), list) and reply["points"]:
        try:
            points = [{"x": float(p["x"]), "y": float(p["y"])} for p in reply["points"]]
            method = "vlm-polyline"
        except (KeyError, TypeError, ValueError):
            points = None
    if points is None:
        points = _trace_curve_offline(_load_image_array(data), axis)

    if not points:
        raise ExtractionError(f"no curve pixels found for track {track!r} within the calibrated axis window")

    depth, value = _pixels_to_series(points, axis)

    provenance = {
        "image_ref": image_ref,
        "source_content_hash": source_hash,
        "source_media_type": media_type,
        "track": track,
        "method": method,
        "vlm": vlm if method == "vlm-polyline" else None,
        "axis": axis.to_dict(),
        "n_points_raw": len(points),
        "n_points": int(depth.shape[0]),
    }
    knowledge_item = _build_knowledge_item(depth, value, axis, track, provenance, source_hash)

    return DepthSeries(
        depth=depth,
        value=value,
        unit=axis.value_unit,
        provenance=provenance,
        knowledge_item=knowledge_item,
    )


def extract_values(
    image_ref: str,
    *,
    axis: AxisCalib,
    track: str = "",
    marker_rgb: tuple[int, int, int] = (200, 30, 30),
    color_tolerance: float = 40.0,
    vlm: str | None = None,
    store: BlobStore | None = None,
) -> list[dict[str, Any]]:
    """Extract discrete point annotations (formation-top picks, scale-check marks, ...) rather than a continuous
    curve. Each result carries a *stable* ``id`` — deterministic from the source image hash, ``track``, and the
    annotation's rounded pixel centroid — so re-running extraction on the same image reproduces the same ids
    (I7 needs that to de-duplicate/verify across repeated runs)."""
    store = store or get_blob_store()
    data, media_type, source_hash = _resolve_image_bytes(image_ref, store)

    prompt = _VALUES_PROMPT.format(track=track or "annotations")
    reply = _call_vlm_json(data, media_type, prompt, vlm)

    method = "offline-marker-blobs"
    centroids: list[tuple[float, float]] | None = None
    if reply is not None and isinstance(reply.get("points"), list) and reply["points"]:
        try:
            centroids = [(float(p["x"]), float(p["y"])) for p in reply["points"]]
            method = "vlm-annotations"
        except (KeyError, TypeError, ValueError):
            centroids = None
    if centroids is None:
        centroids = _detect_marker_centroids(_load_image_array(data), marker_rgb, color_tolerance)

    lo, hi = sorted(axis.value_range)
    results: list[dict[str, Any]] = []
    for x, y in centroids:
        depth, value = _map_point(x, y, axis)
        value = min(max(value, lo), hi)
        annotation_id = hashlib.sha256(
            f"{source_hash}:{track}:{round(x, 2)}:{round(y, 2)}".encode("utf-8")
        ).hexdigest()[:16]
        results.append(
            {
                "id": annotation_id,
                "track": track,
                "depth": depth,
                "value": value,
                "unit": axis.value_unit,
                "pixel": {"x": x, "y": y},
                "provenance": {
                    "image_ref": image_ref,
                    "source_content_hash": source_hash,
                    "method": method,
                    "vlm": vlm if method == "vlm-annotations" else None,
                },
            }
        )
    results.sort(key=lambda r: r["depth"])
    return results
