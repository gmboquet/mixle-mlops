"""I1 -- georeferenced geological-map digitization; I5 -- drillhole plan/collar/trace extraction.

Turns a scanned/rasterized geological map plus a handful of pixel<->CRS control points into named GeoJSON
layers (contacts, faults, mapped units, ...) a physics/GIS pipeline can consume. The heavy lifting is:

  1. resolve the source image bytes from a :class:`~mixle_mlops.multimodal.store.BlobStore` reference and
     retain its content hash (a stable, content-addressed identity for the immutable source image) plus any
     georeferencing metadata; large rasters are tiled through D1's optional ``multimodal.raster`` module
     when it is importable (never a hard import -- see :func:`_load_tiles`).
  2. fit a six-term pixel->CRS affine transform from the supplied ground-control points (least squares; an
     exact solve when exactly three points are given).
  3. call a vision-language model with a structured per-layer schema (:func:`_query_vlm` -- the seam a test
     monkeypatches with a recorded/stubbed response; no live backend is wired here yet).
  4. sanity-check every returned ring/line/point (valid geometry type, enough vertices, closed polygon rings)
     and drop anything degenerate.
  5. reproject the surviving pixel-space geometry through the fitted affine and emit one GeoJSON
     ``FeatureCollection`` per layer.
  6. provenance carries the model id, the source image's content hash, and the target CRS.
  7. the whole digitization is also persisted as an IC-13-shaped ("mixle://schema/property-graph/1") graph
     item under ``provenance["knowledge_item"]``: its ``payload`` retains every Feature's exact id/geometry/
     properties, and a ``derived_from`` relation targets the content-addressed source-image id. A
     ``text_surface`` summary is attached alongside, never inside, that payload.

:func:`extract_collars` (I5) is a sibling extractor over the same kind of map: it reuses an already-fitted
``pixel_to_crs`` affine (rather than fitting its own) to pull drillhole collar markers/hole-id labels and any
plotted plan-view hole traces into a `DrillholeLayer`, merging repeat VLM detections of one collar within a
small pixel tolerance before reprojecting. See the "I5" section below for its own docstrings.

Crosses the mixle-mlops/mixle-pde repo boundary only as data (GeoJSON / plain dicts), per the B1 (``crs.py``)/
B8 (``io/gis.py``) contracts -- this module does not import ``mixle_pde`` at all, and does not import
``mixle_knowledge`` either (IC-13's ``KnowledgeItem``/``PropertyGraphPayload`` classes are not yet landed
there; see the module-level ``PROPERTY_GRAPH_SCHEMA`` note below for the resulting shim).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from .store import BlobStore, get_blob_store

# Mirrors IC-13's frozen `mixle://schema/property-graph/1` constant (mixle-knowledge/contracts.py, workstream
# M). Kept as a local literal rather than a hard dependency: `mixle_knowledge.contracts` does not yet define
# `KnowledgeItem`/`PropertyGraphPayload` (only their supporting enums/refs have landed), so this module builds
# a plain dict shaped to the frozen field names instead of importing a class that does not exist yet -- the
# same judgment call D5 made for `StructuredMediaRef`. Drop-in once mixle-knowledge lands the real models.
PROPERTY_GRAPH_SCHEMA = "mixle://schema/property-graph/1"

_VALID_GEOMETRY_TYPES = {"Point", "LineString", "Polygon"}
_MIN_VERTICES = {"Point": 1, "LineString": 2, "Polygon": 3}


class MapDigitizationError(ValueError):
    """Bad control points, an unusable VLM response, or unsupported geometry."""


@dataclass
class MapDigitization:
    """Result of :func:`digitize_map`.

    ``layers[name]`` is a GeoJSON ``FeatureCollection``; ``pixel_to_crs`` is the fitted six-term affine
    ``(a, b, c, d, e, f)`` such that ``x = a*px + b*py + c`` and ``y = d*px + e*py + f``.
    """

    layers: dict[str, dict] = field(default_factory=dict)
    crs: str = ""
    pixel_to_crs: tuple[float, ...] = ()
    provenance: dict = field(default_factory=dict)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- step 1: source resolution + optional D1 tiling -----------------------------------------------------------


def _load_tiles(data: bytes, content_type: str | None) -> list[bytes]:
    """Split the source image into model-sized tiles via D1's ``raster.tile_raster`` when the media looks
    like a large georeferenced raster (``image/tiff``) and that module is importable; otherwise treat the
    whole image as a single tile. D1 (``mixle_mlops.multimodal.raster``) is a sibling work order -- this stays
    a soft, lazy dependency so ``digitize_map`` works whether or not D1 has landed in the checked-out tree
    (its own non-goal is "no raster tiling engine (D1 owns it)")."""
    is_large_raster = (content_type or "").lower() in {"image/tiff", "image/tif"}
    if is_large_raster:
        try:
            from .raster import tile_raster
        except ImportError:
            pass
        else:
            tiles = tile_raster(data)
            if tiles:
                return [tile.png for tile in tiles]
    return [data]


# --- step 2: affine fit -----------------------------------------------------------------------------------------


def _fit_affine(
    control_points: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float, float, float]:
    """Fit the six-term pixel->CRS affine ``x = a*px + b*py + c``, ``y = d*px + e*py + f`` by least squares.

    Each control point is ``(pixel_x, pixel_y, crs_x, crs_y)``. Three points solve the system exactly; more
    than three are least-squares fit. Raises :class:`MapDigitizationError` with fewer than three points.
    """
    if len(control_points) < 3:
        raise MapDigitizationError(
            f"fitting a pixel->CRS affine needs at least 3 control points, got {len(control_points)}"
        )
    px = np.array([p[0] for p in control_points], dtype=float)
    py = np.array([p[1] for p in control_points], dtype=float)
    cx = np.array([p[2] for p in control_points], dtype=float)
    cy = np.array([p[3] for p in control_points], dtype=float)
    design = np.column_stack([px, py, np.ones_like(px)])
    coeff_x, *_ = np.linalg.lstsq(design, cx, rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, cy, rcond=None)
    a, b, c = (float(v) for v in coeff_x)
    d, e, f = (float(v) for v in coeff_y)
    return (a, b, c, d, e, f)


def _apply_affine(pixel_to_crs: tuple[float, ...], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = pixel_to_crs
    return (a * x + b * y + c, d * x + e * y + f)


# --- step 3: VLM call (the monkeypatch seam) -----------------------------------------------------------------


def _query_vlm(tiles: list[bytes], *, layers: tuple[str, ...], vlm: str | None) -> dict[str, list[dict[str, Any]]]:
    """Prompt a vision-language model with the structured per-layer extraction schema and return
    ``{layer_name: [feature, ...]}`` in pixel space, where each ``feature`` is
    ``{"id", "geometry_type", "pixel_coords", "properties"}``.

    No live backend is wired at this seam yet -- a real integration (gateway model registry + structured
    JSON schema prompt) plugs in at this single call site. Tests monkeypatch this function with a recorded or
    fixture-backed response (see ``tests/fixtures/geomap_stub.json``) to exercise the rest of the pipeline
    deterministically.
    """
    raise NotImplementedError(
        "digitize_map needs a live VLM backend; monkeypatch "
        "mixle_mlops.multimodal.map_digitize._query_vlm (or wire a real model call here) to supply "
        "structured per-layer pixel features."
    )


# --- step 4/5: sanity-check + reproject -----------------------------------------------------------------------


def _polygon_rings(pixel_coords: Any) -> list[list[Sequence[float]]]:
    """Normalize a Polygon's ``pixel_coords`` to a list of rings (each ring a list of ``[x, y]``); accepts
    either a flat single ring or an already-nested list of rings."""
    if not pixel_coords:
        return []
    first = pixel_coords[0]
    if first and isinstance(first[0], (int, float)):
        return [pixel_coords]
    return list(pixel_coords)


def _sanity_check_feature(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Validate one raw pixel-space feature; return it unchanged if usable, else ``None`` (dropped)."""
    geometry_type = raw.get("geometry_type")
    pixel_coords = raw.get("pixel_coords")
    if geometry_type not in _VALID_GEOMETRY_TYPES or not pixel_coords:
        return None
    if geometry_type == "Polygon":
        rings = _polygon_rings(pixel_coords)
        if not rings:
            return None
        ring = rings[0]
        n_unique = len(ring) - (1 if len(ring) > 1 and list(ring[0]) == list(ring[-1]) else 0)
        if n_unique < _MIN_VERTICES["Polygon"]:
            return None
    else:
        if len(pixel_coords) < _MIN_VERTICES[geometry_type]:
            return None
    return raw


def _reproject_geometry(geometry_type: str, pixel_coords: Any, pixel_to_crs: tuple[float, ...]) -> Any:
    if geometry_type == "Point":
        x, y = pixel_coords[0]
        return list(_apply_affine(pixel_to_crs, x, y))
    if geometry_type == "LineString":
        return [list(_apply_affine(pixel_to_crs, x, y)) for x, y in pixel_coords]
    if geometry_type == "Polygon":
        out_rings = []
        for ring in _polygon_rings(pixel_coords):
            coords = [list(_apply_affine(pixel_to_crs, x, y)) for x, y in ring]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            out_rings.append(coords)
        return out_rings
    raise MapDigitizationError(f"unsupported geometry_type {geometry_type!r}")  # pragma: no cover - guarded above


def _build_feature_collection(raw_features: list[dict[str, Any]], pixel_to_crs: tuple[float, ...]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for raw in raw_features:
        checked = _sanity_check_feature(raw)
        if checked is None:
            continue
        coordinates = _reproject_geometry(checked["geometry_type"], checked["pixel_coords"], pixel_to_crs)
        features.append(
            {
                "type": "Feature",
                "id": checked["id"],
                "geometry": {"type": checked["geometry_type"], "coordinates": coordinates},
                "properties": dict(checked.get("properties") or {}),
            }
        )
    return {"type": "FeatureCollection", "features": features}


# --- step 7: IC-13 knowledge item ------------------------------------------------------------------------------


def _source_image_id(content_hash: str) -> str:
    """Stable, content-addressed id for the immutable source image (I1 has no separate ``source_item`` input
    the way I7's ``ingest_extraction`` does, so the source image's own content hash is its identity)."""
    return f"image:{content_hash}"


def _knowledge_item(
    *,
    digitization_id: str,
    layers_geojson: dict[str, dict],
    crs: str,
    vlm: str | None,
    source_image_id: str,
    source_content_hash: str,
) -> dict[str, Any]:
    """Build the IC-13 graph/geospatial item (frozen field names; see the module docstring for why this is a
    plain dict rather than a ``mixle_knowledge.contracts.KnowledgeItem``). ``payload`` retains every Feature's
    exact id/geometry/properties; ``relations`` carries the ``derived_from`` edge to the source image;
    ``text_surface`` is a summary only, never the payload itself."""
    nodes: list[dict[str, Any]] = []
    for layer_name, feature_collection in layers_geojson.items():
        for feature in feature_collection["features"]:
            properties = dict(feature["properties"])
            properties["layer"] = layer_name
            properties["geometry"] = feature["geometry"]
            nodes.append({"id": feature["id"], "type": layer_name, "properties": properties})

    payload = {"nodes": nodes, "edges": []}
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    item_hash = hashlib.sha256(canonical).hexdigest()

    unit_counts = Counter(node["type"] for node in nodes)
    text_surface = "; ".join(f"{count} {name}" for name, count in sorted(unit_counts.items())) or None

    return {
        "id": digitization_id,
        "kind": "artifact",
        "modality": "graph",
        "schema_uri": PROPERTY_GRAPH_SCHEMA,
        "schema_version": "1.0.0",
        "media_type": None,
        "content_hash": item_hash,
        "payload": payload,
        "artifact_ref": None,
        "text_surface": text_surface,
        "provenance": [{"uri": f"vlm://{vlm}" if vlm else "vlm://unset", "sha256": source_content_hash}],
        "relations": [{"predicate": "derived_from", "target_id": source_image_id, "provenance": [], "metadata": {}}],
        "uncertainty": None,
        "metadata": {"crs": crs, "n_features": len(nodes)},
        "access": {},
        "revision": 1,
        "supersedes": [],
        "created_at": _utc_now_iso(),
    }


# --- public API --------------------------------------------------------------------------------------------------


def digitize_map(
    image_ref: str,
    *,
    crs: str,
    control_points: list[tuple[float, float, float, float]],
    layers: tuple[str, ...] = ("contact", "fault", "unit"),
    vlm: str | None = None,
    store: BlobStore | None = None,
) -> MapDigitization:
    """Digitize a georeferenced geological map into named GeoJSON layers.

    ``image_ref`` is a content-addressed :class:`BlobStore` handle (an IC-3-style ``*_ref`` artifact id, see
    the module docstring). ``control_points`` are ``(pixel_x, pixel_y, crs_x, crs_y)`` ground-control tuples
    used to fit the pixel->CRS affine (at least three required). ``crs`` follows IC-4's ``Observation.crs``
    convention (an EPSG/PROJ string).
    """
    store = store or get_blob_store()
    record, data = store.get(image_ref)
    content_hash = hashlib.sha256(data).hexdigest()

    pixel_to_crs = _fit_affine(control_points)
    tiles = _load_tiles(data, record.content_type)

    raw_by_layer = _query_vlm(tiles, layers=layers, vlm=vlm)

    layers_geojson: dict[str, dict] = {
        name: _build_feature_collection(raw_by_layer.get(name, []), pixel_to_crs) for name in layers
    }

    source_image_id = _source_image_id(content_hash)
    digitization_id = f"map-digitization:{content_hash}"
    knowledge_item = _knowledge_item(
        digitization_id=digitization_id,
        layers_geojson=layers_geojson,
        crs=crs,
        vlm=vlm,
        source_image_id=source_image_id,
        source_content_hash=content_hash,
    )

    provenance = {
        "model": vlm,
        "vlm": vlm,
        "source_content_hash": content_hash,
        "source_image_id": source_image_id,
        "crs": crs,
        "n_control_points": len(control_points),
        "layers": list(layers),
        "image_ref": image_ref,
        "knowledge_item": knowledge_item,
    }

    return MapDigitization(
        layers=layers_geojson,
        crs=crs,
        pixel_to_crs=pixel_to_crs,
        provenance=provenance,
    )


# --- I5: drillhole plan/collar/trace extraction ------------------------------------------------------------------
#
# A plan-view geological map often plots drillhole collars (surface entry points, labeled with a hole id and
# sometimes a collar elevation) and, less often, a plan-projected downhole trace. This reuses I1's already-
# fitted ``pixel_to_crs`` affine (the caller passes it in -- unlike ``digitize_map``, `extract_collars` does not
# fit its own control points, matching the pattern I3/I4 use for a downstream extractor over the same map) and
# emits a `DrillholeLayer` a B8 ``io/gis.py`` writer can persist. pde owns turning a plan-view trace into full
# 3D downhole survey geometry (desurveying) -- this stays plan-view pixel/CRS geometry only.


_COLLAR_MERGE_PIXEL_TOLERANCE = 5.0  # px; VLM re-detections of one collar marker land within this radius


@dataclass
class DrillholeLayer:
    """Result of :func:`extract_collars`.

    ``collars[i]`` is ``{"hole_id": str, "x": float, "y": float, "z": float}`` in ``crs`` (``z`` is the
    collar's plotted elevation label, passed through unprojected -- it is not derived from the 2D pixel
    affine). ``traces[i]`` is ``{"hole_id": str, "coordinates": [[x, y], ...]}``, the plan-projected polyline
    for that hole in the same ``crs``, when one was plotted on the map.
    """

    collars: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)
    crs: str = ""
    provenance: dict = field(default_factory=dict)


def _query_vlm_collars(tiles: list[bytes], *, vlm: str | None) -> dict[str, list[dict[str, Any]]]:
    """Prompt a vision-language model for collar markers/hole-id labels and any plotted hole traces; return
    ``{"collars": [...], "traces": [...]}`` in pixel space, where each collar is
    ``{"hole_id", "pixel_coords": [[x, y]], "z"}`` (``z`` optional) and each trace is
    ``{"hole_id", "pixel_coords": [[x, y], ...]}``.

    No live backend is wired at this seam yet, mirroring :func:`_query_vlm` -- tests monkeypatch this function
    with a recorded or fixture-backed response (see ``tests/fixtures/collars_stub.json``).
    """
    raise NotImplementedError(
        "extract_collars needs a live VLM backend; monkeypatch "
        "mixle_mlops.multimodal.map_digitize._query_vlm_collars (or wire a real model call here) to supply "
        "pixel-space collar/trace detections."
    )


def _collar_pixel_point(raw: dict[str, Any]) -> tuple[float, float]:
    pixel_coords = raw.get("pixel_coords") or []
    if not pixel_coords:
        raise MapDigitizationError("collar detection missing pixel_coords")
    x, y = pixel_coords[0]
    return float(x), float(y)


def _cluster_by_pixel_tolerance(points: list[tuple[float, float]], tolerance: float) -> list[list[int]]:
    """Greedy single-linkage clustering of pixel points: a point joins the first existing cluster within
    ``tolerance`` px of that cluster's seed point, else it starts a new cluster. Used to merge repeat VLM
    detections of the same physical collar marker without silently collapsing genuinely distinct ones."""
    clusters: list[list[int]] = []
    for idx, point in enumerate(points):
        for cluster in clusters:
            seed = points[cluster[0]]
            if math.hypot(point[0] - seed[0], point[1] - seed[1]) <= tolerance:
                cluster.append(idx)
                break
        else:
            clusters.append([idx])
    return clusters


def _dedupe_and_reproject_collars(
    raw_collars: list[dict[str, Any]], pixel_to_crs: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Group raw pixel-space collar detections by ``hole_id``, merge near-duplicate detections (nearest-merge
    within :data:`_COLLAR_MERGE_PIXEL_TOLERANCE` px) by averaging their pixel position, then reproject each
    merged collar through ``pixel_to_crs`` into ``{hole_id, x, y, z}``."""
    by_hole: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_collars:
        hole_id = raw.get("hole_id")
        if not hole_id:
            continue
        by_hole.setdefault(hole_id, []).append(raw)

    collars: list[dict[str, Any]] = []
    for hole_id, detections in by_hole.items():
        points = [_collar_pixel_point(d) for d in detections]
        for cluster in _cluster_by_pixel_tolerance(points, _COLLAR_MERGE_PIXEL_TOLERANCE):
            cluster_points = [points[i] for i in cluster]
            cx = sum(p[0] for p in cluster_points) / len(cluster_points)
            cy = sum(p[1] for p in cluster_points) / len(cluster_points)
            x, y = _apply_affine(pixel_to_crs, cx, cy)
            z_values = [detections[i].get("z") for i in cluster if detections[i].get("z") is not None]
            collars.append(
                {
                    "hole_id": hole_id,
                    "x": x,
                    "y": y,
                    "z": float(z_values[0]) if z_values else 0.0,
                }
            )
    return collars


def _reproject_traces(raw_traces: list[dict[str, Any]], pixel_to_crs: tuple[float, ...]) -> list[dict[str, Any]]:
    """Reproject each plan-view drillhole trace's pixel polyline into ``{hole_id, coordinates}``; a trace
    missing a ``hole_id`` or with fewer than two vertices is dropped rather than emitted degenerate."""
    traces: list[dict[str, Any]] = []
    for raw in raw_traces:
        hole_id = raw.get("hole_id")
        pixel_coords = raw.get("pixel_coords") or []
        if not hole_id or len(pixel_coords) < 2:
            continue
        coordinates = [list(_apply_affine(pixel_to_crs, x, y)) for x, y in pixel_coords]
        traces.append({"hole_id": hole_id, "coordinates": coordinates})
    return traces


def extract_collars(
    image_ref: str,
    *,
    pixel_to_crs: tuple[float, ...],
    crs: str,
    vlm: str | None = None,
    store: BlobStore | None = None,
) -> DrillholeLayer:
    """Extract drillhole collars (and any plotted plan-view traces) from a georeferenced map image.

    ``pixel_to_crs`` is the already-fitted six-term affine from :func:`digitize_map` (or an equivalent I6
    georeferencing pass) -- this function reprojects but does not itself fit an affine. ``crs`` follows IC-4's
    ``Observation.crs`` convention. Collars are de-duplicated by ``hole_id``, merging repeat VLM detections
    that land within a small pixel tolerance of each other.
    """
    store = store or get_blob_store()
    record, data = store.get(image_ref)
    content_hash = hashlib.sha256(data).hexdigest()
    tiles = _load_tiles(data, record.content_type)

    raw = _query_vlm_collars(tiles, vlm=vlm)
    collars = _dedupe_and_reproject_collars(raw.get("collars") or [], pixel_to_crs)
    traces = _reproject_traces(raw.get("traces") or [], pixel_to_crs)

    provenance = {
        "vlm": vlm,
        "image_content_hash": content_hash,
        "n_collars": len(collars),
        "image_ref": image_ref,
        "crs": crs,
    }

    return DrillholeLayer(collars=collars, traces=traces, crs=crs, provenance=provenance)
