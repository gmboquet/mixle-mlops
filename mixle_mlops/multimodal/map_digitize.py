"""I1 -- georeferenced geological-map digitization.

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

Crosses the mixle-mlops/mixle-pde repo boundary only as data (GeoJSON), per the B1 (``crs.py``)/B8
(``io/gis.py``) contracts -- this module does not import ``mixle_pde`` at all, and does not import
``mixle_knowledge`` either (IC-13's ``KnowledgeItem``/``PropertyGraphPayload`` classes are not yet landed
there; see the module-level ``PROPERTY_GRAPH_SCHEMA`` note below for the resulting shim).

I3 -- legend / symbol / annotation parsing -- adds two more functions on top of the above, without touching
any of I1's existing logic:

  * :func:`parse_legend` reads a map's legend box (``{swatch_color: unit_name}``), any borehole/strike-dip
    symbol glyphs, and the scale bar, through the same VLM-seam pattern as :func:`_query_vlm` (see
    :func:`_query_legend_vlm`).
  * :func:`apply_legend` binds each digitized polygon's dominant swatch color to the nearest legend unit
    (RGB L2) and, when the legend read a scale bar, cross-checks it against the digitization's own
    control-point affine, recording the comparison in provenance rather than raising -- a soft cross-check,
    not a georef-quality gate (that hard-fail behavior belongs to I6's ``georeference``).
  * ``digitize_map`` gained one optional keyword, ``legend: bool = False``: when set, it runs the same image
    through :func:`parse_legend` and folds the result in via :func:`apply_legend` before returning. This is
    a small, additive post-pass -- the rest of the function body is unchanged from I1.
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

from ..config import get_settings
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
    legend: bool = False,
) -> MapDigitization:
    """Digitize a georeferenced geological map into named GeoJSON layers.

    ``image_ref`` is a content-addressed :class:`BlobStore` handle (an IC-3-style ``*_ref`` artifact id, see
    the module docstring). ``control_points`` are ``(pixel_x, pixel_y, crs_x, crs_y)`` ground-control tuples
    used to fit the pixel->CRS affine (at least three required). ``crs`` follows IC-4's ``Observation.crs``
    convention (an EPSG/PROJ string). ``legend=True`` runs the I3 post-pass (:func:`parse_legend` +
    :func:`apply_legend`) on the same image before returning, binding each digitized polygon to a named
    legend unit.
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

    result = MapDigitization(
        layers=layers_geojson,
        crs=crs,
        pixel_to_crs=pixel_to_crs,
        provenance=provenance,
    )

    if legend:
        result = apply_legend(result, parse_legend(image_ref, store=store))

    return result


# --- I3: legend / symbol / annotation parsing ----------------------------------------------------------------


@dataclass
class LegendMap:
    """Result of :func:`parse_legend`.

    ``units`` maps a legend swatch's dominant color (a ``"#RRGGBB"`` hex string -- the "swatch label" the
    legend key is read by) to the semantic unit name it labels (e.g. ``"#4a7d3c": "Qal"``); :func:`apply_legend`
    matches each digitized polygon's own dominant color against these swatches by RGB L2 distance. ``symbols``
    maps a recognized glyph name to its semantic type (e.g. ``"circle-open": "borehole"``,
    ``"tick-strike": "strike_dip"``) -- read but not otherwise interpreted here (no new symbol taxonomy beyond
    what the legend itself defines). ``scale_m_per_px`` is ``None`` when no scale bar was found/read.
    """

    units: dict[str, str] = field(default_factory=dict)
    symbols: dict[str, str] = field(default_factory=dict)
    scale_m_per_px: float | None = None
    provenance: dict = field(default_factory=dict)


def _query_legend_vlm(tiles: list[bytes], *, vlm: str | None) -> dict[str, Any]:
    """Prompt a vision-language model to read the legend box, symbol glyphs, and scale bar; return
    ``{"units": {swatch_color: unit_name, ...}, "symbols": {glyph: semantic_type, ...},
    "scale_m_per_px": float | None}``.

    Same seam pattern as :func:`_query_vlm`: no live backend is wired here yet. Tests monkeypatch this
    function with a recorded/stubbed response (see ``tests/fixtures/legend_stub.json``).
    """
    raise NotImplementedError(
        "parse_legend needs a live VLM backend; monkeypatch "
        "mixle_mlops.multimodal.map_digitize._query_legend_vlm (or wire a real model call here) to supply "
        "the legend/symbol/scale-bar reading."
    )


def parse_legend(image_ref: str, *, store: BlobStore | None = None) -> LegendMap:
    """Read a map's legend box, symbol glyphs, and scale bar into a :class:`LegendMap`.

    ``image_ref`` is the same kind of content-addressed :class:`BlobStore` handle :func:`digitize_map` takes
    (usually the same map image). The VLM model id used is the platform's configured default model
    (``Settings.default_model``) -- unlike :func:`digitize_map`, this signature takes no ``vlm`` override, so
    provenance records whichever default was active when the legend was read.
    """
    store = store or get_blob_store()
    record, data = store.get(image_ref)
    content_hash = hashlib.sha256(data).hexdigest()
    tiles = _load_tiles(data, record.content_type)

    vlm = get_settings().default_model
    raw = _query_legend_vlm(tiles, vlm=vlm)

    units = {str(k): str(v) for k, v in dict(raw.get("units") or {}).items()}
    symbols = {str(k): str(v) for k, v in dict(raw.get("symbols") or {}).items()}
    raw_scale = raw.get("scale_m_per_px")
    scale_m_per_px = float(raw_scale) if raw_scale is not None else None

    provenance = {
        "vlm": vlm,
        "image_content_hash": content_hash,
        "n_units": len(units),
        "scale_source": "legend_scale_bar" if scale_m_per_px is not None else None,
    }
    return LegendMap(units=units, symbols=symbols, scale_m_per_px=scale_m_per_px, provenance=provenance)


def _hex_to_rgb(color: str) -> tuple[float, float, float] | None:
    """Parse a ``"#RRGGBB"`` (or ``"RRGGBB"``) string into an ``(r, g, b)`` triple; ``None`` if unparseable."""
    text = color.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(float(int(text[i : i + 2], 16)) for i in (0, 2, 4))
    except ValueError:
        return None


def _nearest_unit(color: str, units: dict[str, str]) -> str | None:
    """The unit name of the legend swatch nearest ``color`` by RGB L2 distance; ``None`` if neither the
    polygon color nor any swatch color is parseable."""
    target = _hex_to_rgb(color)
    if target is None:
        return None
    best_name: str | None = None
    best_dist: float | None = None
    for swatch_color, unit_name in units.items():
        swatch_rgb = _hex_to_rgb(swatch_color)
        if swatch_rgb is None:
            continue
        dist = sum((t - s) ** 2 for t, s in zip(target, swatch_rgb))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_name = unit_name
    return best_name


def _affine_scale_m_per_px(pixel_to_crs: tuple[float, ...]) -> float | None:
    """The ground scale (meters per pixel) implied by a fitted ``pixel_to_crs`` affine: the average length,
    in CRS units, of the two pixel-axis step vectors ``(a, d)`` and ``(b, e)`` -- exact for an axis-aligned or
    uniformly-rotated raster with square pixels."""
    if len(pixel_to_crs) != 6:
        return None
    a, b, _c, d, e, _f = pixel_to_crs
    return (math.hypot(a, d) + math.hypot(b, e)) / 2.0


def apply_legend(dig: MapDigitization, legend: LegendMap) -> MapDigitization:
    """Bind each digitized polygon's dominant swatch color to its nearest legend unit (RGB L2), and, when the
    legend read a scale bar, cross-check it against ``dig``'s own control-point affine.

    A polygon `Feature` gains ``properties["unit"]`` when its properties already carry a ``"color"`` string
    (the dominant swatch color the digitization step read off the map) and the legend has at least one usable
    swatch; features without a ``"color"`` property, or non-polygon features, pass through unchanged. The
    scale cross-check is recorded under ``provenance["legend"]["scale_check"]`` -- a soft comparison (the
    ratio of the legend-read scale to the affine-implied one); it never raises (I6's ``georeference`` owns
    hard-failing a bad georeference). Returns a new :class:`MapDigitization`; neither input is mutated.
    """
    new_layers: dict[str, dict] = {}
    n_bound = 0
    for layer_name, feature_collection in dig.layers.items():
        features: list[dict[str, Any]] = []
        for raw_feature in feature_collection.get("features", []):
            properties = dict(raw_feature.get("properties") or {})
            geometry = raw_feature.get("geometry") or {}
            color = properties.get("color")
            if geometry.get("type") == "Polygon" and legend.units and isinstance(color, str):
                unit_name = _nearest_unit(color, legend.units)
                if unit_name is not None:
                    properties["unit"] = unit_name
                    n_bound += 1
            features.append({**raw_feature, "properties": properties})
        new_layers[layer_name] = {
            "type": feature_collection.get("type", "FeatureCollection"),
            "features": features,
        }

    scale_check: dict[str, Any] | None = None
    if legend.scale_m_per_px is not None:
        affine_scale = _affine_scale_m_per_px(dig.pixel_to_crs)
        scale_check = {
            "legend_scale_m_per_px": legend.scale_m_per_px,
            "affine_scale_m_per_px": affine_scale,
            "ratio": (legend.scale_m_per_px / affine_scale) if affine_scale else None,
        }

    provenance = dict(dig.provenance)
    provenance["legend"] = {
        **legend.provenance,
        "n_units_bound": n_bound,
        "scale_check": scale_check,
    }

    return MapDigitization(
        layers=new_layers,
        crs=dig.crs,
        pixel_to_crs=dig.pixel_to_crs,
        provenance=provenance,
    )
