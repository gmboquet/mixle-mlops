"""I1 -- georeferenced geological-map digitization; I3 -- legend / symbol / annotation parsing; I5 --
drillhole plan/collar/trace extraction.

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


def _extract_json_object(text: str) -> str:
    """Pull the first ``{...}`` JSON object out of a model reply, tolerating markdown fences/prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model reply did not contain a JSON object")
    return text[start : end + 1]


def _map_extraction_prompt(layers: tuple[str, ...], width: int, height: int) -> str:
    return (
        "You are digitizing a geological map image. For EACH of these layers, extract the vector "
        f"features you can see: {list(layers)}. The image is {width} pixels wide and {height} tall; "
        "give every coordinate as [x, y] pixel positions (origin top-left). Reply with ONLY a JSON "
        "object of the form {\"<layer>\": [{\"id\": str, \"geometry_type\": \"Point\"|\"LineString\"|"
        "\"Polygon\", \"pixel_coords\": [[x,y], ...], \"properties\": {..}}, ...], ...}. A Point has a "
        "single [x,y]; a LineString a list of [x,y]; a Polygon a closed ring of [x,y]. Include only "
        "features you actually see; use an empty list for a layer with none. No prose, only the JSON."
    )


def _resolve_vision_adapter(vlm: str | None) -> Any:
    """Resolve a vision adapter through the capability layer -- ``vlm`` (if given) overrides the model
    that serves the ``vision`` capability; otherwise the configured default is used. Raises a clear
    error (from :func:`~mixle_mlops.models.capabilities.resolve_adapter`) if no vision backend is
    configured -- digitizing a map genuinely requires a VLM, so this fails loud rather than silently."""
    from ..models.capabilities import VISION, resolve_from_settings

    override = {VISION: vlm} if vlm else None
    return resolve_from_settings(VISION, capability_models=override)


def _run_vision_json(adapter: Any, model_id: str, image_bytes: bytes, prompt: str) -> dict[str, Any]:
    import asyncio
    import base64

    from ..core.adapters import ChatMessage, ChatRequest, ImagePart, TextPart

    b64 = base64.b64encode(image_bytes).decode("ascii")
    req = ChatRequest(
        model=model_id,
        messages=[ChatMessage(role="user", content=[
            TextPart(text=prompt),
            ImagePart(image_url={"url": f"data:image/png;base64,{b64}"}),
        ])],
        temperature=0.0,
    )
    completion = asyncio.run(adapter.chat(req))
    text = completion.choices[0].message.text()
    return json.loads(_extract_json_object(text))


def _query_vlm(tiles: list[bytes], *, layers: tuple[str, ...], vlm: str | None) -> dict[str, list[dict[str, Any]]]:
    """Prompt a vision-language model with the structured per-layer extraction schema and return
    ``{layer_name: [feature, ...]}`` in pixel space, where each ``feature`` is
    ``{"id", "geometry_type", "pixel_coords", "properties"}``.

    Wired through the capability-routed model layer (:mod:`mixle_mlops.models.capabilities`): it asks
    for the ``vision`` capability (or the model named by ``vlm``), sends the map tile plus the
    structured-JSON prompt, and parses the reply. Tests may still monkeypatch this whole function with a
    recorded fixture (``tests/fixtures/geomap_stub.json``) for deterministic, network-free runs.

    Single-tile is handled today (the common case after raster windowing); multi-tile pixel-coordinate
    stitching across tiles is a documented follow-up -- only ``tiles[0]`` is sent, and the caller's
    downstream sanity-check + reproject steps still apply to whatever the VLM returns (a real,
    first-line guard against VLM feature hallucination; a fuller calibration/verification gate on VLM
    output is the next step)."""
    if not tiles:
        return {layer: [] for layer in layers}
    adapter = _resolve_vision_adapter(vlm)
    from PIL import Image
    import io as _io

    width, height = Image.open(_io.BytesIO(tiles[0])).size
    prompt = _map_extraction_prompt(layers, width, height)
    raw = _run_vision_json(adapter, vlm or adapter.name, tiles[0], prompt)
    if not isinstance(raw, dict):
        return {layer: [] for layer in layers}
    return {layer: list(raw.get(layer, []) or []) for layer in layers}


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
