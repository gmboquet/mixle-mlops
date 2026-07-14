"""I3 -- legend / symbol / annotation parsing.

Exercises :func:`parse_legend` against a stubbed VLM response (``tests/fixtures/legend_stub.json``) and
:func:`apply_legend` against a hand-built :class:`MapDigitization` with three colored unit polygons, so the
whole pipeline is deterministic and needs no live model call.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from mixle_mlops.multimodal import map_digitize
from mixle_mlops.multimodal.map_digitize import (
    LegendMap,
    MapDigitization,
    apply_legend,
    digitize_map,
    parse_legend,
)
from mixle_mlops.multimodal.store import LocalBlobStore

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legend_stub.json").read_text())
MAP_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "geomap_stub.json").read_text())

# a 1x1 transparent PNG -- a stand-in map image; the "VLM" responses are fully stubbed via monkeypatch below,
# so the actual pixel content never matters to these tests.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# The same 100x100 px, 1 m/px, north-up raster I1's fixture uses.
_CONTROL_POINTS = [
    (0.0, 0.0, 500000.0, 4100100.0),
    (100.0, 0.0, 500100.0, 4100100.0),
    (0.0, 100.0, 500000.0, 4100000.0),
    (100.0, 100.0, 500100.0, 4100000.0),
]
_PIXEL_TO_CRS = (1.0, 0.0, 500000.0, 0.0, -1.0, 4100100.0)

# Three unit polygons, each colored a slight shade off one of the three legend swatches -- exercises nearest
# (not exact-match) RGB L2 binding.
_UNIT_FEATURES = [
    {
        "type": "Feature",
        "id": "unit-alluvium",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
        "properties": {"label": "alluvium", "color": "#4b7e3d"},  # near "#4a7d3c" -> Qal
    },
    {
        "type": "Feature",
        "id": "unit-volcanics",
        "geometry": {"type": "Polygon", "coordinates": [[[10, 0], [20, 0], [20, 10], [10, 10], [10, 0]]]},
        "properties": {"label": "volcanics", "color": "#caa228"},  # near "#c9a227" -> Tv
    },
    {
        "type": "Feature",
        "id": "unit-granite",
        "geometry": {"type": "Polygon", "coordinates": [[[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]]},
        "properties": {"label": "granite", "color": "#8998ab"},  # near "#8899aa" -> Kgd
    },
]


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(root=tmp_path / "blobs")


def _put_image(blob_store) -> str:
    record = blob_store.put(_PNG, filename="legend.png", content_type="image/png")
    return record.id


def test_legend_binds_units_and_scale(store, monkeypatch):
    monkeypatch.setattr(map_digitize, "_query_legend_vlm", lambda tiles, *, vlm: FIXTURE)

    image_ref = _put_image(store)
    legend = parse_legend(image_ref, store=store)

    assert isinstance(legend, LegendMap)
    assert legend.units == FIXTURE["units"]
    assert legend.symbols == FIXTURE["symbols"]
    assert legend.scale_m_per_px == pytest.approx(1.02)
    assert legend.provenance["n_units"] == 3
    assert legend.provenance["scale_source"] == "legend_scale_bar"
    assert len(legend.provenance["image_content_hash"]) == 64

    dig = MapDigitization(
        layers={"unit": {"type": "FeatureCollection", "features": _UNIT_FEATURES}},
        crs="EPSG:32611",
        pixel_to_crs=_PIXEL_TO_CRS,
        provenance={},
    )

    result = apply_legend(dig, legend)

    assert isinstance(result, MapDigitization)
    bound = {f["id"]: f["properties"]["unit"] for f in result.layers["unit"]["features"]}
    assert bound == {
        "unit-alluvium": "Qal",
        "unit-volcanics": "Tv",
        "unit-granite": "Kgd",
    }
    assert result.provenance["legend"]["n_units_bound"] == 3

    # scale_m_per_px (1.02) cross-checked against the control-point affine's exact 1 m/px scale, within 5%
    scale_check = result.provenance["legend"]["scale_check"]
    assert scale_check["affine_scale_m_per_px"] == pytest.approx(1.0)
    assert abs(scale_check["ratio"] - 1.0) < 0.05

    # the original digitization is not mutated
    assert dig.layers["unit"]["features"][0]["properties"] == {"label": "alluvium", "color": "#4b7e3d"}


def test_apply_legend_skips_features_without_color_or_geometry(store, monkeypatch):
    monkeypatch.setattr(map_digitize, "_query_legend_vlm", lambda tiles, *, vlm: FIXTURE)
    image_ref = _put_image(store)
    legend = parse_legend(image_ref, store=store)

    dig = MapDigitization(
        layers={
            "fault": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "fault-1",
                        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                        "properties": {"kind": "normal"},
                    }
                ],
            },
            "unit": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "unit-no-color",
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"label": "unknown"},
                    }
                ],
            },
        },
        crs="EPSG:32611",
        pixel_to_crs=_PIXEL_TO_CRS,
        provenance={},
    )

    result = apply_legend(dig, legend)

    assert "unit" not in result.layers["fault"]["features"][0]["properties"]
    assert "unit" not in result.layers["unit"]["features"][0]["properties"]
    assert result.provenance["legend"]["n_units_bound"] == 0


def test_digitize_map_legend_post_pass(store, monkeypatch):
    """``digitize_map(..., legend=True)`` runs the same image through parse_legend/apply_legend."""
    monkeypatch.setattr(map_digitize, "_query_vlm", lambda tiles, *, layers, vlm: MAP_FIXTURE)
    monkeypatch.setattr(map_digitize, "_query_legend_vlm", lambda tiles, *, vlm: FIXTURE)

    image_ref = _put_image(store)
    result = digitize_map(
        image_ref=image_ref,
        crs="EPSG:32611",
        control_points=_CONTROL_POINTS,
        store=store,
        vlm="stub-vlm",
        legend=True,
    )

    assert "legend" in result.provenance
    assert result.provenance["legend"]["n_units"] == 3


def test_parse_legend_handles_missing_scale_bar(store, monkeypatch):
    no_scale = {"units": {"#000000": "Fill"}, "symbols": {}, "scale_m_per_px": None}
    monkeypatch.setattr(map_digitize, "_query_legend_vlm", lambda tiles, *, vlm: no_scale)
    image_ref = _put_image(store)

    legend = parse_legend(image_ref, store=store)
    assert legend.scale_m_per_px is None
    assert legend.provenance["scale_source"] is None
