"""I1 -- georeferenced geological-map digitization.

Exercises the whole pipeline against a stubbed VLM response (``tests/fixtures/geomap_stub.json``) so it is
deterministic and needs no live model call: fit the pixel->CRS affine from four known corner control points,
reproject the stub's pixel-space fault/contact/unit features, and check the reprojected fault registers to
the expected UTM coordinates within tolerance. Also checks that the IC-13-shaped knowledge item persisted
alongside the digitization round-trips through JSON unchanged and keeps its source-image ``derived_from``
relation even with its text surface stripped.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from mixle_mlops.multimodal import map_digitize
from mixle_mlops.multimodal.map_digitize import MapDigitization, digitize_map
from mixle_mlops.multimodal.store import LocalBlobStore

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "geomap_stub.json").read_text())

# a 1x1 transparent PNG -- a stand-in map image; the "VLM" response is fully stubbed via monkeypatch below,
# so the actual pixel content never matters to this test.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Four exact corners of a 100x100 px, north-up raster at 1 m/px: row 0 is the north edge (northing 4100100),
# row 100 is the south edge (northing 4100000). This is solvable exactly (no least-squares residual), so the
# reprojected fault line can be checked against an exact expectation.
_CONTROL_POINTS = [
    (0.0, 0.0, 500000.0, 4100100.0),
    (100.0, 0.0, 500100.0, 4100100.0),
    (0.0, 100.0, 500000.0, 4100000.0),
    (100.0, 100.0, 500100.0, 4100000.0),
]


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(root=tmp_path / "blobs")


def _put_image(blob_store) -> str:
    record = blob_store.put(_PNG, filename="geomap.png", content_type="image/png")
    return record.id


def test_fault_registers_to_utm(store, monkeypatch):
    monkeypatch.setattr(map_digitize, "_query_vlm", lambda tiles, *, layers, vlm: FIXTURE)

    image_ref = _put_image(store)
    result = digitize_map(
        image_ref=image_ref,
        crs="EPSG:32611",
        control_points=_CONTROL_POINTS,
        store=store,
        vlm="stub-vlm",
    )

    assert isinstance(result, MapDigitization)
    assert result.crs == "EPSG:32611"
    assert len(result.pixel_to_crs) == 6

    fault_fc = result.layers["fault"]
    assert fault_fc["type"] == "FeatureCollection"
    assert len(fault_fc["features"]) == 1
    feature = fault_fc["features"][0]
    assert feature["id"] == "fault-1"

    # pixel (10,10) -> (500010, 4100090); pixel (90,90) -> (500090, 4100010)
    expected = [(500010.0, 4100090.0), (500090.0, 4100010.0)]
    coords = feature["geometry"]["coordinates"]
    assert len(coords) == len(expected)
    for (x, y), (ex, ey) in zip(coords, expected):
        assert abs(x - ex) < 2.0
        assert abs(y - ey) < 2.0

    # the degenerate unit polygon in the fixture (a single repeated pixel) must be dropped, not crash the run
    unit_ids = {f["id"] for f in result.layers["unit"]["features"]}
    assert unit_ids == {"unit-1"}

    source_hash = result.provenance["source_content_hash"]
    assert len(source_hash) == 64
    assert all(c in "0123456789abcdef" for c in source_hash)

    # --- IC-13 knowledge-item round trip -------------------------------------------------------------------
    item = result.provenance["knowledge_item"]
    restored = json.loads(json.dumps(item))

    fault_node = next(n for n in restored["payload"]["nodes"] if n["id"] == "fault-1")
    assert fault_node["properties"]["geometry"]["coordinates"] == coords
    assert restored["metadata"]["crs"] == "EPSG:32611"

    relation = restored["relations"][0]
    assert relation["predicate"] == "derived_from"
    assert relation["target_id"] == result.provenance["source_image_id"]

    # stripping the text surface must not touch the payload or the source-image relation
    restored["text_surface"] = None
    assert restored["relations"][0]["target_id"] == relation["target_id"]
    assert restored["payload"]["nodes"] == item["payload"]["nodes"]


def test_affine_requires_at_least_three_control_points(store, monkeypatch):
    monkeypatch.setattr(map_digitize, "_query_vlm", lambda tiles, *, layers, vlm: FIXTURE)
    image_ref = _put_image(store)
    with pytest.raises(ValueError):
        digitize_map(
            image_ref=image_ref,
            crs="EPSG:32611",
            control_points=_CONTROL_POINTS[:2],
            store=store,
        )


def test_layers_default_to_contact_fault_unit(store, monkeypatch):
    monkeypatch.setattr(map_digitize, "_query_vlm", lambda tiles, *, layers, vlm: FIXTURE)
    image_ref = _put_image(store)
    result = digitize_map(
        image_ref=image_ref,
        crs="EPSG:32611",
        control_points=_CONTROL_POINTS,
        store=store,
    )
    assert set(result.layers) == {"contact", "fault", "unit"}
