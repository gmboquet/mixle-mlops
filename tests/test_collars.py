"""I5 -- drillhole plan/collar/trace extraction.

Exercises `extract_collars` against a stubbed VLM response (`tests/fixtures/collars_stub.json`) so it is
deterministic and needs no live model call: reproject pixel-space collar markers and plotted hole traces
through an already-fitted (I1-style) pixel->CRS affine, merge the two near-duplicate detections the stub
gives for "DDH-01" (a re-detection artifact 1-2 px apart) into a single collar, and check every collar
registers to the expected UTM coordinates in `tests/fixtures/collars_truth.csv` within 5 m.
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import pytest

from mixle_mlops.multimodal import map_digitize
from mixle_mlops.multimodal.map_digitize import DrillholeLayer, extract_collars
from mixle_mlops.multimodal.store import LocalBlobStore

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "collars_stub.json").read_text())

with open(Path(__file__).parent / "fixtures" / "collars_truth.csv", newline="") as fh:
    TRUTH = {row["hole_id"]: row for row in csv.DictReader(fh)}

# same 100x100 px, north-up, 1 m/px raster as test_map_digitize.py's control points: pixel (0,0) -> UTM
# (500000, 4100100), y flips down the page. Exact affine (no least-squares residual): x = px + 500000,
# y = -py + 4100100.
_PIXEL_TO_CRS = (1.0, 0.0, 500000.0, 0.0, -1.0, 4100100.0)

# a 1x1 transparent PNG -- a stand-in drillhole-plan image; the "VLM" response is fully stubbed via
# monkeypatch below, so the actual pixel content never matters to this test.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(root=tmp_path / "blobs")


def _put_image(blob_store) -> str:
    record = blob_store.put(_PNG, filename="collar-plan.png", content_type="image/png")
    return record.id


def test_collars_register_to_crs(store, monkeypatch):
    monkeypatch.setattr(map_digitize, "_query_vlm_collars", lambda tiles, *, vlm: FIXTURE)

    image_ref = _put_image(store)
    result = extract_collars(
        image_ref=image_ref,
        pixel_to_crs=_PIXEL_TO_CRS,
        crs="EPSG:32611",
        store=store,
        vlm="stub-vlm",
    )

    assert isinstance(result, DrillholeLayer)
    assert result.crs == "EPSG:32611"

    # the two near-duplicate DDH-01 detections merge into a single collar
    assert {c["hole_id"] for c in result.collars} == set(TRUTH)
    assert len(result.collars) == len(TRUTH)

    by_hole = {c["hole_id"]: c for c in result.collars}
    for hole_id, truth_row in TRUTH.items():
        collar = by_hole[hole_id]
        assert abs(collar["x"] - float(truth_row["x"])) < 5.0
        assert abs(collar["y"] - float(truth_row["y"])) < 5.0
        assert abs(collar["z"] - float(truth_row["z"])) < 1e-6

    assert result.provenance["n_collars"] == len(TRUTH)
    assert len(result.provenance["image_content_hash"]) == 64

    # plotted traces reproject too, tagged by hole_id
    trace_ids = {t["hole_id"] for t in result.traces}
    assert trace_ids == {"DDH-01", "DDH-02"}
    ddh1_trace = next(t for t in result.traces if t["hole_id"] == "DDH-01")
    assert ddh1_trace["coordinates"][0] == pytest.approx([500020.0, 4100080.0])


def test_collars_require_hole_id(store, monkeypatch):
    monkeypatch.setattr(
        map_digitize,
        "_query_vlm_collars",
        lambda tiles, *, vlm: {"collars": [{"pixel_coords": [[5, 5]], "z": 1000.0}], "traces": []},
    )
    image_ref = _put_image(store)
    result = extract_collars(
        image_ref=image_ref,
        pixel_to_crs=_PIXEL_TO_CRS,
        crs="EPSG:32611",
        store=store,
    )
    assert result.collars == []
