"""I2 -- cross-section & log-plot curve/value extraction.

``test_curve_matches_las`` is the Definition-of-Done test: digitize the synthetic resistivity ("log-plot")
fixture with no VLM configured (exercising the offline pixel-tracing fallback exclusively), check the recovered
values track the fixture's ground-truth curve to within 3% RMS error, and check the IC-13 ``knowledge_item``
survives a JSON round trip -- exact depth/value arrays, dtype, unit, axis calibration, and source-image hash --
even after its ``text_surface`` summary is deleted. (The fixture's "truth" is a CSV, not a parsed LAS file --
LAS parsing is out of scope here, owned by mixle-pde B3; the CSV plays the role a parsed LAS curve would.)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mixle_mlops.multimodal.extract import (
    AxisCalib,
    DepthSeries,
    ExtractionError,
    extract_curve,
    extract_values,
    load_depth_series,
)
from mixle_mlops.multimodal.store import LocalBlobStore

FIXTURES = Path(__file__).parent / "fixtures"


def _truth() -> tuple[np.ndarray, np.ndarray]:
    depths, values = [], []
    with open(FIXTURES / "log_track_truth.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            depths.append(float(row["depth"]))
            values.append(float(row["value"]))
    return np.array(depths), np.array(values)


def _axis() -> AxisCalib:
    return AxisCalib(
        depth_px=(40.0, 360.0),
        depth_range=(1000.0, 1080.0),
        value_px=(20.0, 420.0),
        value_range=(0.2, 2000.0),
        value_log=True,
        depth_unit="m",
        value_unit="ohm.m",
        curve_rgb=(20, 90, 160),
    )


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(tmp_path / "blobs")


def _put_fixture(store) -> str:
    data = (FIXTURES / "log_track_stub.png").read_bytes()
    record = store.put(data, filename="log_track_stub.png", content_type="image/png")
    return record.id


def test_curve_matches_las(store):
    blob_id = _put_fixture(store)

    series = extract_curve(blob_id, axis=_axis(), track="resistivity", vlm=None, store=store)

    assert isinstance(series, DepthSeries)
    assert series.provenance["method"] == "offline-pixel-trace"  # no vlm configured -> offline path only
    assert series.unit == "ohm.m"

    truth_depth, truth_value = _truth()
    assert series.depth.shape == truth_depth.shape
    assert np.allclose(series.depth, truth_depth, atol=1e-6)

    rel_err = (series.value - truth_value) / truth_value
    rel_rmse = math.sqrt(float(np.mean(rel_err**2)))
    assert rel_rmse < 0.03, f"relative RMSE {rel_rmse:.4%} exceeds the 3% DoD budget"

    # --- IC-13 reload: JSON round trip preserves every depth/value, dtype/unit/axis calibration + source hash ---
    item = series.knowledge_item
    assert item["schema_uri"] == "mixle://schema/typed-table/1"
    source_hash = item["provenance"]["source_content_hash"]
    assert len(source_hash) == 64 and all(c in "0123456789abcdef" for c in source_hash)

    wire = json.loads(json.dumps(item))  # simulates a store/reload round trip
    reloaded = load_depth_series(wire)

    assert np.array_equal(reloaded.depth, series.depth)
    assert np.array_equal(reloaded.value, series.value)
    assert reloaded.depth.dtype == series.depth.dtype
    assert reloaded.value.dtype == series.value.dtype
    assert reloaded.unit == series.unit
    assert reloaded.knowledge_item["payload"]["axis_calibration"] == item["payload"]["axis_calibration"]
    assert reloaded.provenance["source_content_hash"] == source_hash

    # --- deleting the summary must not affect arrays/hash ---
    assert "text_surface" in wire
    del wire["text_surface"]
    reloaded2 = load_depth_series(wire)
    assert np.array_equal(reloaded2.depth, series.depth)
    assert np.array_equal(reloaded2.value, series.value)
    assert reloaded2.knowledge_item["content_hash"] == item["content_hash"]
    assert reloaded2.provenance["source_content_hash"] == source_hash


def test_extract_curve_is_deterministic_across_repeated_calls(store):
    blob_id = _put_fixture(store)
    a = extract_curve(blob_id, axis=_axis(), track="resistivity", vlm=None, store=store)
    b = extract_curve(blob_id, axis=_axis(), track="resistivity", vlm=None, store=store)
    assert np.array_equal(a.depth, b.depth)
    assert np.array_equal(a.value, b.value)
    assert a.knowledge_item["content_hash"] == b.knowledge_item["content_hash"]


def test_extract_curve_accepts_a_data_url_without_a_store():
    data = (FIXTURES / "log_track_stub.png").read_bytes()
    import base64

    data_url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    series = extract_curve(data_url, axis=_axis(), track="resistivity", vlm=None)
    assert series.depth.shape[0] > 0


def test_extract_curve_unknown_blob_raises(store):
    with pytest.raises(ExtractionError):
        extract_curve("file-does-not-exist", axis=_axis(), track="resistivity", vlm=None, store=store)


def test_extract_values_returns_stable_deterministic_ids(store, tmp_path):
    from PIL import Image

    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    marker = (200, 30, 30)
    arr[10:13, 10:13] = marker
    arr[60:63, 70:73] = marker
    path = tmp_path / "markers.png"
    Image.fromarray(arr, mode="RGB").save(path)
    record = store.put(path.read_bytes(), filename="markers.png", content_type="image/png")

    axis = AxisCalib(
        depth_px=(0.0, 100.0),
        depth_range=(0.0, 100.0),
        value_px=(0.0, 100.0),
        value_range=(0.0, 100.0),
    )

    first = extract_values(record.id, axis=axis, track="tops", vlm=None, store=store)
    second = extract_values(record.id, axis=axis, track="tops", vlm=None, store=store)

    assert len(first) == 2
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len({r["id"] for r in first}) == 2
    assert all(len(r["id"]) == 16 for r in first)
    assert first == sorted(first, key=lambda r: r["depth"])
