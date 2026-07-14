"""I4 -- contour/isoline extraction -> gridded surface.

``test_grid_reconstructs_surface`` is the DoD: feed ``contour_stub.json`` (isolines traced off
``surface_truth.npy`` at a dozen levels -- the shape a VLM's structured read of a printed contour map would
already be parsed into) through ``extract_contours`` + ``contours_to_grid`` and check the kriged
reconstruction against the known-truth grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mixle_mlops.multimodal.contours import Isoline, contours_to_grid, extract_contours, register_vlm_backend
from mixle_mlops.multimodal.store import LocalBlobStore

FIXTURES = Path(__file__).parent / "fixtures"
IDENTITY_AFFINE = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # x = px, y = py -- CRS coincides with pixel/grid index space


def _put_stub(store: LocalBlobStore) -> str:
    data = (FIXTURES / "contour_stub.json").read_bytes()
    record = store.put(data, filename="contour_stub.json", content_type="application/json")
    return record.id


def test_grid_reconstructs_surface(tmp_path):
    store = LocalBlobStore(root=tmp_path)
    image_ref = _put_stub(store)

    isolines = extract_contours(image_ref, pixel_to_crs=IDENTITY_AFFINE, store=store)
    assert isolines and all(isinstance(iso, Isoline) for iso in isolines)
    assert all(iso.xy.shape[1] == 2 for iso in isolines)

    truth = np.load(FIXTURES / "surface_truth.npy")
    ny, nx = truth.shape
    grid = contours_to_grid(
        isolines,
        grid_shape=(ny, nx),
        bounds=(0.0, 0.0, float(nx - 1), float(ny - 1)),
    )

    assert grid.shape == truth.shape
    rmse = np.sqrt(np.mean((grid - truth) ** 2))
    nrmse = rmse / (truth.max() - truth.min())
    assert nrmse < 0.1, f"normalized RMSE {nrmse:.4f} >= 0.1"


def test_extract_contours_applies_affine(tmp_path):
    store = LocalBlobStore(root=tmp_path)
    payload = {"isolines": [{"level": 1.5, "pixels": [[0, 0], [1, 0], [1, 1]]}]}
    record = store.put(json.dumps(payload).encode("utf-8"), filename="s.json", content_type="application/json")

    isolines = extract_contours(record.id, pixel_to_crs=(2.0, 0.0, 10.0, 0.0, -1.0, 5.0), store=store)
    assert len(isolines) == 1
    iso = isolines[0]
    assert iso.level == 1.5
    np.testing.assert_allclose(iso.xy, [[10.0, 5.0], [12.0, 5.0], [12.0, 4.0]])


def test_unknown_backend_raises(tmp_path):
    store = LocalBlobStore(root=tmp_path)
    record = store.put(b'{"isolines": []}', filename="s.json", content_type="application/json")
    with pytest.raises(ValueError, match="no VLM backend registered"):
        extract_contours(record.id, pixel_to_crs=IDENTITY_AFFINE, vlm="not-a-real-backend", store=store)


def test_contours_to_grid_rejects_unknown_method():
    iso = Isoline(level=1.0, xy=np.array([[0.0, 0.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="unsupported interpolation method"):
        contours_to_grid([iso], grid_shape=(2, 2), bounds=(0.0, 0.0, 1.0, 1.0), method="idw")


def test_register_vlm_backend_is_dispatchable(tmp_path):
    store = LocalBlobStore(root=tmp_path)
    record = store.put(b"raw-bytes-from-a-real-image", filename="map.png", content_type="image/png")

    def fake_vlm(data: bytes, image_ref: str):
        assert data == b"raw-bytes-from-a-real-image"
        return [{"level": 3.0, "pixels": [[0, 0], [2, 0]]}]

    register_vlm_backend("fake-vision-model", fake_vlm)
    isolines = extract_contours(record.id, pixel_to_crs=IDENTITY_AFFINE, vlm="fake-vision-model", store=store)
    assert len(isolines) == 1 and isolines[0].level == 3.0
    np.testing.assert_allclose(isolines[0].xy, [[0.0, 0.0], [2.0, 0.0]])
