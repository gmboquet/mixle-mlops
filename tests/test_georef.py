"""I6 -- map registration / control-point georeferencing.

`georeference` fits a pixel->CRS transform from control points read off a map's corner graticule
(`tests/fixtures/graticule_stub.json`), and must reject a registration whose residual is too large to
trust rather than silently returning a warped fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mixle_mlops.multimodal.georef import (
    DEFAULT_RESIDUAL_TOLERANCE_M,
    GeoreferenceError,
    GeoRef,
    georeference,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "graticule_stub.json").read_text())


def _corners() -> list[tuple[float, float, float, float]]:
    return [tuple(p) for p in FIXTURE["corners"]]


def test_corners_georegister():
    """Four known graticule corners register with sub-metre residual; adding the fixture's corrupted
    5th point pushes the RMS over tolerance and the call raises instead of returning a bad fit."""
    control_points = _corners()

    result = georeference(
        FIXTURE["image_ref"],
        crs=FIXTURE["crs"],
        control_points=control_points,
        detect_grid=False,
    )

    assert isinstance(result, GeoRef)
    assert result.n_points == 4
    assert result.residual_m < 1.0
    assert result.provenance["crs"] == FIXTURE["crs"]
    assert result.provenance["n_points"] == 4
    assert result.provenance["residual_m"] == pytest.approx(result.residual_m)
    assert result.provenance["method"] in {"affine", "projective"}

    corrupted_points = control_points + [tuple(FIXTURE["corrupted_point"])]
    with pytest.raises(GeoreferenceError):
        georeference(
            FIXTURE["image_ref"],
            crs=FIXTURE["crs"],
            control_points=corrupted_points,
            detect_grid=False,
        )


def test_three_points_fit_affine():
    """Exactly 3 points is the minimum viable fit and must use the affine (not projective) method."""
    control_points = _corners()[:3]
    result = georeference("unused-ref", crs=FIXTURE["crs"], control_points=control_points, detect_grid=False)
    assert result.provenance["method"] == "affine"
    assert len(result.pixel_to_crs) == 6
    assert result.residual_m < DEFAULT_RESIDUAL_TOLERANCE_M


def test_four_points_fit_projective():
    control_points = _corners()
    result = georeference("unused-ref", crs=FIXTURE["crs"], control_points=control_points, detect_grid=False)
    assert result.provenance["method"] == "projective"
    assert len(result.pixel_to_crs) == 8


def test_too_few_points_raises_before_fitting():
    with pytest.raises(GeoreferenceError):
        georeference(
            "unused-ref",
            crs=FIXTURE["crs"],
            control_points=_corners()[:2],
            detect_grid=False,
        )


def test_georef_apply_reprojects_pixel_points():
    control_points = _corners()
    result = georeference("unused-ref", crs=FIXTURE["crs"], control_points=control_points, detect_grid=False)
    pixel_xy = np.array([[p[0], p[1]] for p in control_points])
    crs_xy = np.array([[p[2], p[3]] for p in control_points])
    reprojected = result.apply(pixel_xy)
    assert np.allclose(reprojected, crs_xy, atol=1.0)


def test_detect_grid_merges_detected_correspondences(monkeypatch):
    """`detect_grid=True` merges CV/VLM-detected graticule ticks with any given control points --
    exercised here by monkeypatching the (currently no-op) tick detector, since this task's frozen
    Public API takes no `vlm=` kwarg to wire a real backend in."""
    import mixle_mlops.multimodal.georef as georef_mod
    from mixle_mlops.multimodal.store import LocalBlobStore

    corners = _corners()
    detected = [corners[-1]]

    monkeypatch.setattr(georef_mod, "_detect_graticule_ticks", lambda image_bytes: detected)

    store = LocalBlobStore(root=Path(__file__).parent / "fixtures" / ".blob_tmp_i6")
    record = store.put(b"stub-map-bytes", filename="map.png", content_type="image/png")

    result = georeference(
        record.id,
        crs=FIXTURE["crs"],
        control_points=corners[:3],
        detect_grid=True,
        store=store,
    )
    assert result.n_points == 4
    assert result.provenance["image_content_hash"] == __import__("hashlib").sha256(b"stub-map-bytes").hexdigest()

    # clean up the throwaway local store directory
    import shutil

    shutil.rmtree(store.root, ignore_errors=True)


def test_detect_grid_skips_silently_when_image_unresolvable():
    """`detect_grid=True` with an unresolvable `image_ref` degrades to the given control points only
    (no crash, no fabricated correspondences)."""
    from mixle_mlops.multimodal.store import LocalBlobStore

    control_points = _corners()
    store = LocalBlobStore(root=Path(__file__).parent / "fixtures" / ".blob_tmp_i6_missing")
    result = georeference(
        "no-such-blob-id",
        crs=FIXTURE["crs"],
        control_points=control_points,
        detect_grid=True,
        store=store,
    )
    assert result.n_points == 4
    assert "image_content_hash" not in result.provenance

    import shutil

    shutil.rmtree(store.root, ignore_errors=True)
