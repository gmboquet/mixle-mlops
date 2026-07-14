"""I4 -- contour/isoline extraction -> gridded surface.

A contour/isoline map (topographic, gravity, bathymetric, potential-field, ...) encodes a scalar field as
labeled lines rather than a raster of values. ``extract_contours`` turns such an image into a list of
:class:`Isoline` -- each one a labeled level plus its vertices reprojected into CRS coordinates via the
affine handed in from I1's map registration. Reading the pixel-space geometry *and* the printed level label
off the raster is a vision task; this module deliberately does not do that work itself. It delegates to a
named, registered "VLM backend" (:func:`register_vlm_backend`) so a real vision-model call can be swapped in
without touching the reprojection/gridding logic, and ships a deterministic default backend that reads
pre-digitized isolines (the shape a structured VLM response would already be parsed into) -- the same
fixture-friendly seam D5's ``media_ref_from_tile`` uses for tile provenance.

``contours_to_grid`` is the other half: it stacks every isoline's ``(x, y, level)`` samples into one
scattered point cloud and interpolates it onto a regular grid via ``mixle.analysis.kriging`` (ordinary
kriging) -- the sole interpolator anywhere in this pipeline, reached only through its public function. This
module never imports ``mixle_pde``: the reconstructed grid is a plain array; a caller that wants an IC-2
field-posterior artifact wraps it on the physics side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .store import BlobStore, get_blob_store

#: A registered extraction backend: ``fn(image_bytes, image_ref) -> [{"level": float, "pixels": [[px, py], ...]},
#: ...]``, one entry per traced isoline, in pixel space.
VlmContourBackend = Callable[[bytes, str], list[dict[str, Any]]]

_BACKENDS: dict[str, VlmContourBackend] = {}


@dataclass
class Isoline:
    """One traced contour line: its labeled ``level`` and ``(n, 2)`` vertices, already reprojected to CRS
    coordinates by ``pixel_to_crs``."""

    level: float
    xy: np.ndarray


def register_vlm_backend(name: str, fn: VlmContourBackend) -> None:
    """Register an extraction backend under ``name`` for :func:`extract_contours`'s ``vlm=`` argument.

    A production deployment registers a call into a vision-capable model here (prompted to trace each
    isoline and read its printed level label) and passes that name as ``vlm``. Backends are looked up by
    name rather than passed as callables directly so ``extract_contours`` stays a plain, serializable-args
    function -- the same shape as every other tool surface in this codebase.
    """
    _BACKENDS[name] = fn


def _stub_backend(data: bytes, image_ref: str) -> list[dict[str, Any]]:
    """Default backend: treat ``data`` as already-digitized isolines (JSON ``{"isolines": [...]}`` or a bare
    list), the shape a structured VLM response is parsed into upstream. This is what the DoD fixture
    (``contour_stub.json``) exercises, and it is a legitimate path in its own right: any caller that already
    has pre-traced isolines (e.g. re-processing a cached VLM response) skips the model call entirely.
    """
    payload = json.loads(data.decode("utf-8"))
    isolines = payload["isolines"] if isinstance(payload, dict) else payload
    return list(isolines)


register_vlm_backend("stub", _stub_backend)


def _apply_affine(pixels: np.ndarray, pixel_to_crs: tuple[float, ...]) -> np.ndarray:
    """Six-term affine ``(a, b, c, d, e, f)``: ``x = a*px + b*py + c``, ``y = d*px + e*py + f`` -- the same
    convention as ``multimodal.content.GeoRef.pixel_to_crs`` (D5) and IC-13's ``pixel_to_crs``."""
    a, b, c, d, e, f = pixel_to_crs
    px = pixels[:, 0]
    py = pixels[:, 1]
    x = a * px + b * py + c
    y = d * px + e * py + f
    return np.stack([x, y], axis=1)


def extract_contours(
    image_ref: str,
    *,
    pixel_to_crs: tuple[float, ...],
    vlm: str | None = None,
    store: BlobStore | None = None,
) -> list[Isoline]:
    """Extract isolines from a contour-map image and reproject them into CRS coordinates.

    ``image_ref`` is a blob id resolved through ``store`` (the process blob store by default). ``vlm``
    selects the registered extraction backend (see :func:`register_vlm_backend`); ``None`` uses the
    ``"stub"`` backend. Vertices come back in pixel space from the backend and are reprojected here with
    ``pixel_to_crs`` (I1's affine) before being wrapped in :class:`Isoline`.
    """
    backend_name = vlm or "stub"
    backend = _BACKENDS.get(backend_name)
    if backend is None:
        raise ValueError(
            f"no VLM backend registered under {backend_name!r}; call register_vlm_backend first "
            f"(known backends: {sorted(_BACKENDS)})"
        )

    store = store or get_blob_store()
    _, data = store.get(image_ref)

    isolines: list[Isoline] = []
    for entry in backend(data, image_ref):
        pixels = np.atleast_2d(np.asarray(entry["pixels"], dtype=float))
        isolines.append(Isoline(level=float(entry["level"]), xy=_apply_affine(pixels, pixel_to_crs)))
    return isolines


def contours_to_grid(
    isolines: list[Isoline],
    *,
    grid_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    method: str = "kriging",
) -> np.ndarray:
    """Interpolate isoline samples onto a regular grid.

    Stacks every isoline's reprojected ``(x, y)`` vertices with its labeled ``level`` into one scattered
    ``(x, y, z)`` point cloud and interpolates it via ``mixle.analysis.kriging`` (ordinary kriging) --
    called across the repo boundary through its public function, never by importing ``mixle_pde``.

    Args:
        isolines: as returned by :func:`extract_contours`.
        grid_shape: ``(n_rows, n_cols)`` of the output grid, i.e. ``(ny, nx)``.
        bounds: ``(xmin, ymin, xmax, ymax)`` the grid spans, inclusive.
        method: interpolation method; only ``"kriging"`` is implemented (the only interpolator this
            pipeline owns -- no new one is added here).

    Returns:
        A ``(ny, nx)`` array, the reconstructed surface.
    """
    if method != "kriging":
        raise ValueError(f"unsupported interpolation method {method!r}; only 'kriging' is implemented")
    if not isolines:
        raise ValueError("contours_to_grid requires at least one isoline")

    from mixle.analysis.kriging import fit_variogram, ordinary_kriging

    xs = np.concatenate([iso.xy[:, 0] for iso in isolines])
    ys = np.concatenate([iso.xy[:, 1] for iso in isolines])
    zs = np.concatenate([np.full(iso.xy.shape[0], iso.level) for iso in isolines])
    coords = np.stack([xs, ys], axis=1)

    ny, nx = grid_shape
    xmin, ymin, xmax, ymax = bounds
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    query = np.stack([gx.ravel(), gy.ravel()], axis=1)

    variogram = fit_variogram(coords, zs)
    result = ordinary_kriging(coords, zs, variogram, query)
    return result["prediction"].reshape(ny, nx)


__all__ = [
    "Isoline",
    "VlmContourBackend",
    "register_vlm_backend",
    "extract_contours",
    "contours_to_grid",
]
