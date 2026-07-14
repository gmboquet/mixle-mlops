"""I6 -- map registration / control-point georeferencing (workstream I, wave 2).

A scanned/photographed map only becomes useful to the physics layer once its pixel grid is tied to a
real coordinate reference system. :func:`georeference` fits that tie: given a handful of known
``(pixel_x, pixel_y, crs_x, crs_y)`` control points -- typically read off the map's corner graticule --
it solves a least-squares affine (>= 3 points) or projective (>= 4 points) transform, reports the
per-point and RMS residual in the *units of ``crs``* (assumed metres, e.g. a projected UTM CRS -- see
module non-goals), and refuses to hand back a transform whose RMS exceeds tolerance. A bad georef must
fail loudly, not silently pass a warped map downstream.

This refined affine/projective fit supersedes I1's coarse digitization-time affine everywhere
downstream: I1 tiles+digitizes fast with an approximate transform, I6 is the module that is allowed to
take its time getting the registration right.

Cross-repo boundary: CRS/datum math is entirely mixle-pde's B1 ``geospatial/crs.py`` (IC-4's ``crs``
string convention -- an EPSG string such as ``"EPSG:32611"``); this module never reimplements a
projection. Following this repo's existing lazy-import convention for `mixle_pde` (see
``gateway/routes/interpret.py``), the one place this module reaches across that boundary --
enriching provenance with a geographic (EPSG:4326) bounding box -- imports it lazily and degrades to
omitting that provenance key if `mixle_pde` isn't installed, rather than hard-failing the whole
registration over a nice-to-have.

Non-goals (owned elsewhere): no CRS/datum implementation (B1 owns `crs.py`); no reprojection of
downstream layers -- callers apply the returned `pixel_to_crs` themselves.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .store import BlobStore, get_blob_store

#: RMS residual, in CRS units (assumed metres), above which a fit is rejected outright (see
#: `georeference`'s Algorithm step 4: "a bad georef must fail loudly, not silently").
DEFAULT_RESIDUAL_TOLERANCE_M = 1.0

_MIN_AFFINE_POINTS = 3
_MIN_PROJECTIVE_POINTS = 4

ControlPoint = tuple[float, float, float, float]


class GeoreferenceError(ValueError):
    """A control-point/graticule fit's RMS residual exceeded tolerance, or too few points were given
    to fit anything. Raised instead of returning a low-confidence `GeoRef` -- callers must not receive
    a transform they didn't ask to have silently downgraded."""


@dataclass
class GeoRef:
    """The outcome of registering a map's pixel space to `crs`.

    ``pixel_to_crs`` is a 6-tuple ``(a, b, c, d, e, f)`` for an affine fit --
    ``x = a*px + b*py + c``, ``y = d*px + e*py + f`` -- or an 8-tuple ``(a, b, c, d, e, f, g, h)`` for
    a projective fit, which adds the homogeneous denominator ``g*px + h*py + 1``. The tuple's own
    length tells the two apart; `provenance["method"]` (``"affine"``/``"projective"``) names it
    explicitly too. ``residual_m`` is the RMS of all points' reprojection error in CRS units.
    """

    pixel_to_crs: tuple[float, ...]
    residual_m: float
    n_points: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def apply(self, pixel_xy: Any) -> np.ndarray:
        """Reproject an ``(n, 2)`` array of pixel coordinates into `crs` using the fitted transform.

        A convenience for callers that want to reproject their own downstream layers (this module's
        own non-goal) rather than re-deriving the affine/projective algebra themselves.
        """
        return _apply_transform(self.pixel_to_crs, np.atleast_2d(np.asarray(pixel_xy, dtype=float)))


def _apply_transform(params: tuple[float, ...], px_xy: np.ndarray) -> np.ndarray:
    px, py = px_xy[:, 0], px_xy[:, 1]
    if len(params) == 6:
        a, b, c, d, e, f = params
        x = a * px + b * py + c
        y = d * px + e * py + f
    elif len(params) == 8:
        a, b, c, d, e, f, g, h = params
        denom = g * px + h * py + 1.0
        x = (a * px + b * py + c) / denom
        y = (d * px + e * py + f) / denom
    else:  # pragma: no cover - defensive; only this module constructs `pixel_to_crs`
        raise GeoreferenceError(f"unrecognised pixel_to_crs arity {len(params)} (expected 6 or 8)")
    return np.column_stack([x, y])


def _fit_affine(px_xy: np.ndarray, crs_xy: np.ndarray) -> tuple[float, ...]:
    """6-parameter affine least squares. ``x`` and ``y`` don't share parameters, so this is just two
    independent linear regressions against the design matrix ``[px, py, 1]``."""
    design = np.column_stack([px_xy[:, 0], px_xy[:, 1], np.ones(px_xy.shape[0])])
    abc, *_ = np.linalg.lstsq(design, crs_xy[:, 0], rcond=None)
    de_f, *_ = np.linalg.lstsq(design, crs_xy[:, 1], rcond=None)
    return (*abc.tolist(), *de_f.tolist())


def _fit_projective(px_xy: np.ndarray, crs_xy: np.ndarray) -> tuple[float, ...]:
    """8-parameter projective (homography) fit via the direct linear transform: rearrange
    ``X = (a*px + b*py + c) / (g*px + h*py + 1)`` into a linear equation in the 8 unknowns and solve
    by least squares (exact at exactly 4 non-degenerate points; overdetermined beyond that)."""
    n = px_xy.shape[0]
    px, py = px_xy[:, 0], px_xy[:, 1]
    x_t, y_t = crs_xy[:, 0], crs_xy[:, 1]
    zeros, ones = np.zeros(n), np.ones(n)
    row_x = np.column_stack([px, py, ones, zeros, zeros, zeros, -px * x_t, -py * x_t])
    row_y = np.column_stack([zeros, zeros, zeros, px, py, ones, -px * y_t, -py * y_t])
    a_mat = np.empty((2 * n, 8))
    a_mat[0::2] = row_x
    a_mat[1::2] = row_y
    b_vec = np.empty(2 * n)
    b_vec[0::2] = x_t
    b_vec[1::2] = y_t
    params, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    return tuple(params.tolist())


def _residuals_m(params: tuple[float, ...], px_xy: np.ndarray, crs_xy: np.ndarray) -> np.ndarray:
    predicted = _apply_transform(params, px_xy)
    return np.sqrt(np.sum((predicted - crs_xy) ** 2, axis=1))


def _rms(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals**2)))


def _detect_graticule_ticks(image_bytes: bytes) -> list[ControlPoint]:
    """Best-effort CV/VLM read of a map's graticule tick labels -> additional
    ``(pixel_x, pixel_y, crs_x, crs_y)`` correspondences to merge with any given `control_points`.

    Unlike I1/I2/I3/I5, this task's frozen Public API takes no ``vlm=`` kwarg, so there is no wired
    model call here yet -- this is the pluggable seam a real tick-reading backend drops into. It
    returns no correspondences by default (a no-op, not a guess); tests/callers can monkeypatch this
    function to exercise the merge-with-control-points path.
    """
    return []


def _resolve_image_bytes(image_ref: str, store: BlobStore) -> bytes | None:
    """Best-effort resolve `image_ref` to raw bytes via `store`; ``None`` if it isn't a known blob id
    (`detect_grid`'s CV pass is then silently skipped -- see `_detect_graticule_ticks`)."""
    try:
        if store.has(image_ref):
            return store.get(image_ref)[1]
    except Exception:
        return None
    return None


def _geographic_extent(crs: str, crs_xy: np.ndarray) -> list[list[float]] | None:
    """Best-effort ``[[min_lon, min_lat], [max_lon, max_lat]]`` of the fitted control points in
    EPSG:4326, via mixle-pde's B1 `crs.py` (lazy import -- see module docstring). ``None`` if
    `mixle_pde` isn't installed or the reprojection fails for any reason; this is provenance
    enrichment, never load-bearing for the fit itself."""
    try:
        from mixle_pde.geospatial.crs import to_geographic
    except ImportError:
        return None
    xyz = np.column_stack([crs_xy, np.zeros(len(crs_xy))])
    try:
        geo = to_geographic(xyz, src_crs=crs)
    except Exception:
        return None
    lon, lat = geo[:, 0], geo[:, 1]
    return [[float(lon.min()), float(lat.min())], [float(lon.max()), float(lat.max())]]


def georeference(
    image_ref: str,
    *,
    crs: str,
    control_points: list[ControlPoint] | None = None,
    detect_grid: bool = True,
    store: BlobStore | None = None,
) -> GeoRef:
    """Register a map's pixel space to `crs` from `control_points` and/or a detected graticule.

    Algorithm: 1) start from any given `control_points`. 2) if `detect_grid`, resolve `image_ref` and
    read additional graticule tick correspondences, merging them in. 3) fit an affine transform
    (>= 3 total points) or a projective one (>= 4); 4) compute each point's residual and the RMS in
    `crs` units; 5) raise `GeoreferenceError` if the RMS exceeds `DEFAULT_RESIDUAL_TOLERANCE_M` -- a
    bad registration must fail loudly rather than hand back a low-confidence transform. `provenance`
    always carries ``{crs, n_points, residual_m, method}`` plus the per-point residuals and (best
    effort) a geographic bounding box.
    """
    points: list[ControlPoint] = list(control_points or [])

    image_content_hash: str | None = None
    if detect_grid:
        active_store = store or get_blob_store()
        image_bytes = _resolve_image_bytes(image_ref, active_store)
        if image_bytes is not None:
            image_content_hash = hashlib.sha256(image_bytes).hexdigest()
            points.extend(_detect_graticule_ticks(image_bytes))

    n_points = len(points)
    if n_points < _MIN_AFFINE_POINTS:
        raise GeoreferenceError(
            f"georeference needs >= {_MIN_AFFINE_POINTS} control points (affine) or "
            f">= {_MIN_PROJECTIVE_POINTS} (projective); got {n_points}"
        )

    arr = np.asarray(points, dtype=float)
    px_xy, crs_xy = arr[:, :2], arr[:, 2:]

    if n_points >= _MIN_PROJECTIVE_POINTS:
        method = "projective"
        params = _fit_projective(px_xy, crs_xy)
    else:
        method = "affine"
        params = _fit_affine(px_xy, crs_xy)

    residuals = _residuals_m(params, px_xy, crs_xy)
    rms = _rms(residuals)

    if rms > DEFAULT_RESIDUAL_TOLERANCE_M:
        raise GeoreferenceError(
            f"georeference RMS residual {rms:.3f} m exceeds tolerance "
            f"{DEFAULT_RESIDUAL_TOLERANCE_M} m over {n_points} points ({method} fit) -- "
            "rejecting a bad registration rather than returning it silently"
        )

    provenance: dict[str, Any] = {
        "crs": crs,
        "n_points": n_points,
        "residual_m": rms,
        "method": method,
        "tolerance_m": DEFAULT_RESIDUAL_TOLERANCE_M,
        "per_point_residual_m": residuals.tolist(),
    }
    if image_content_hash is not None:
        provenance["image_content_hash"] = image_content_hash
    extent = _geographic_extent(crs, crs_xy)
    if extent is not None:
        provenance["extent_geographic"] = extent

    return GeoRef(pixel_to_crs=params, residual_m=rms, n_points=n_points, provenance=provenance)
