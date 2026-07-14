"""N5 -- ecological monitoring & decline early-warning (work-plan Wave 3, mlops-owned).

Three primitives on top of an occupancy-or-biodiversity-index time series and (optionally) an N1
habitat-suitability baseline, mirroring G7's exceedance-monitoring shape (``mixle_mlops/monitoring.py``)
for the opposite tail -- decline, not exceedance:

* :func:`occupancy_trend` -- a Theil-Sen slope + confidence interval over an occupancy/biodiversity-index
  series; ``declining`` is set when the whole interval sits below zero (a confidently negative trend, not
  just a negative point estimate).

* :func:`decline_alarm` -- G7's one-sided split-conformal alarm, mirrored for the lower tail: build a
  one-sided LOWER bound via core's ``mixle.inference.conformal.split_conformal`` and raise the alarm the
  moment that bound falls below a compliance ``floor``. Because the bound is the exact finite-sample
  split-conformal quantile, the false-alarm rate is bounded by ``alpha`` whenever ``cal_series`` and
  ``index_series`` are exchangeable (the population hasn't actually left the calibration regime) -- see
  ``tests/test_n5_decline.py`` for the Monte-Carlo demonstration this is the Definition of Done.

* :func:`monitor_biodiversity` -- combines a drift verdict against the N1 baseline (``HabitatModel``,
  IC-1 ``Posterior``) with :func:`decline_alarm` into one retrain/rescreen trigger, emits a provenanced,
  queryable alert record (the "same endpoint pattern as G7" -- a private log + a public getter), and
  reuses ``drift_retrain.retrain_and_promote`` (the swap path G7's own ``monitor_and_maybe_retrain``
  delegates to) rather than re-deriving a training loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.inference.conformal import split_conformal
from mixle.inference.production import DriftReport, detect_drift
from mixle.stats import GaussianDistribution

__all__ = [
    "occupancy_trend",
    "decline_alarm",
    "monitor_biodiversity",
    "biodiversity_alerts",
]

# Nominal false-alarm rate for the drift/retrain decision baked into `monitor_biodiversity` -- its frozen
# Public API signature has no `alpha` slot, so this fixes the same default `decline_alarm` and G7 use.
_ALPHA = 0.1
_MIN_REFERENCE_DRAWS = 64

# In-process, queryable alert log -- one record per `monitor_biodiversity` call, never mutated after
# append. Mirrors G7's `monitoring.alerts()` shape (a private list + a public getter) rather than sharing
# G7's actual list, since this task's Files/Parallel-safety only touches this one new module.
_ALERTS: list[dict[str, Any]] = []


def biodiversity_alerts() -> list[dict[str, Any]]:
    """Every alert record :func:`monitor_biodiversity` has emitted this process's lifetime, oldest
    first -- the serving surface (same endpoint pattern as G7's ``monitoring.alerts``) queries this."""
    return list(_ALERTS)


def occupancy_trend(series: np.ndarray, *, times: np.ndarray | None = None) -> dict:
    """Theil-Sen slope + 95% CI of an occupancy-or-biodiversity-index time series.

    ``declining`` is ``True`` only when the *upper* end of the slope's confidence interval is itself
    below zero -- a confidently negative trend, not merely a negative point estimate (a single noisy dip
    should not trip an early-warning flag).
    """
    y = np.atleast_1d(np.asarray(series, dtype=float))
    x = np.arange(y.size, dtype=float) if times is None else np.atleast_1d(np.asarray(times, dtype=float))
    if y.size < 2 or np.allclose(x, x[0]):
        # Degenerate input (a single reading, or no time spread): no evidence of a trend either way.
        return {"slope": 0.0, "intercept": float(y[0]) if y.size else 0.0, "ci": (0.0, 0.0), "declining": False}

    from scipy.stats import theilslopes

    slope, intercept, lo, hi = theilslopes(y, x)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "ci": (float(lo), float(hi)),
        "declining": bool(hi < 0.0),
    }


def _conformal_lower_bound(current: np.ndarray, reference: np.ndarray, *, alpha: float) -> float:
    """The one-sided split-conformal ``(1 - alpha)`` LOWER bound on a fresh reading exchangeable with
    ``reference``, evaluated at ``current``'s own mean level -- G7's ``_conformal_upper_bound``, mirrored
    for the lower tail (``side="lower"`` instead of ``"upper"``; alarm fires below, not above)."""
    baseline = float(np.mean(reference))
    cal_pred = np.full(reference.shape, baseline)
    test_pred = np.full((1,), float(np.mean(current)))
    lower, _ = split_conformal(cal_pred, reference, test_pred, alpha=alpha, side="lower")
    return float(lower[0])


def decline_alarm(index_series: np.ndarray, cal_series: np.ndarray, *, floor: float, alpha: float = 0.1) -> bool:
    """Raise the decline alarm iff the one-sided split-conformal LOWER bound on ``index_series`` falls
    below ``floor`` (G7's exceedance-alarm logic, mirrored for decline).

    ``cal_series`` calibrates the ``(1 - alpha)`` natural-variability margin used to project
    ``index_series``'s current level backward/downward. Because that margin is the exact finite-sample
    split-conformal quantile, the alarm's false-alarm rate is bounded by ``alpha`` whenever ``cal_series``
    and ``index_series`` are exchangeable (the population has not actually left the calibration
    regime) -- see ``tests/test_n5_decline.py`` for the Monte-Carlo demonstration this is the Definition
    of Done.
    """
    series_arr = np.atleast_1d(np.asarray(index_series, dtype=float))
    cal_arr = np.atleast_1d(np.asarray(cal_series, dtype=float))
    bound = _conformal_lower_bound(series_arr, cal_arr, alpha=alpha)
    return bool(bound < float(floor))


def _reference_index_from_baseline(baseline: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Flatten an IC-1 ``Posterior``/``HabitatModel`` baseline's field draws into a scalar occupancy-index
    reference sample directly comparable to ``current_index``: ``baseline.samples(n, rng)`` draws the
    ``(n, d)`` suitability/occupancy field per IC-1, and each component (grid cell / draw) stands in for
    one occupancy-index-like reading at the baseline's natural variability. Flattening (rather than
    averaging each draw down to a single number) keeps that per-reading variability intact -- averaging
    would shrink it by the usual central-limit factor and understate how much the reference regime
    naturally varies, biasing `detect_drift`'s distributional comparison against `current_index`. Only
    ``.samples`` is used here -- the one IC-1 method every baseline (a core distribution or a
    ``mixle_pde`` field posterior) is guaranteed to have."""
    draws = np.atleast_1d(np.asarray(baseline.samples(n, rng), dtype=float))
    return draws.reshape(-1)


def monitor_biodiversity(baseline: "HabitatModel", current_index: np.ndarray, *, floor: float) -> dict:  # noqa: F821
    """Combine an N1-baseline drift verdict with :func:`decline_alarm` into one retrain/rescreen trigger.

    ``baseline`` is duck-typed against IC-1's ``Posterior`` protocol (only ``.samples`` is called) so any
    conforming N1 ``HabitatModel`` works without this module importing it. ``detect_drift`` (core) wants a
    density-scoring ``model`` (``.log_density``/``.seq_log_density``), which HabitatModel's frozen IC-12
    surface (``samples``/``mean``/``cov``/``credible_interval``/``derived_quantity``) does not promise --
    rather than reach into either frozen contract, this moment-matches a ``mixle.stats.GaussianDistribution``
    to the baseline-derived reference sample to serve as that scoring model, a thin composition local to
    this function.

    Retrain reuses ``mixle_mlops.drift_retrain.retrain_and_promote`` -- the same swap path G7's
    ``monitor_and_maybe_retrain`` delegates to -- directly, rather than calling
    ``monitor_and_maybe_retrain`` itself: that function's own alarm is a fixed one-sided UPPER exceedance
    check, a different (wrong-direction) condition than this decline verdict, and calling it would let its
    internal alarm silently override the decline signal computed here.
    """
    rng = np.random.default_rng(0)
    current_arr = np.atleast_1d(np.asarray(current_index, dtype=float))
    n_reference = max(current_arr.size, _MIN_REFERENCE_DRAWS)
    reference = _reference_index_from_baseline(baseline, n_reference, rng)

    ref_mean = float(np.mean(reference))
    ref_std = float(np.std(reference))
    reference_model = GaussianDistribution(ref_mean, max(ref_std, 1e-6))

    report: DriftReport = detect_drift(reference_model, reference, current_arr)
    declining = decline_alarm(current_arr, reference, floor=floor, alpha=_ALPHA)
    retrain_triggered = bool(report.drift or declining)

    record: dict[str, Any] = {
        "floor": float(floor),
        "current_mean": float(np.mean(current_arr)) if current_arr.size else float("nan"),
        "declining": bool(declining),
        "drift": bool(report.drift),
        "retrain_triggered": retrain_triggered,
        "drift_report": report,
    }

    new_version = None
    if retrain_triggered:
        from mixle_mlops.drift_retrain import retrain_and_promote

        new_version = retrain_and_promote(reference.tolist(), current_arr.tolist())
    record["new_version"] = new_version

    _ALERTS.append(record)
    return record
