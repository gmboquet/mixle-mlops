"""G7 -- exceedance probability + monitoring-drift detection (work-plan Wave 3, mlops-owned).

Two related monitoring primitives sit on top of core's inference-production layer and A5's decision
quantities, neither of which this module owns:

* :func:`exceedance_probability` -- for a caller already holding a real IC-1 ``Posterior`` over a
  concentration (or any monitored) field (e.g. a ``mixle_pde`` block-model or well posterior), this is a
  thin dispatch to IC-8's ``mixle_pde.decision_quantities.prob_exceed``: ``P(field > regulatory_limit)``
  as a distribution -- samples, a credible interval, and the ``prior_dominated`` honesty flag -- never a
  bare point. The posterior itself is A5's job; this module only calls into it.

* :func:`exceedance_alarm` / :func:`monitor_and_maybe_retrain` -- the lighter-weight path the drift-retrain
  CronJob (``drift_retrain.py``) actually runs on: it only ever sees flat numeric monitoring batches, no
  fitted field posterior. A one-sided split-conformal upper confidence bound on the process level,
  calibrated against a reference/calibration sample, raises the alarm the moment natural variability
  alone could carry the process over the limit -- an early-warning signal, not just a breach detector.
  This reuses core's ``mixle.inference.conformal.split_conformal`` rather than re-deriving a conformal
  quantile locally.

``monitor_and_maybe_retrain`` combines both signals (drift *or* exceedance alarm) into one retrain
trigger and reuses ``drift_retrain``'s existing register/promote swap path
(:func:`mixle_mlops.drift_retrain.retrain_and_promote`) instead of duplicating it -- G7 is a monitoring
and serving surface over A5/core's machinery, not a second training pipeline (see this task's
non-goals).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.inference.conformal import split_conformal
from mixle.inference.production import DriftReport, detect_drift

__all__ = [
    "exceedance_probability",
    "exceedance_alarm",
    "monitor_and_maybe_retrain",
    "alerts",
]

# In-process, queryable alert log: every call to `monitor_and_maybe_retrain` appends one record here
# (records are never mutated after append). This is the "wire the alarm/threshold into the serving
# endpoint so alerts are queryable" half of the algorithm (step 5) -- a serving route can import
# `alerts()` to answer "what has this monitor flagged recently" without this task touching any gateway
# route file (Files: this task only edits `drift_retrain.py` and this module).
_ALERTS: list[dict[str, Any]] = []


def alerts() -> list[dict[str, Any]]:
    """Every alert record :func:`monitor_and_maybe_retrain` has emitted this process's lifetime, oldest
    first."""
    return list(_ALERTS)


def exceedance_probability(posterior: Any, threshold: float, *, level: float = 0.9) -> Any:
    """``P(field > threshold)`` over the whole posterior domain, as an IC-8 decision quantity.

    Dispatches to ``mixle_pde.decision_quantities.prob_exceed`` (A5's IC-8 fill-in) with a region mask
    that selects every component of ``posterior`` -- the monitoring use case is a single scalar
    concentration (or a short vector across a handful of monitoring points), not a spatial block model
    with an actual sub-region to carve out of a larger field, so "the region" is the whole posterior.
    Returns A5's ``SampledDerivedQuantity``: samples (the per-draw exceeding fraction), a credible
    interval at ``level`` (attached as ``.ci``), and ``prior_dominated`` -- never a bare point.
    """
    try:
        from mixle_pde.decision_quantities import prob_exceed
    except ImportError as exc:  # pragma: no cover -- exercised only when mixle-pde truly isn't installed
        raise ImportError(
            "exceedance_probability needs mixle_pde's decision-quantity surface (IC-8, work-plan A5); "
            "install mixle_pde."
        ) from exc

    mean = np.atleast_1d(np.asarray(posterior.mean, dtype=float))
    region = np.ones_like(mean, dtype=bool)
    quantity = prob_exceed(posterior, region, threshold=float(threshold))
    quantity.ci = quantity.credible_interval(level)
    quantity.level = float(level)
    return quantity


def _conformal_upper_bound(current: np.ndarray, reference: np.ndarray, *, alpha: float) -> float:
    """The one-sided split-conformal ``(1 - alpha)`` upper bound on a fresh reading exchangeable with
    ``reference``, evaluated at ``current``'s own mean level.

    Calibrates ``split_conformal`` around a constant baseline (``reference``'s mean) so the residual
    quantile reflects only the reference regime's natural variability, then re-centers that margin on
    ``current``'s mean -- "given how much natural variability the reference period showed, how high
    could a fresh reading at today's average level plausibly go, at (1 - alpha) confidence." A constant
    baseline keeps the bound exact and model-free: the raw monitoring batches ``drift_retrain.py``
    consumes have no fitted forecaster to calibrate around.
    """
    baseline = float(np.mean(reference))
    cal_pred = np.full(reference.shape, baseline)
    test_pred = np.full((1,), float(np.mean(current)))
    _, upper = split_conformal(cal_pred, reference, test_pred, alpha=alpha, side="upper")
    return float(upper[0])


def exceedance_alarm(series: Any, limit: float, cal_series: Any, *, alpha: float = 0.1) -> bool:
    """Raise the exceedance alarm iff the one-sided split-conformal upper bound on ``series`` clears
    ``limit`` (algorithm step 2).

    ``cal_series`` calibrates the ``(1 - alpha)`` natural-variability margin used to project
    ``series``'s current level forward. Because that margin is the exact finite-sample split-conformal
    quantile, the alarm's false-alarm rate is bounded by ``alpha`` whenever ``cal_series`` and ``series``
    are exchangeable (the monitored process has not actually left the calibration regime) -- see
    ``tests/test_exceedance.py`` for the Monte-Carlo demonstration this is the work-plan G7 Definition of
    Done.
    """
    series_arr = np.atleast_1d(np.asarray(series, dtype=float))
    cal_arr = np.atleast_1d(np.asarray(cal_series, dtype=float))
    bound = _conformal_upper_bound(series_arr, cal_arr, alpha=alpha)
    return bool(bound > float(limit))


def monitor_and_maybe_retrain(
    model: Any,
    reference: Any,
    current: Any,
    limit: float,
    *,
    alpha: float = 0.1,
    registry_root: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Combine drift + exceedance into one retrain decision, and act on it (algorithm steps 1-4).

    Runs ``detect_drift`` (core) and the conformal :func:`exceedance_alarm` over the same
    ``reference``/``current`` monitoring batch. If either fires, retrains via
    ``mixle_mlops.drift_retrain.retrain_and_promote`` (the existing register + promote swap path,
    imported lazily here to avoid a module-load cycle with ``drift_retrain``, which imports this
    module) and records the new registry version. Always builds + returns a provenanced alert record
    (threshold, the empirical exceedance probability over ``current``, the conformal bound, and the
    drift verdict) and appends it to the process-local :func:`alerts` log (algorithm step 5).

    ``current``/``reference`` are treated as flat numeric monitoring batches (matching
    ``drift_retrain.py``'s existing convention), not IC-1 posteriors -- :func:`exceedance_probability` is
    the entry point for callers holding a real calibrated posterior instead.
    """
    reference_arr = np.asarray(reference, dtype=float)
    current_arr = np.asarray(current, dtype=float)

    report: DriftReport = detect_drift(model, reference, current)
    bound = _conformal_upper_bound(current_arr, reference_arr, alpha=alpha)
    alarm = bound > float(limit)
    probability = float(np.mean(current_arr > float(limit))) if current_arr.size else 0.0
    retrain_triggered = bool(report.drift or alarm)

    record: dict[str, Any] = {
        "threshold": float(limit),
        "probability": probability,
        "conformal_bound": bound,
        "alarm": bool(alarm),
        "drift": bool(report.drift),
        "retrain_triggered": retrain_triggered,
        "drift_report": report,
    }

    new_version = None
    if retrain_triggered:
        from mixle_mlops.drift_retrain import retrain_and_promote

        new_version = retrain_and_promote(list(reference), list(current), registry_root=registry_root, name=name)
    record["new_version"] = new_version

    _ALERTS.append(record)
    return record
