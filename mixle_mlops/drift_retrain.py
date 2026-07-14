"""Drift-triggered retrain-and-swap (run by the Kubernetes CronJob).

Loads the current production model + its drift reference, evaluates drift on a recent production batch, and
if drift is detected retrains a fresh model (with new provenance), registers it, and promotes it to
``production``. The serving Deployment then picks the new model up on its next ``POST /reload`` or a rolling
restart (``kubectl rollout restart deployment/mixle-model``).

``_recent_batch()`` is a stub: wire it to your real production-data store (warehouse, log sink, the
serving activity log, ...). The estimator here is a Gaussian to match seed_registry.py -- swap in yours.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from mixle.inference.conformal import split_conformal
from mixle.inference.production import Registry, detect_drift, fit_with_provenance
from mixle.stats import GaussianDistribution

ROOT = os.environ.get("MIXLE_REGISTRY_ROOT", "./models")
NAME = os.environ.get("MIXLE_MODEL_NAME", "model")


def _recent_batch() -> list:
    """STUB: return the recent production records to test for drift. Replace with a real data pull."""
    path = os.environ.get("MIXLE_RECENT_BATCH_PATH")
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    # demo fallback: a shifted sample so the example actually triggers a retrain
    return np.random.RandomState(1).normal(6.0, 2.0, 1000).tolist()


def main() -> None:
    registry = Registry(ROOT)
    model, _header = registry.current(NAME, "production")
    ref_path = os.path.join(ROOT, NAME, "reference.json")
    with open(ref_path) as fh:
        reference = json.load(fh)

    current = _recent_batch()
    report = detect_drift(model, reference, current)
    print(report)

    if not report.drift:
        print("no drift -- keeping the current production model")
        return

    train = reference + current  # retrain on reference + the new regime
    new_model, header = fit_with_provenance(train, GaussianDistribution(0.0, 1.0).estimator(), max_its=50)
    version = registry.register(new_model, NAME)
    registry.promote(NAME, version, alias="production")
    with open(ref_path, "w") as fh:
        json.dump(train, fh)
    print(f"drift detected -> retrained, registered {version}, promoted to production")
    print(header)


def detect_stream_drift(stream: str, reference: Any, current: Any, *, alpha: float = 0.1) -> dict[str, Any]:
    """H9 addition: a monitoring-series drift alarm for grade-forecast / demand / throughput streams.

    G7 (`mixle_mlops/monitoring.py` -- `exceedance_alarm` / `monitor_and_maybe_retrain`, built on this
    same core primitive) had not landed on this branch as of this PR (see the H9 PR body's Notes for the
    full explanation -- both tasks share Wave 3 and G7 was still in flight). Rather than block H9's
    `on_drift` on it, this reproduces G7's declared one-sided split-conformal alarm shape directly:
    calibrate an upper bound from ``reference`` (predicting its own mean) and flag drift when
    ``current``'s empirical rate of exceeding that bound is itself larger than the nominal false-alarm
    budget ``alpha`` the bound was built to hold -- the stream is behaving unlike a draw from the
    reference regime, at a strength past what ``alpha`` alone would produce by chance. This is additive
    (a new function; ``main``/``_recent_batch`` above are untouched) and is a drop-in call for G7's
    richer monitoring surface once that lands -- no change to `on_drift`'s call site expected.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    baseline = float(ref.mean()) if ref.size else 0.0
    _, upper = split_conformal(
        np.full(ref.shape, baseline), ref, np.full(cur.shape, baseline), alpha=alpha, side="upper"
    )
    bound = float(upper[0]) if upper.size else float("inf")
    exceed_rate = float(np.mean(cur > bound)) if cur.size else 0.0
    return {
        "stream": stream,
        "drift": bool(exceed_rate > alpha),
        "exceedance_rate": exceed_rate,
        "bound": bound,
        "alpha": alpha,
        "baseline": baseline,
    }


if __name__ == "__main__":
    main()
