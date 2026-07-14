"""Drift-triggered retrain-and-swap (run by the Kubernetes CronJob).

Loads the current production model + its drift reference, evaluates drift on a recent production batch, and
if drift is detected retrains a fresh model (with new provenance), registers it, and promotes it to
``production``. The serving Deployment then picks the new model up on its next ``POST /reload`` or a rolling
restart (``kubectl rollout restart deployment/mixle-model``).

G7 adds a second, complementary trigger alongside drift: a monitoring-series exceedance alarm (a
one-sided split-conformal upper bound against a regulatory limit, ``mixle_mlops.monitoring``). When a
limit is configured (``MIXLE_REGULATORY_LIMIT``), :func:`main` defers the whole decision -- drift *or*
exceedance -- to :func:`mixle_mlops.monitoring.monitor_and_maybe_retrain`; without one it falls back to
the original drift-only check, so existing deployments that haven't configured a limit are unaffected.

``_recent_batch()`` is a stub: wire it to your real production-data store (warehouse, log sink, the
serving activity log, ...). The estimator here is a Gaussian to match seed_registry.py -- swap in yours.
"""

from __future__ import annotations

import json
import os

import numpy as np

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


def _regulatory_limit() -> float | None:
    """STUB: the monitored variable's regulatory exceedance limit. Real permit/obligation ingestion is
    G8's job; this only reads ``MIXLE_REGULATORY_LIMIT`` if an operator has already set one. ``None``
    disables the exceedance-alarm path entirely (drift-only, this job's pre-G7 behavior)."""
    raw = os.environ.get("MIXLE_REGULATORY_LIMIT")
    return float(raw) if raw is not None else None


def retrain_and_promote(
    reference: list,
    current: list,
    *,
    registry_root: str | None = None,
    name: str | None = None,
) -> str:
    """Fit a fresh model on ``reference + current`` and swap it into ``production``; return the new
    version id.

    This is the register/promote swap path :func:`main` has always used, pulled out so
    ``mixle_mlops.monitoring.monitor_and_maybe_retrain`` can trigger the identical swap from an
    exceedance alarm (not just a drift verdict) without duplicating it (work-plan G7, algorithm step 3).
    """
    root = registry_root or ROOT
    model_name = name or NAME
    registry = Registry(root)
    train = list(reference) + list(current)
    new_model, header = fit_with_provenance(train, GaussianDistribution(0.0, 1.0).estimator(), max_its=50)
    version = registry.register(new_model, model_name)
    registry.promote(model_name, version, alias="production")
    ref_path = os.path.join(root, model_name, "reference.json")
    with open(ref_path, "w") as fh:
        json.dump(train, fh)
    print(header)
    return version


def main() -> None:
    registry = Registry(ROOT)
    model, _header = registry.current(NAME, "production")
    ref_path = os.path.join(ROOT, NAME, "reference.json")
    with open(ref_path) as fh:
        reference = json.load(fh)

    current = _recent_batch()
    limit = _regulatory_limit()

    if limit is None:
        report = detect_drift(model, reference, current)
        print(report)
        if not report.drift:
            print("no drift -- keeping the current production model")
            return
        version = retrain_and_promote(reference, current)
        print(f"drift detected -> retrained, registered {version}, promoted to production")
        return

    from mixle_mlops.monitoring import monitor_and_maybe_retrain

    result = monitor_and_maybe_retrain(model, reference, current, limit)
    print(result["drift_report"])
    print(
        f"exceedance: probability={result['probability']:.4f} conformal_bound={result['conformal_bound']:.4f} "
        f"limit={limit:.4f} alarm={result['alarm']}"
    )
    if not result["retrain_triggered"]:
        print("no drift/exceedance -- keeping the current production model")
        return
    print(f"retrain triggered -> registered {result['new_version']}, promoted to production")


if __name__ == "__main__":
    main()
