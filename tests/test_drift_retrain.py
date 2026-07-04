"""I4: automated drift -> re-solve -> promotion. Verifies the cron entrypoint's full loop."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

mixle_prod = pytest.importorskip("mixle.inference.production")

from mixle.inference.production import Registry, fit_with_provenance  # noqa: E402
from mixle.stats import GaussianDistribution  # noqa: E402

import mixle_mlops.drift_retrain as dr  # noqa: E402


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A seeded production registry + reference, with the module pointed at the tmp root."""
    root = str(tmp_path / "models")
    reference = np.random.RandomState(0).normal(0.0, 1.0, 800).tolist()
    model, _header = fit_with_provenance(reference, GaussianDistribution(0.0, 1.0).estimator(), max_its=30)
    reg = Registry(root)
    version = reg.register(model, "model")
    reg.promote("model", version, alias="production")
    os.makedirs(os.path.join(root, "model"), exist_ok=True)
    with open(os.path.join(root, "model", "reference.json"), "w") as fh:
        json.dump(reference, fh)
    monkeypatch.setattr(dr, "ROOT", root)
    monkeypatch.setattr(dr, "NAME", "model")
    return reg, version, tmp_path


def _batch_file(tmp_path, data, monkeypatch):
    p = tmp_path / "batch.json"
    p.write_text(json.dumps(data))
    monkeypatch.setenv("MIXLE_RECENT_BATCH_PATH", str(p))


def _production_version(root: str) -> str:
    return open(os.path.join(root, "model", "production.alias")).read().strip()


def test_drift_triggers_retrain_and_promotion(registry, monkeypatch, capsys):
    reg, v0, tmp_path = registry
    assert _production_version(dr.ROOT) == v0
    # a clearly shifted batch: mean 6 vs the reference's 0
    _batch_file(tmp_path, np.random.RandomState(1).normal(6.0, 1.0, 800).tolist(), monkeypatch)
    dr.main()
    out = capsys.readouterr().out
    assert "drift detected" in out
    assert _production_version(dr.ROOT) != v0  # a NEW version was promoted
    model, _header = reg.current("model", "production")
    # the promoted model has absorbed the new regime (its mean moved off 0)
    assert abs(float(model.mu)) > 1.0
    # the reference was rolled forward so the next run compares against the new regime
    ref = json.loads(open(os.path.join(dr.ROOT, "model", "reference.json")).read())
    assert len(ref) == 1600


def test_no_drift_keeps_the_production_model(registry, monkeypatch, capsys):
    reg, v0, tmp_path = registry
    # an in-distribution batch: same regime as the reference
    _batch_file(tmp_path, np.random.RandomState(2).normal(0.0, 1.0, 800).tolist(), monkeypatch)
    dr.main()
    out = capsys.readouterr().out
    assert "no drift" in out
    assert _production_version(dr.ROOT) == v0  # untouched: no silent re-fit without a drift receipt
