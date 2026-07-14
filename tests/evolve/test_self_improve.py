"""M7 -- the self-improvement loop: a week of synthetic usage runs ``improve_once`` end to end, and an
injected M8 harness gates the promotion on top of the existing verify-gated worker."""

from dataclasses import dataclass, field

import mixle_mlops.storage.db as db
import numpy as np
import pytest
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from sqlmodel import Session

from mixle_mlops.config import get_settings
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.evolve.policy import EvolutionPolicy
from mixle_mlops.evolve.self_improve import HarnessResult, improve_once, record_usage
from mixle_mlops.models.mixle_model import MixleAdapter


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    from mixle_mlops.storage.db import get_engine, init_db

    init_db()
    with Session(get_engine()) as s:
        yield s
    get_settings.cache_clear()
    db._engine = None


def _seed_week_of_usage(session, model_id, *, truth_mean=5.0, n=80, canonical_ref="predict", verified=True):
    """A week of synthetic, verified usage: each observation is an IC-5 trace envelope whose ``outcome`` is
    the ground truth the model should have retrained toward."""
    rng = np.random.RandomState(0)
    for value in rng.normal(truth_mean, 1.0, size=n):
        trace = {
            "prompt": "predict the next observation",
            "steps": [{"tool": "predict", "args": {}, "result": float(value), "model": model_id, "verdict": None}],
            "outcome": float(value),
            "provenance": {"source": "synthetic-usage-week"},
        }
        record_usage(session, model_id, trace, verified=verified, canonical_ref=canonical_ref)


@dataclass
class _Verdict:
    passed: bool = True
    score: float = 1.0
    reasons: list = field(default_factory=list)
    kind: str = "calibration"


class _PassVerifier:
    def verify(self, claim, context):
        return _Verdict(passed=True, score=1.0)


class _TaskHarness:
    """A deterministic end-to-end task-success harness: it re-measures the *currently served* model each
    time (never a call counter), so calling ``evaluate`` before and after ``improve_once`` reflects whatever
    the promotion actually did."""

    def __init__(self, registry, model_id, *, target=5.0, regressed=False):
        self.registry = registry
        self.model_id = model_id
        self.target = target
        self.regressed = regressed

    def evaluate(self, model_id):
        model = self.registry.get(model_id)._model
        success = -abs(float(model.mu) - self.target)
        return HarnessResult(
            success=success,
            regressed=self.regressed,
            checks={"structure_fidelity": True, "gap_conflict_safety": True, "access_isolation": True},
        )


def _bad_champion_registry(model_id, data):
    reg = ModelRegistry()
    reg.register(MixleAdapter(model_id, model=GaussianDistribution(0.0, 1.0)))
    return reg


def test_a_week_of_usage_raises_end_to_end_task_success(session):
    _seed_week_of_usage(session, "g")
    reg = _bad_champion_registry("g", None)
    harness = _TaskHarness(reg, "g", target=5.0, regressed=False)

    before = harness.evaluate("g")
    run = improve_once(
        session,
        model_id="g",
        verifier=_PassVerifier(),
        harness=harness,
        registry=reg,
        policy=EvolutionPolicy(objective="nll"),
    )
    after = harness.evaluate("g")

    assert run.verified and run.promoted
    assert after.success > before.success


def test_seeded_calibration_regression_auto_blocks_promotion(session):
    _seed_week_of_usage(session, "g2")
    reg = _bad_champion_registry("g2", None)
    champion = reg.get("g2")._model
    harness = _TaskHarness(reg, "g2", target=5.0, regressed=True)  # seeded: the harness always reports a regression

    run = improve_once(
        session,
        model_id="g2",
        verifier=_PassVerifier(),
        harness=harness,
        registry=reg,
        policy=EvolutionPolicy(objective="nll"),
    )

    assert run.promoted is False
    assert reg.get("g2")._model is champion  # rolled back to the original champion


def test_mining_only_uses_verified_transitions_on_the_allowed_canonical_ref(session):
    _seed_week_of_usage(session, "g3", truth_mean=5.0, canonical_ref="predict", verified=True)
    _seed_week_of_usage(session, "g3", truth_mean=-40.0, n=80, canonical_ref="untrusted", verified=True)
    _seed_week_of_usage(session, "g3", truth_mean=90.0, n=80, canonical_ref="predict", verified=False)
    reg = _bad_champion_registry("g3", None)
    harness = _TaskHarness(reg, "g3", target=5.0, regressed=False)

    run = improve_once(
        session,
        model_id="g3",
        verifier=_PassVerifier(),
        harness=harness,
        registry=reg,
        policy=EvolutionPolicy(objective="nll"),
        allowed_canonical_refs=["predict"],
    )

    assert run.promoted
    assert abs(reg.get("g3")._model.mu - 5.0) < 1.0  # neither the unverified nor the disallowed-ref usage leaked in


def test_record_usage_rejects_a_trace_missing_frozen_ic5_keys(session):
    with pytest.raises(ValueError):
        record_usage(session, "g", {"prompt": "p", "steps": []}, verified=True)
