"""E10 — physics/calibration verifier: PhysicalVerifier + CalibrationVerifier (IC-6 `Verifier`),
and their registration into the gateway's `build_verifier`."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from scipy.stats import norm

from mixle_mlops.gateway.verifiers import build_verifier
from mixle_mlops.verification.base import Verdict
from mixle_mlops.verification.physics import CalibrationVerifier, PhysicalVerifier


class _GaussianPosterior:
    """Minimal IC-1-shaped posterior: independent per-component Gaussians about a fixed mean.

    Only implements `credible_interval`, which is all `CalibrationVerifier` needs — the point is that
    a caller never has to import the real `Posterior` protocol to satisfy it (duck typing)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean_ = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        z = norm.ppf(0.5 + level / 2.0)
        return self.mean_ - z * self.std, self.mean_ + z * self.std


# ---------------------------------------------------------------------------
# PhysicalVerifier
# ---------------------------------------------------------------------------


def test_physical_verifier_rejects_negative_porosity_naming_the_bound():
    verdict = PhysicalVerifier().verify({"porosity": np.array([0.2, 0.3, -0.05, 0.4])}, {})
    assert isinstance(verdict, Verdict)
    assert verdict.passed is False
    assert verdict.kind == "physical"
    assert any("porosity" in reason and "bounds" in reason for reason in verdict.reasons)


def test_physical_verifier_rejects_negative_density():
    verdict = PhysicalVerifier().verify({"density": np.array([2200.0, -10.0, 2400.0])}, {})
    assert verdict.passed is False
    assert verdict.kind == "physical"
    assert any("density" in reason for reason in verdict.reasons)


def test_physical_verifier_flags_mass_balance_violation():
    claim = {"mass_balance": {"inflow": 100.0, "outflow": 40.0, "storage_change": 2.0}}
    verdict = PhysicalVerifier().verify(claim, {})
    assert verdict.passed is False
    assert verdict.kind == "physical"
    assert any("mass balance" in reason for reason in verdict.reasons)


def test_physical_verifier_flags_vs_exceeding_vp():
    claim = {"vp": np.array([1500.0, 2000.0]), "vs": np.array([1800.0, 900.0])}
    verdict = PhysicalVerifier().verify(claim, {})
    assert verdict.passed is False
    assert any("rock-physics" in reason for reason in verdict.reasons)


def test_physical_verifier_accepts_a_valid_admissible_field():
    claim = {
        "porosity": np.array([0.10, 0.22, 0.31]),
        "density": np.array([2200.0, 2400.0, 2600.0]),
        "bulk_modulus": np.array([2.0e10, 2.5e10, 3.0e10]),
        "shear_modulus": np.array([1.0e10, 1.2e10, 1.5e10]),
        "vp": np.array([2500.0, 3000.0, 3500.0]),
        "vs": np.array([1200.0, 1500.0, 1800.0]),
        "mass_balance": {"inflow": 100.0, "outflow": 98.0, "storage_change": 2.0},
    }
    verdict = PhysicalVerifier().verify(claim, {})
    assert verdict.passed is True
    assert verdict.kind == "physical"
    assert verdict.score == pytest.approx(1.0)
    assert verdict.reasons == []


# ---------------------------------------------------------------------------
# CalibrationVerifier
# ---------------------------------------------------------------------------


def test_calibration_verifier_passes_a_well_calibrated_posterior():
    rng = np.random.default_rng(0)
    d = 4000
    true_std = 1.0
    mean = np.zeros(d)
    truth = mean + rng.normal(scale=true_std, size=d)
    posterior = _GaussianPosterior(mean, np.full(d, true_std))

    verdict = CalibrationVerifier().verify(
        {"posterior": posterior, "nominal": 0.9},
        {"truth": truth},
    )
    assert verdict.kind == "calibration"
    assert verdict.passed is True
    assert verdict.score == pytest.approx(0.9, abs=0.05)


def test_calibration_verifier_fails_when_posterior_is_9x_overconfident():
    # The posterior's assumed spread is 9x too NARROW relative to the true residual variance (equivalently:
    # the held-out truth is drawn from a distribution whose variance is inflated 9x relative to what the
    # posterior's credible interval assumes). This is the direction that actually produces "empirical coverage
    # much less than nominal" the DoD describes -- widening a posterior's own interval while holding truth
    # fixed can only ever raise coverage, never collapse it, so an overconfident (too-narrow) posterior is the
    # only way to reproduce the specified observable.
    rng = np.random.default_rng(0)
    d = 4000
    true_std = 3.0  # 3**2 == 9x the variance the posterior below assumes
    mean = np.zeros(d)
    truth = mean + rng.normal(scale=true_std, size=d)
    posterior = _GaussianPosterior(mean, np.full(d, 1.0))

    verdict = CalibrationVerifier().verify(
        {"posterior": posterior, "nominal": 0.9},
        {"truth": truth},
    )
    assert verdict.kind == "calibration"
    assert verdict.passed is False
    assert verdict.score < 0.9 - 0.05
    assert any("overconfident" in reason or "coverage" in reason for reason in verdict.reasons)


def test_calibration_verifier_requires_a_posterior_and_truth():
    v = CalibrationVerifier()
    missing_posterior = v.verify({}, {"truth": np.zeros(3)})
    assert missing_posterior.passed is False and missing_posterior.kind == "calibration"

    posterior = _GaussianPosterior(np.zeros(3), np.ones(3))
    missing_truth = v.verify({"posterior": posterior}, {})
    assert missing_truth.passed is False and missing_truth.kind == "calibration"


# ---------------------------------------------------------------------------
# gateway wiring: build_verifier({"kind": "physical" | "calibration", ...})
# ---------------------------------------------------------------------------


def test_build_verifier_wires_physical_kind():
    verifier = build_verifier({"kind": "physical", "context": {}}, registry=None)
    assert verifier is not None
    failing_claim = json.dumps({"porosity": [0.1, 0.2, -0.1]})
    assert asyncio.run(verifier(failing_claim)) == 0.0

    passing_claim = json.dumps({"porosity": [0.1, 0.2, 0.3]})
    assert asyncio.run(verifier(passing_claim)) == pytest.approx(1.0)


def test_build_verifier_wires_calibration_kind_with_a_static_claim():
    posterior = _GaussianPosterior(np.zeros(2000), np.ones(2000))
    rng = np.random.default_rng(1)
    truth = rng.normal(scale=1.0, size=2000)

    verifier = build_verifier(
        {
            "kind": "calibration",
            "claim": {"posterior": posterior, "nominal": 0.9},
            "context": {"truth": truth},
        },
        registry=None,
    )
    assert verifier is not None
    score = asyncio.run(verifier("candidate text is irrelevant when claim is static"))
    assert score == pytest.approx(0.9, abs=0.05)


def test_build_verifier_returns_none_for_unknown_kind():
    assert build_verifier({"kind": "not-a-real-kind"}, registry=None) is None
