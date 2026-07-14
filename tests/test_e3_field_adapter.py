"""DoD test for E3 -- ``FieldPosteriorAdapter`` serving a physics posterior (IC-1/IC-2).

The fixture below is a small, directly-constructed object satisfying the frozen IC-1 ``Posterior``
protocol (``mixle.reason.posterior_protocol``) -- the same style the IC-1 conformance test itself uses --
standing in for a fitted ``mixle_pde.latent.PosteriorField3D`` (E1 lands real IC-1 conformance there).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from scipy import stats

from mixle_mlops.core.adapters import CapabilityError, ChatMessage, ChatRequest
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.models.field_posterior import FieldPosteriorAdapter, register_field_posterior


class _ToyDerivedQuantity:
    def __init__(self, samples: np.ndarray) -> None:
        self.samples = np.asarray(samples, dtype=float)
        self.prior_dominated = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a), np.quantile(self.samples, 1.0 - a)


class _ToyFieldPosterior:
    """A minimal IC-1-conforming fixture: an independent-Gaussian field over ``d`` cells."""

    def __init__(self, mean: np.ndarray, var: np.ndarray) -> None:
        self._mean = np.asarray(mean, dtype=float)
        self._var = np.asarray(var, dtype=float)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._mean + rng.standard_normal((n, self._mean.size)) * np.sqrt(self._var)

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        return np.diag(self._var)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        z = stats.norm.ppf(0.5 + level / 2.0)
        spread = z * np.sqrt(self._var)
        return self._mean - spread, self._mean + spread

    def derived_quantity(self, fn, n, rng) -> _ToyDerivedQuantity:
        return _ToyDerivedQuantity(fn(self.samples(n, rng)))


@pytest.fixture
def posterior() -> _ToyFieldPosterior:
    rng = np.random.default_rng(0)
    d = 12
    mean = rng.uniform(0.05, 0.3, size=d)  # e.g. a porosity-like field
    var = np.full(d, 0.01)
    return _ToyFieldPosterior(mean, var)


def test_isinstance_posterior_protocol(posterior):
    """Confirms the fixture is a fair stand-in for a real IC-1 conforming field posterior."""
    try:
        from mixle.reason.posterior_protocol import Posterior

        assert isinstance(posterior, Posterior)
    except ImportError:
        pytest.skip("mixle.reason.posterior_protocol (IC-1) not landed yet in this checkout")


def test_registers_and_advertises_capabilities(posterior):
    registry = ModelRegistry()
    adapter = register_field_posterior(registry, "toy-field", posterior)
    assert registry.has("toy-field")
    assert adapter.capabilities() == {"predict", "decide", "score"}
    assert adapter.kind == "field"


def test_predict_returns_calibrated_slice(posterior):
    adapter = FieldPosteriorAdapter("toy-field", posterior=posterior)
    region = {"indices": [0, 1, 2]}
    out = asyncio.run(adapter.predict([region], level=0.9))
    rec = out["records"][0]
    assert rec["indices"] == [0, 1, 2]
    assert len(rec["mean"]) == 3
    lo, hi = rec["interval"]
    assert all(lo_i <= hi_i for lo_i, hi_i in zip(lo, hi))
    assert np.allclose(rec["mean"], posterior.mean[:3])


def test_decide_returns_an_action_id(posterior):
    adapter = FieldPosteriorAdapter("toy-field", posterior=posterior)
    region = {"indices": list(range(12))}  # whole field -> region "mass"

    def newsvendor_loss(action, draw):
        # stock `action` units against uncertain demand `draw`: over/under-stocking asymmetric loss.
        draw = np.asarray(draw, dtype=float)
        return np.where(draw > action, 2.0 * (draw - action), 1.0 * (action - draw))

    actions = [1.0, 2.0, 3.0, 4.0]
    out = asyncio.run(adapter.decide([region], loss=newsvendor_loss, actions=actions, n=2000, seed=0))
    assert out["action"] in actions
    assert 0 <= out["action_index"] < len(actions)
    assert "risk_profile" in out
    assert out["prior_dominated"] is False


def test_score_reports_empirical_coverage(posterior):
    adapter = FieldPosteriorAdapter("toy-field", posterior=posterior)
    region = {"indices": [0]}
    truth = [[float(posterior.mean[0])]]  # dead-on truth is covered at any reasonable level
    out = asyncio.run(adapter.score([region], truth=truth, level=0.9))
    assert out["coverage"] == 1.0
    assert out["n"] == 1


def test_stream_refuses_like_a_non_chat_model(posterior):
    adapter = FieldPosteriorAdapter("toy-field", posterior=posterior)

    async def _drain():
        async for _ in adapter.stream(ChatRequest(messages=[ChatMessage(role="user", content="hi")])):
            pass

    with pytest.raises(CapabilityError):
        asyncio.run(_drain())


def test_needs_one_of_posterior_ref_or_registry():
    with pytest.raises(ValueError):
        FieldPosteriorAdapter("no-source")
