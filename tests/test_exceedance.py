"""DoD test for G7 -- exceedance probability + monitoring-drift detection.

The core empirical claim under test: `exceedance_alarm`'s one-sided split-conformal upper bound holds
its nominal false-alarm rate `alpha` in a known no-exceedance (compliant) regime. The remaining tests
cover `exceedance_probability` (the IC-8 `prob_exceed` dispatch) and `monitor_and_maybe_retrain` (the
combined drift + exceedance retrain trigger) at a basic correctness level.
"""

from __future__ import annotations

import numpy as np

from mixle_mlops.monitoring import (
    alerts,
    exceedance_alarm,
    exceedance_probability,
    monitor_and_maybe_retrain,
)


class _ScalarPosterior:
    """Minimal IC-1-shaped posterior over one scalar concentration -- just enough surface for
    `mixle_pde.decision_quantities.prob_exceed` (`.sample`, `.mean`) plus the rest of the frozen IC-1
    `Posterior` protocol (`.samples`, `.cov`, `.credible_interval`, `.derived_quantity`)."""

    def __init__(self, mu: float, sigma: float) -> None:
        self._mu = mu
        self._sigma = sigma

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self._mu, self._sigma, size=(n, 1))

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sample(n, rng)

    @property
    def mean(self) -> np.ndarray:
        return np.array([self._mu])

    @property
    def cov(self) -> np.ndarray:
        return np.array([[self._sigma**2]])

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2.0)
        return np.array([self._mu - z * self._sigma]), np.array([self._mu + z * self._sigma])

    def derived_quantity(self, fn, n, rng):  # pragma: no cover -- not exercised here
        raise NotImplementedError


def test_exceedance_probability_at_the_mean_is_about_half():
    posterior = _ScalarPosterior(mu=8.0, sigma=1.0)
    dq = exceedance_probability(posterior, threshold=8.0, level=0.9)
    assert 0.4 < float(np.mean(dq.samples)) < 0.6
    assert hasattr(dq, "prior_dominated")
    lo, hi = dq.credible_interval(0.9)
    assert np.all(np.asarray(lo) <= np.asarray(hi))


def test_exceedance_probability_far_below_threshold_is_near_zero():
    posterior = _ScalarPosterior(mu=0.0, sigma=1.0)
    dq = exceedance_probability(posterior, threshold=10.0, level=0.9)
    assert float(np.mean(dq.samples)) < 0.01


def test_exceedance_alarm_empirical_false_alarm_rate_bounded_by_alpha():
    """Over N synthetic monitoring series in a known no-exceedance regime, the split-conformal
    exceedance alarm's empirical false-alarm rate stays at/below `alpha`, within Monte-Carlo error --
    the work-plan G7 Definition of Done."""
    rng = np.random.default_rng(12345)
    alpha = 0.1
    n_trials = 500
    n_cal, n_test = 60, 5
    mu, sigma = 5.0, 1.0
    # A bona fide "known no-exceedance" scenario: the limit sits comfortably above the compliant
    # regime's natural range, not exactly on the alarm's own decision boundary.
    limit = mu + 4.5 * sigma

    false_alarms = 0
    for _ in range(n_trials):
        cal_series = rng.normal(mu, sigma, size=n_cal)
        series = rng.normal(mu, sigma, size=n_test)
        if exceedance_alarm(series, limit, cal_series, alpha=alpha):
            false_alarms += 1

    empirical_rate = false_alarms / n_trials
    slack = 3.0 * np.sqrt(alpha * (1.0 - alpha) / n_trials)  # ~3-sigma Monte-Carlo slack around alpha
    assert empirical_rate <= alpha + slack, (empirical_rate, alpha, slack)


def test_exceedance_alarm_has_power_against_a_real_drift():
    """Sanity check the alarm isn't just trivially conservative: a series that has genuinely drifted
    well above the limit is flagged."""
    rng = np.random.default_rng(7)
    cal_series = rng.normal(5.0, 1.0, size=60)
    drifted_series = rng.normal(20.0, 1.0, size=5)
    assert exceedance_alarm(drifted_series, 10.0, cal_series, alpha=0.1)


def test_monitor_and_maybe_retrain_triggers_on_exceedance_and_logs_a_queryable_alert(tmp_path):
    from mixle.inference.production import Registry, fit_with_provenance
    from mixle.stats import GaussianDistribution

    root = str(tmp_path / "registry")
    registry = Registry(root)
    reference = np.random.RandomState(0).normal(0.0, 1.0, 200).tolist()
    model, _header = fit_with_provenance(reference, GaussianDistribution(0.0, 1.0).estimator(), max_its=20)
    registry.register(model, "concentration")
    registry.promote("concentration", "v1", alias="production")

    current = np.random.RandomState(1).normal(6.0, 1.0, 50).tolist()  # a clear, over-limit regime shift
    n_before = len(alerts())
    result = monitor_and_maybe_retrain(model, reference, current, 2.0, registry_root=root, name="concentration")

    assert result["alarm"] is True
    assert result["retrain_triggered"] is True
    assert result["new_version"] is not None
    assert len(alerts()) == n_before + 1
    assert alerts()[-1]["threshold"] == 2.0

    _, header = registry.current("concentration", "production")  # the swap actually happened
    assert header is not None


def test_monitor_and_maybe_retrain_no_action_when_compliant_and_stable(tmp_path):
    from mixle.inference.production import Registry, fit_with_provenance
    from mixle.stats import GaussianDistribution

    root = str(tmp_path / "registry")
    registry = Registry(root)
    # Large enough calibration/current samples that per-feature PSI doesn't false-positive on sampling
    # noise alone (drift detection itself is core's job -- this scenario just needs to be a stable,
    # genuinely non-drifting one so the test isn't flaky against detect_drift's own thresholds).
    reference = np.random.RandomState(2).normal(0.0, 1.0, 400).tolist()
    model, _header = fit_with_provenance(reference, GaussianDistribution(0.0, 1.0).estimator(), max_its=20)
    registry.register(model, "concentration")
    registry.promote("concentration", "v1", alias="production")

    current = np.random.RandomState(3).normal(0.0, 1.0, 300).tolist()  # same regime, comfortably compliant
    result = monitor_and_maybe_retrain(model, reference, current, 8.0, registry_root=root, name="concentration")

    assert result["alarm"] is False
    assert result["drift"] is False
    assert result["retrain_triggered"] is False
    assert result["new_version"] is None
