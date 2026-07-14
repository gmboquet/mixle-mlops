"""DoD test for N5 -- ecological monitoring & decline early-warning.

The core empirical claim under test: `decline_alarm`'s one-sided split-conformal LOWER bound holds its
nominal false-alarm rate `alpha` over synthetic stable-occupancy series (no true decline), and fires on a
series with an injected downward step below `floor` -- mirroring G7's own exceedance-alarm DoD test
(`tests/test_exceedance.py`) for the opposite tail. `occupancy_trend` and `monitor_biodiversity` are
covered at a basic correctness level.
"""

from __future__ import annotations

import numpy as np

from mixle_mlops.biodiversity_monitoring import (
    biodiversity_alerts,
    decline_alarm,
    monitor_biodiversity,
    occupancy_trend,
)


class _StubHabitatBaseline:
    """Minimal IC-1-Posterior-shaped stand-in for an N1 `HabitatModel`: only `.samples` is exercised by
    `monitor_biodiversity` (per IC-1, the one method every conforming baseline is guaranteed to have)."""

    def __init__(self, mu: float, sigma: float, dim: int = 4) -> None:
        self._mu = mu
        self._sigma = sigma
        self._dim = dim

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self._mu, self._sigma, size=(n, self._dim))

    @property
    def mean(self) -> np.ndarray:
        return np.full(self._dim, self._mu)

    @property
    def cov(self) -> np.ndarray:
        return np.eye(self._dim) * self._sigma**2

    def credible_interval(self, level: float):
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2.0)
        return np.full(self._dim, self._mu - z * self._sigma), np.full(self._dim, self._mu + z * self._sigma)

    def derived_quantity(self, fn, n, rng):  # pragma: no cover -- not exercised here
        raise NotImplementedError


def test_decline_alarm_empirical_false_alarm_rate_bounded_by_alpha():
    """Over N synthetic stable-occupancy series (no true decline), `decline_alarm`'s empirical
    false-alarm rate stays at/below `alpha`, within Monte-Carlo error -- the N5 Definition of Done."""
    rng = np.random.default_rng(20260714)
    alpha = 0.1
    n_trials = 500
    n_cal, n_test = 60, 5
    mu, sigma = 0.5, 0.05

    # A bona fide "known no-decline" scenario: the floor sits comfortably below the stable regime's
    # natural range, not exactly on the alarm's own decision boundary.
    floor = mu - 4.5 * sigma

    false_alarms = 0
    for _ in range(n_trials):
        cal_series = rng.normal(mu, sigma, size=n_cal)
        index_series = rng.normal(mu, sigma, size=n_test)
        if decline_alarm(index_series, cal_series, floor=floor, alpha=alpha):
            false_alarms += 1

    empirical_rate = false_alarms / n_trials
    slack = 3.0 * np.sqrt(alpha * (1.0 - alpha) / n_trials)  # ~3-sigma Monte-Carlo slack around alpha
    assert empirical_rate <= alpha + slack, (empirical_rate, alpha, slack)


def test_decline_alarm_triggers_on_an_injected_downward_step_below_floor():
    """Sanity check the alarm isn't just trivially conservative: a series that has genuinely stepped
    down below the floor is flagged."""
    rng = np.random.default_rng(11)
    cal_series = rng.normal(0.5, 0.05, size=60)
    stepped_down_series = rng.normal(0.2, 0.05, size=5)  # injected downward step
    assert decline_alarm(stepped_down_series, cal_series, floor=0.35, alpha=0.1)


def test_decline_alarm_does_not_trigger_on_a_stable_series_above_floor():
    rng = np.random.default_rng(3)
    cal_series = rng.normal(0.6, 0.05, size=60)
    stable_series = rng.normal(0.6, 0.05, size=5)
    assert not decline_alarm(stable_series, cal_series, floor=0.35, alpha=0.1)


def test_occupancy_trend_flags_a_confident_decline():
    rng = np.random.default_rng(5)
    n = 40
    series = 1.0 - 0.02 * np.arange(n) + rng.normal(0.0, 0.01, size=n)
    result = occupancy_trend(series)
    assert result["slope"] < 0.0
    assert result["ci"][0] <= result["ci"][1]
    assert result["declining"] is True


def test_occupancy_trend_does_not_flag_a_stationary_series():
    rng = np.random.default_rng(6)
    n = 40
    series = rng.normal(1.0, 0.05, size=n)
    result = occupancy_trend(series)
    assert result["declining"] is False


def test_occupancy_trend_accepts_explicit_times():
    rng = np.random.default_rng(7)
    times = np.array([0.0, 1.0, 2.0, 5.0, 8.0, 13.0])
    series = 2.0 - 0.3 * times + rng.normal(0.0, 0.01, size=times.size)
    result = occupancy_trend(series, times=times)
    assert result["slope"] < 0.0


def test_occupancy_trend_handles_a_degenerate_single_point_series():
    result = occupancy_trend(np.array([1.0]))
    assert result["declining"] is False
    assert result["ci"] == (0.0, 0.0)


def test_monitor_biodiversity_flags_decline_and_triggers_retrain(monkeypatch, tmp_path):
    calls = {}

    def _fake_retrain_and_promote(reference, current, **kwargs):
        calls["reference_len"] = len(reference)
        calls["current_len"] = len(current)
        return "v-fake-1"

    monkeypatch.setattr("mixle_mlops.drift_retrain.retrain_and_promote", _fake_retrain_and_promote)

    baseline = _StubHabitatBaseline(mu=0.6, sigma=0.05)
    current_index = np.random.default_rng(9).normal(0.2, 0.05, size=10)  # a clear decline below floor

    n_before = len(biodiversity_alerts())
    record = monitor_biodiversity(baseline, current_index, floor=0.4)

    assert record["declining"] is True
    assert record["retrain_triggered"] is True
    assert record["new_version"] == "v-fake-1"
    assert calls["reference_len"] > 0
    assert calls["current_len"] == current_index.size
    assert len(biodiversity_alerts()) == n_before + 1
    assert biodiversity_alerts()[-1]["floor"] == 0.4


def test_monitor_biodiversity_no_action_when_stable_and_compliant(monkeypatch):
    def _unexpected_retrain(*args, **kwargs):  # pragma: no cover -- should never be called
        raise AssertionError("retrain should not trigger for a stable, compliant baseline")

    monkeypatch.setattr("mixle_mlops.drift_retrain.retrain_and_promote", _unexpected_retrain)

    baseline = _StubHabitatBaseline(mu=0.6, sigma=0.03)
    rng = np.random.default_rng(13)
    current_index = rng.normal(0.6, 0.03, size=200)  # same regime, comfortably compliant

    record = monitor_biodiversity(baseline, current_index, floor=0.2)

    assert record["declining"] is False
    assert record["drift"] is False
    assert record["retrain_triggered"] is False
    assert record["new_version"] is None
