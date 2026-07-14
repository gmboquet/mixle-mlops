"""H9 DoD -- serve + monitor the production optimizer (notes/exec/workstream-H.md).

An injected grade-forecast drift trips `on_drift`'s exceedance alarm, which re-solves
`mixle.stochastic_opt.two_stage_stochastic_plan` behind `serve_reoptimize` and publishes a new IC-5
receipt into the substrate. The new receipt's `receipt_ref` resolves to a hashed lineage (a
`provenance.content_hash`) distinct from the pre-drift plan's, and its `links` chain back to the
receipt it supersedes -- a lineage edge, not a silent overwrite. A no-alarm signal is a no-op: no
re-solve, the prior receipt passes through unchanged.
"""

from __future__ import annotations

import numpy as np

from mixle.substrate.core import Substrate
from mixle.task.trace_record import validate_trace_record
from mixle_mlops.production_serving import on_drift, serve_reoptimize

N_BLOCKS = 4


class _GradePosterior:
    """A minimal IC-1 `Posterior` stub: per-block grade centered on `mean_grade`."""

    def __init__(self, n_blocks: int, mean_grade: float):
        self.n_blocks = n_blocks
        self.mean_grade = mean_grade

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.clip(self.mean_grade + rng.normal(0.0, 0.05, size=(n, self.n_blocks)), 0.0, None)


def _base_request(substrate: Substrate, *, mean_grade: float = 1.0, price_scenario: str = "base") -> dict:
    return {
        "posterior": _GradePosterior(N_BLOCKS, mean_grade),
        "block_cost": np.full(N_BLOCKS, 0.8),
        "price": 1.0,
        "k_scenarios": 30,
        "alpha": 0.9,
        "seed": 0,
        "block_model_version": "bm-v1",
        "price_scenario": price_scenario,
        "capacities": [100.0, 100.0],
        "substrate": substrate,
    }


def test_serve_reoptimize_emits_a_valid_ic5_receipt():
    substrate = Substrate()
    result = serve_reoptimize(_base_request(substrate))

    assert set(result) >= {"plan", "receipt_ref", "assumptions"}
    item = substrate.get(result["receipt_ref"])
    assert item is not None
    validate_trace_record(item.payload)
    assert item.payload["outcome"] == result["plan"]


def test_on_drift_triggers_a_resolve_with_a_distinct_lineage_edge():
    substrate = Substrate()

    pre = serve_reoptimize(_base_request(substrate, mean_grade=1.0))
    pre_item = substrate.get(pre["receipt_ref"])
    assert pre_item is not None

    # Deterministic injected drift: the reference regime centers on 0 (residual-from-forecast units);
    # the current window is a fixed, unambiguous shift well past the calibrated upper bound -- no
    # random-seed luck required for the alarm to fire.
    reference = np.zeros(200)
    current = np.full(200, 5.0)

    signal = {
        "stream": "grade_forecast",
        "reference": reference,
        "current": current,
        "alpha": 0.1,
        "parent_receipt_ref": pre["receipt_ref"],
        # the re-solve reflects the drifted grade belief (a lower mean grade than the pre-drift plan used)
        "request": _base_request(substrate, mean_grade=0.4),
    }

    post = on_drift(signal)

    assert post["drift"]["drift"] is True
    assert post["receipt_ref"] is not None
    assert post["receipt_ref"] != pre["receipt_ref"]

    post_item = substrate.get(post["receipt_ref"])
    assert post_item is not None
    validate_trace_record(post_item.payload)

    # The new receipt resolves to a hashed lineage distinct from the pre-drift one, and its lineage
    # chains back to the receipt it supersedes.
    assert post_item.provenance["content_hash"] != pre_item.provenance["content_hash"]
    assert pre["receipt_ref"] in post_item.links


def test_on_drift_is_a_noop_when_no_alarm_fires():
    substrate = Substrate()
    pre = serve_reoptimize(_base_request(substrate))

    reference = np.zeros(200)
    current = np.zeros(200)  # identical regime -- no drift

    signal = {
        "stream": "grade_forecast",
        "reference": reference,
        "current": current,
        "alpha": 0.1,
        "parent_receipt_ref": pre["receipt_ref"],
        "request": _base_request(substrate, mean_grade=1.0),
    }

    result = on_drift(signal)
    assert result["drift"]["drift"] is False
    assert result["plan"] is None
    assert result["receipt_ref"] == pre["receipt_ref"]  # no re-solve; the prior receipt passes through


def test_receipt_content_hash_distinguishes_identity_tags_even_with_an_identical_decision():
    substrate = Substrate()
    a = serve_reoptimize(_base_request(substrate, price_scenario="scenario-a"))
    b = serve_reoptimize(_base_request(substrate, price_scenario="scenario-b"))

    # Same posterior/cost/price/seed -> the solved decision itself is identical...
    assert a["plan"] == b["plan"]
    # ...but the two receipts still fingerprint distinctly, because the price-scenario identity tag
    # (a content-hashed input, per the H9 algorithm) differs.
    item_a = substrate.get(a["receipt_ref"])
    item_b = substrate.get(b["receipt_ref"])
    assert item_a.provenance["content_hash"] != item_b.provenance["content_hash"]
