"""H9 -- serve + monitor the production optimizer (work-plan §7-H).

Wraps H4's :func:`mixle.stochastic_opt.two_stage_stochastic_plan` behind a near-real-time
:func:`serve_reoptimize` endpoint: every solve emits an IC-5 ``{prompt, steps, outcome, provenance}``
receipt (:func:`mixle.task.trace_record.validate_trace_record`), content-hashing the solve's identity
inputs -- block model version, price scenario, capacities -- the same way E7's ``decision_receipt``
(:func:`mixle.reason.receipt.content_edge_hash`) fingerprints a lineage edge. The receipt is persisted
as a ``trace`` item in a :class:`mixle.substrate.core.Substrate`, so ``receipt_ref`` is an independently
resolvable id and, on a re-solve, ``links`` back to the receipt it supersedes -- a lineage edge, not a
silent overwrite.

:func:`on_drift` is the re-solve trigger: given a monitoring ``signal`` (a stream name plus
reference/current series and the request to re-solve with), it runs
:func:`mixle_mlops.drift_retrain.detect_stream_drift` and, only on an alarm past its nominal
false-alarm rate, re-solves and publishes a new, lineage-linked receipt.

Repo-boundary / Wave-3 race note (full detail in the PR body): this task's declared dependencies are
H4 (:mod:`mixle.stochastic_opt`, landed), E7 (:mod:`mixle.reason.receipt` / IC-5
:mod:`mixle.task.trace_record`, landed), and G7 (``mixle_mlops/monitoring.py``'s exceedance/drift
surface). G7 had not landed on ``release/0.8.0`` as of this PR -- both tasks share Wave 3 and G7 was
still mid-flight in a sibling worktree. Rather than block on it, :func:`on_drift` is built directly on
G7's own declared building block (:func:`mixle.inference.conformal.split_conformal`) via the additive
``detect_stream_drift`` in ``drift_retrain.py``; swapping in G7's richer
``exceedance_alarm``/``monitor_and_maybe_retrain`` later is a drop-in replacement for that one call, not
a change to this module's public API.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mixle.data.hashing import dataset_hash
from mixle.stochastic_opt import StochasticPlan, two_stage_stochastic_plan
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.task.trace_record import validate_trace_record

from mixle_mlops.drift_retrain import detect_stream_drift

__all__ = ["serve_reoptimize", "on_drift", "DEFAULT_SUBSTRATE"]

# A process-local receipt store so `receipt_ref`s resolve within one serving process without every
# caller having to thread a `Substrate` through `request["substrate"]`. A real deployment injects its
# own durable substrate root the same way (`request["substrate"] = Substrate(root=...)`).
DEFAULT_SUBSTRATE = Substrate()


def _inputs_hash(block_model_version: str, price_scenario: str, capacities: Any) -> str:
    """Content hash over the solve's identity fields (H9 algorithm step 1) -- independent of the
    (potentially large) grade-posterior draws, since a re-solve is keyed by *what changed* (model
    version / price scenario / plant capacities), not the sampled scenarios themselves."""
    caps = np.asarray(capacities, dtype=float).tolist() if capacities is not None else []
    return dataset_hash(
        [{"block_model_version": block_model_version, "price_scenario": price_scenario, "capacities": caps}]
    )


def _plan_to_dict(plan: StochasticPlan) -> dict[str, Any]:
    return {
        "extract": plan.extract.astype(bool).tolist(),
        "expected_value": float(plan.expected_value),
        "cvar": float(plan.cvar),
        "n_scenarios": int(plan.scenarios.shape[0]),
    }


def serve_reoptimize(request: dict[str, Any]) -> dict[str, Any]:
    """Near-real-time endpoint over H4's ``two_stage_stochastic_plan``: solve, then emit + persist an
    IC-5 receipt.

    ``request`` keys: ``posterior`` (an IC-1 ``Posterior``), ``block_cost``, ``price`` (required);
    optional ``k_scenarios`` (default 50), ``alpha`` (default 0.9), ``seed`` (default 0),
    ``block_model_version`` / ``price_scenario`` / ``capacities`` (the content-hashed identity fields),
    ``parent_receipt_ref`` (links the new receipt to the one it supersedes), and ``substrate`` (defaults
    to a process-local store).

    Returns ``{"plan": {...}, "receipt_ref": <substrate item id>, "assumptions": {...}}``.
    """
    # NOTE: `Substrate` defines `__len__`, so an *empty* substrate is falsy under `or` -- an `is None`
    # check is required here or a freshly constructed (still-empty) caller-supplied substrate would be
    # silently discarded in favor of `DEFAULT_SUBSTRATE`.
    requested_substrate = request.get("substrate")
    substrate: Substrate = requested_substrate if requested_substrate is not None else DEFAULT_SUBSTRATE
    posterior = request["posterior"]
    block_cost = request["block_cost"]
    price = float(request["price"])
    k_scenarios = int(request.get("k_scenarios", 50))
    alpha = float(request.get("alpha", 0.9))
    seed = int(request.get("seed", 0))
    block_model_version = str(request.get("block_model_version", "unknown"))
    price_scenario = str(request.get("price_scenario", "base"))
    capacities = request.get("capacities", [])
    parent_receipt_ref = request.get("parent_receipt_ref")

    rng = np.random.default_rng(seed)
    plan = two_stage_stochastic_plan(posterior, block_cost, price, k_scenarios=k_scenarios, alpha=alpha, rng=rng)

    inputs_hash = _inputs_hash(block_model_version, price_scenario, capacities)
    plan_dict = _plan_to_dict(plan)
    # Mix the identity-fields hash into the outcome hash: a receipt whose *decision* happens to match an
    # earlier one (same posterior/cost/price/seed) still fingerprints distinctly whenever the block model
    # version, price scenario, or capacities it was solved under differ.
    outcome_hash = dataset_hash([plan_dict, inputs_hash])

    assumptions = {
        "block_model_version": block_model_version,
        "price_scenario": price_scenario,
        "price": price,
        "k_scenarios": k_scenarios,
        "alpha": alpha,
        "seed": seed,
    }

    lineage = [
        {"stage": "inputs", "content_hash": inputs_hash, "parent_hash": None},
        {"stage": "plan", "content_hash": outcome_hash, "parent_hash": inputs_hash},
    ]

    receipt = {
        "prompt": f"reoptimize {block_model_version}/{price_scenario}",
        "steps": [
            {
                "tool": "two_stage_stochastic_plan",
                "args": {**assumptions, "inputs_hash": inputs_hash},
                "result": {"content_hash": outcome_hash, "parent_hash": inputs_hash, **plan_dict},
                "model": None,
                "verdict": None,
            }
        ],
        "outcome": plan_dict,
        "provenance": {
            "lineage": lineage,
            "content_hash": outcome_hash,
            "parent_receipt_ref": parent_receipt_ref,
            "created_at": time.time(),
        },
    }
    validate_trace_record(receipt)

    links = [parent_receipt_ref] if parent_receipt_ref else []
    item = SubstrateItem(
        kind="trace", text=receipt["prompt"], payload=receipt, provenance=receipt["provenance"], links=links
    )
    receipt_ref = substrate.put(item)

    return {"plan": plan_dict, "receipt_ref": receipt_ref, "assumptions": assumptions}


def on_drift(signal: dict[str, Any]) -> dict[str, Any]:
    """Monitor trigger: on a grade-forecast / demand / throughput drift alarm past its nominal
    false-alarm rate, re-solve and publish a new, lineage-linked receipt.

    ``signal`` keys: ``stream`` (name, default ``"grade_forecast"``), ``reference`` / ``current`` (the
    monitoring series), ``alpha`` (nominal false-alarm rate, default 0.1), ``request`` (the
    ``serve_reoptimize`` request to re-solve with on an alarm), and ``parent_receipt_ref`` (the prior
    receipt this signal is monitoring against -- threaded onto the re-solve so the new receipt's lineage
    points back at the plan it supersedes).

    Returns the same shape as :func:`serve_reoptimize` plus a ``"drift"`` verdict block; when no alarm
    fires, no re-solve happens and the prior receipt/plan pass through unchanged.
    """
    stream = str(signal.get("stream", "grade_forecast"))
    alpha = float(signal.get("alpha", 0.1))
    alarm = detect_stream_drift(stream, signal["reference"], signal["current"], alpha=alpha)

    parent_receipt_ref = signal.get("parent_receipt_ref")
    if not alarm["drift"]:
        return {"plan": None, "receipt_ref": parent_receipt_ref, "assumptions": {}, "drift": alarm}

    request = dict(signal.get("request") or {})
    request["parent_receipt_ref"] = parent_receipt_ref
    result = serve_reoptimize(request)
    result["drift"] = alarm
    return result
