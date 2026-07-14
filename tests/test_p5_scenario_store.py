"""P5 -- scenario library & comparison (IC-11 Scenario/SimResult persistence, diff, decision memo).

``mixle_pde.simulation_service`` (IC-11) owns the real ``Scenario``/``ScenarioStep``/``SimResult``
dataclasses; per the ``mixle_pde``-optional precedent already established by ``test_p1_simulate_tool.py``
in this repo, this suite stands in a minimal local dataclass pair with the exact same field shape so the
suite runs whether or not the sibling ``mixle-pde`` package/PR has landed. ``scenario_store`` only ever
duck-types its ``scenario``/``result`` arguments (a serialisable ``steps``/``couplings``/``provenance``
scenario and a ``result_ref``/``provenance`` result), so a real IC-11 object round-trips identically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from mixle.task.trace_record import validate_trace_record
from mixle_mlops.artifacts.scenario_store import (
    diff_scenarios,
    load_scenario,
    save_scenario,
    scenario_memo,
)


@dataclass
class _Step:
    op: str
    inputs_ref: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Scenario:
    steps: list[_Step]
    couplings: list[tuple] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SimResult:
    result_ref: str
    uncertainty: Any | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def _hash_of(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def test_save_load_diff_and_memo_round_trip(tmp_path):
    store_dir = str(tmp_path / "scenarios")

    baseline_scenario = _Scenario(
        steps=[_Step(op="flow", inputs_ref="field-1", params={"conductivity": 1.0, "source": 1.0})],
        provenance={"author": "p5-dod"},
    )
    baseline_result = _SimResult(result_ref=_hash_of("baseline-arrays"), provenance={"op": "flow"})

    alt_scenario = _Scenario(
        steps=[_Step(op="flow", inputs_ref="field-1", params={"conductivity": 2.5, "source": 1.0})],
        provenance={"author": "p5-dod"},
    )
    alt_result = _SimResult(result_ref=_hash_of("alt-arrays"), provenance={"op": "flow"})

    saved_baseline = save_scenario("baseline", baseline_scenario, baseline_result, store_dir=store_dir)
    saved_alt = save_scenario("alt", alt_scenario, alt_result, store_dir=store_dir)

    assert saved_baseline.result_ref == baseline_result.result_ref
    assert saved_alt.result_ref == alt_result.result_ref

    # round trip
    loaded_baseline = load_scenario("baseline", store_dir=store_dir)
    loaded_alt = load_scenario("alt", store_dir=store_dir)
    assert loaded_baseline == saved_baseline
    assert loaded_alt == saved_alt

    diff = diff_scenarios(loaded_baseline, loaded_alt)

    # changed step params
    step_diff = diff["steps"][0]
    assert step_diff["params"]["conductivity"] == {"a": 1.0, "b": 2.5}
    assert "source" not in step_diff["params"]  # unchanged param excluded from the delta

    # result_ref delta with each side's provenance
    assert diff["result_ref"]["changed"] is True
    assert diff["result_ref"]["a"] == baseline_result.result_ref
    assert diff["result_ref"]["b"] == alt_result.result_ref
    assert diff["result_ref"]["provenance"]["a"] == loaded_baseline.provenance
    assert diff["result_ref"]["provenance"]["b"] == loaded_alt.provenance

    memo = scenario_memo(loaded_baseline, loaded_alt)
    validate_trace_record(memo)
    assert memo["outcome"] == diff
    lineage = memo["provenance"]["decision_receipt"]["provenance"]["lineage"]
    assert lineage[0]["content_hash"] == baseline_result.result_ref


def test_diff_scenarios_identical_result_is_provably_the_same(tmp_path):
    store_dir = str(tmp_path / "scenarios2")
    scenario = _Scenario(steps=[_Step(op="transport", inputs_ref="x", params={"dt": 1.0})])
    result = _SimResult(result_ref=_hash_of("same-arrays"))

    a = save_scenario("a", scenario, result, store_dir=store_dir)
    b = save_scenario("b", scenario, result, store_dir=store_dir)

    diff = diff_scenarios(a, b)
    assert diff["result_ref"]["changed"] is False
    assert diff["steps"] == []

    memo = scenario_memo(a, b)
    validate_trace_record(memo)
