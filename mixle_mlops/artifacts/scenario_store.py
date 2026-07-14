"""P5 -- scenario library & comparison (work-plan workstream P; IC-11 / IC-2 / IC-5).

A named, on-disk library of forward-simulation runs: ``save_scenario`` pins an IC-11 ``Scenario`` next
to the content-hashed ``SimResult`` it produced, ``load_scenario`` round-trips it by name,
``diff_scenarios`` compares two saved runs step-by-step, and ``scenario_memo`` turns that comparison
into an IC-5 ``TraceRecord`` decision memo carrying an E7 provenance lineage -- so a "what if we'd used
a lower conductivity" question always resolves to hashed, re-derivable artifacts rather than a claim
anyone has to take on faith.

Backing store: a bare directory of ``{blob_id}.bin`` + ``{blob_id}.json`` pairs written through
``mixle_mlops.multimodal.store.LocalBlobStore`` (store.py:70), plus a small ``scenario_index.json``
sidecar mapping the human-given ``name`` to the blob id -- ``LocalBlobStore`` itself is UUID-keyed (it
backs anonymous chat uploads), so the name index is the thin layer this module adds on top of it.

``scenario``/``result`` are accepted duck-typed rather than imported from ``mixle_pde.simulation_service``:
this mirrors the precedent ``mcp/sim_tools.py`` and ``tests/test_p1_simulate_tool.py`` already set for
IC-11 (a sibling-package, possibly-not-yet-landed dependency) -- a real ``Scenario``/``SimResult``
dataclass round-trips through here unchanged since this module only ever needs
``scenario.steps``/``.couplings``/``.provenance`` and ``result.result_ref``/``.provenance`` to exist.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mixle.reason.receipt import content_edge_hash, decision_receipt
from mixle.substrate.core import Substrate
from mixle.task.trace_record import validate_trace_record

from ..multimodal.store import LocalBlobStore

if TYPE_CHECKING:  # pragma: no cover - type-checking only; see module docstring on the duck-typing choice
    from mixle_pde.simulation_service import Scenario, SimResult

SCENARIO_STORE_SCHEMA = "mixle_mlops.scenario_store/v1"
_INDEX_FILENAME = "scenario_index.json"


@dataclass
class SavedScenario:
    """A named scenario pinned to the (content-hashed) result it produced."""

    name: str
    scenario: dict  # serialised IC-11 Scenario
    result_ref: str  # content-hashed SimResult ref
    provenance: dict = field(default_factory=dict)


def _to_plain(obj: Any) -> Any:
    """Best-effort JSON-safe serialisation: a dataclass (IC-11's ``Scenario``/``ScenarioStep``/
    ``SimResult``, or a test/production stand-in shaped like them) becomes a plain dict via
    ``dataclasses.asdict``; a plain dict/list passes through unchanged (recursively)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _result_ref(result: Any) -> str:
    """Pull the IC-2 content-hashed handle off a ``SimResult``-shaped object. IC-11 defines
    ``SimResult.result_ref`` as *already* the frozen ``sha256_of_arrays`` content hash of the result
    arrays (``mixle_pde.simulation_service.write_result_artifact`` derives it via that exact rule), so
    no re-hashing is needed here -- reusing the IC-2 hash means trusting the handle IC-11 already
    stamped with it, not recomputing it from scratch."""
    ref = getattr(result, "result_ref", None)
    if ref is None and isinstance(result, dict):
        ref = result.get("result_ref")
    if not ref:
        raise ValueError("result has no 'result_ref' (an IC-11 SimResult must carry a content-hashed ref)")
    return str(ref)


def _result_provenance(result: Any) -> dict[str, Any]:
    prov = getattr(result, "provenance", None)
    if prov is None and isinstance(result, dict):
        prov = result.get("provenance")
    return _to_plain(prov) if prov else {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _index_path(store: LocalBlobStore) -> Path:
    return store.root / _INDEX_FILENAME


def _load_index(store: LocalBlobStore) -> dict[str, str]:
    path = _index_path(store)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_index(store: LocalBlobStore, index: dict[str, str]) -> None:
    _index_path(store).write_text(json.dumps(index, indent=2, sort_keys=True))


def save_scenario(name: str, scenario: "Scenario", result: "SimResult", *, store_dir: str) -> SavedScenario:
    """Write the serialised ``scenario`` + ``result``'s content hash under ``name`` via
    ``LocalBlobStore.put`` (store.py:83), stamping the IC-2 content hash of the result and an E7
    lineage-edge hash of the scenario body into ``provenance``. Idempotent by name: saving the same
    ``name`` again overwrites the index entry with a fresh blob."""
    store = LocalBlobStore(root=store_dir)

    scenario_dict = _to_plain(scenario)
    result_ref = _result_ref(result)
    provenance = {
        "schema": SCENARIO_STORE_SCHEMA,
        "content_hash": result_ref,  # IC-2 content hash of the SimResult this scenario produced
        "scenario_hash": content_edge_hash(scenario_dict),  # E7-style lineage edge over the scenario body
        "result_provenance": _result_provenance(result),
        "saved_at": _now_iso(),
    }

    saved = SavedScenario(name=name, scenario=scenario_dict, result_ref=result_ref, provenance=provenance)
    payload = json.dumps(dataclasses.asdict(saved), indent=2, sort_keys=True, default=str).encode("utf-8")
    record = store.put(payload, filename=f"{name}.json", content_type="application/json")

    index = _load_index(store)
    index[name] = record.id
    _save_index(store, index)
    return saved


def load_scenario(name: str, *, store_dir: str) -> SavedScenario:
    """Reconstruct the ``SavedScenario`` written by ``save_scenario`` for ``name``."""
    store = LocalBlobStore(root=store_dir)
    index = _load_index(store)
    blob_id = index.get(name)
    if blob_id is None or not store.has(blob_id):
        raise KeyError(f"no saved scenario named {name!r} under {store_dir!r}")
    _, data = store.get(blob_id)
    payload = json.loads(data.decode("utf-8"))
    return SavedScenario(**payload)


def _param_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keys present in either side whose values differ; unchanged keys are omitted."""
    delta: dict[str, dict[str, Any]] = {}
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if va != vb:
            delta[key] = {"a": va, "b": vb}
    return delta


def diff_scenarios(a: SavedScenario, b: SavedScenario) -> dict:
    """A structured diff between two saved scenarios: per-step ``op``/``params`` deltas, plus whether
    ``result_ref`` changed -- a changed hash means a materially different outcome, an identical hash
    means the two scenarios provably produced the same result."""
    steps_a = (a.scenario or {}).get("steps") or []
    steps_b = (b.scenario or {}).get("steps") or []

    step_diffs: list[dict[str, Any]] = []
    for i in range(max(len(steps_a), len(steps_b))):
        sa = steps_a[i] if i < len(steps_a) else None
        sb = steps_b[i] if i < len(steps_b) else None
        if sa == sb:
            continue
        op_a = (sa or {}).get("op")
        op_b = (sb or {}).get("op")
        params_a = (sa or {}).get("params") or {}
        params_b = (sb or {}).get("params") or {}
        step_diffs.append(
            {
                "index": i,
                "op": {"a": op_a, "b": op_b, "changed": op_a != op_b},
                "params": _param_delta(params_a, params_b),
            }
        )

    result_changed = a.result_ref != b.result_ref
    return {
        "name_a": a.name,
        "name_b": b.name,
        "steps": step_diffs,
        "result_ref": {
            "a": a.result_ref,
            "b": b.result_ref,
            "changed": result_changed,
            "provenance": {"a": a.provenance, "b": b.provenance},
        },
        "materially_different": result_changed,
    }


def _posterior_ref_dict(saved: SavedScenario) -> dict[str, Any]:
    """Shape ``saved`` as the IC-2-header-like dict ``decision_receipt`` accepts for its
    ``posterior_ref`` argument, with ``content_hash`` pinned to the exact ``result_ref`` this scenario
    already carries -- so the receipt's posterior-stage hash is that same value, not a re-derived one."""
    meta: dict[str, Any] = dict(saved.provenance.get("result_provenance") or {})
    meta["content_hash"] = saved.result_ref
    meta.setdefault("artifact_ref", saved.result_ref)
    meta.setdefault("schema", saved.provenance.get("schema", SCENARIO_STORE_SCHEMA))
    return meta


def scenario_memo(baseline: SavedScenario, alternative: SavedScenario) -> dict:
    """Assemble a decision memo comparing ``baseline`` to ``alternative`` as an IC-5 ``TraceRecord``:
    ``steps`` are the two scenarios' forwards (each carrying its ``content_hash``) followed by the E7
    ``decision_receipt`` chain, ``outcome`` is the ``diff_scenarios`` result, and ``provenance`` carries
    the full ``decision_receipt`` (data -> posterior -> claim -> decision) lineage -- so every number in
    the memo resolves to a hashed artifact. Validated with ``validate_trace_record`` before returning.
    """
    diff = diff_scenarios(baseline, alternative)

    receipt = decision_receipt(
        dataset_ref=baseline.result_ref,
        posterior_ref=_posterior_ref_dict(alternative),
        claim={"baseline": baseline.name, "alternative": alternative.name, "diff": diff},
        decision={
            "recommendation": "alternative" if diff["materially_different"] else "equivalent",
            "baseline": baseline.name,
            "alternative": alternative.name,
        },
        substrate=Substrate(),
    )

    steps = [
        {
            "tool": "scenario_forward",
            "args": {"name": baseline.name, "scenario": baseline.scenario},
            "result": {"content_hash": baseline.result_ref},
            "model": None,
            "verdict": None,
        },
        {
            "tool": "scenario_forward",
            "args": {"name": alternative.name, "scenario": alternative.scenario},
            "result": {"content_hash": alternative.result_ref},
            "model": None,
            "verdict": None,
        },
        *receipt["steps"],
    ]

    memo: dict[str, Any] = {
        "prompt": f"scenario comparison memo: {baseline.name!r} vs {alternative.name!r}",
        "steps": steps,
        "outcome": diff,
        "provenance": {
            "decision_receipt": receipt,
            "lineage": receipt["provenance"]["lineage"],
        },
    }
    validate_trace_record(memo)
    return memo
