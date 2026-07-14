"""E7 -- cross-chain provenance receipt (mixle-mlops integration slice).

E7's Public API (`decision_receipt`) belongs to mixle-pde's `reasoning` module, and the IC-5 frozen
envelope validator (`validate_trace_record`) belongs to core mixle's `mixle.task.trace_record` --
both are the primary-repo half of this task, tracked in a separate PR against those repos, and are not
reimplemented here.

What mixle-mlops actually owns for E7 is the DoD test: proof that a decision receipt built the way the
work order describes -- data -> inversion -> interpretation -> decision, each edge stamped with a
sha256 lineage hash -- round-trips through the two pieces of shared infrastructure the algorithm calls
out for reuse (`mixle.substrate.ingest.ingest_artifacts` and
`mixle.inference.production.provenance.build_header`), and satisfies IC-5's frozen envelope shape.

Until `mixle_pde.reasoning.decision_receipt` and `mixle.task.trace_record` land upstream, this module
builds the chain with a small local stand-in (`_decision_receipt`) so the substrate-ingest wiring and
hash-lineage checking are exercised against real data today instead of being skipped. `_decision_receipt`
follows the frozen algorithm exactly (IC-5 envelope, per-edge `provenance.content_hash`) and is not a
public API of this package -- swap it for `from mixle_pde.reasoning import decision_receipt` once that
lands.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mixle.data.hashing import dataset_hash
from mixle.inference.production.provenance import build_header
from mixle.substrate.core import Substrate
from mixle.substrate.ingest import ingest_artifacts

try:  # IC-5's frozen validator (mixle.task.trace_record) -- not yet landed in core mixle (Wave-0 gap
    # relative to this task's dependencies); fall back to a byte-identical copy of the frozen reference
    # body from notes/exec/contracts.md so this test enforces the real rule either way.
    from mixle.task.trace_record import STEP_KEYS, TRACE_KEYS, validate_trace_record
except ImportError:  # pragma: no cover - exercised until IC-5 lands
    TRACE_KEYS = ("prompt", "steps", "outcome", "provenance")
    STEP_KEYS = ("tool", "args", "result", "model", "verdict")

    def validate_trace_record(d: dict[str, Any]) -> None:
        missing = [k for k in TRACE_KEYS if k not in d]
        if missing:
            raise ValueError(f"trace record missing frozen keys: {missing}")
        for i, s in enumerate(d.get("steps") or []):
            for k in ("tool", "args", "result"):
                if k not in s:
                    raise ValueError(f"step {i} missing frozen key {k!r}")


def _sha256_of_arrays(arrays: dict[str, Any]) -> str:
    """IC-2's frozen hashing rule (`mixle_pde.io.artifacts.sha256_of_arrays`), copied verbatim: E2 has
    not landed `mixle_pde/io/artifacts.py` yet, so there is nothing to import from -- this reproduces the
    exact frozen body rather than inventing a different hashing rule."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode("utf-8"))
        h.update(memoryview(arrays[k]).tobytes() if hasattr(arrays[k], "tobytes") else bytes(arrays[k]))
    return h.hexdigest()


def _write_posterior_artifact(
    root: Path, *, name: str, mean: np.ndarray, cov_diag: np.ndarray, parent_content_hash: str
) -> tuple[Path, str]:
    """Write ``{name}/manifest.json`` (+ a sibling ``.npz``) in the shape E2's ``save_posterior`` will
    write: IC-2 header keys, plus a ``provenance.content_hash`` back-pointer to the dataset that fed the
    inversion (Algorithm step 1: "attach the IC-2 content_hash ... to its provenance")."""
    adir = root / name
    adir.mkdir(parents=True, exist_ok=True)
    arrays = {"mean": mean, "cov_diag": cov_diag}
    np.savez(adir / "posterior.npz", **arrays)
    content_hash = _sha256_of_arrays(arrays)
    manifest = {
        "mixle_artifact": "field_posterior",
        "schema": "mixle_pde.field_posterior/v1",
        "content_hash": content_hash,
        "grid": {"shape": list(mean.shape), "origin": [0.0, 0.0, 0.0], "spacing": 1.0},
        "crs": "EPSG:32611",
        "units": "kg/m3",
        "provenance": {"content_hash": parent_content_hash, "stage": "inversion"},
        "created": time.time(),
        "meta": {"task": "gravity_inversion"},
    }
    (adir / "manifest.json").write_text(json.dumps(manifest))
    return adir, content_hash


def _item_hash(ref: str, manifest: dict[str, Any]) -> str:
    """An IC-13-style enclosing item hash over metadata+ref (work-plan M1a): sha256 of the artifact
    ref path plus the canonicalised manifest, so the substrate item's identity is independent of --
    but re-derivable alongside -- the IC-2 array digest it wraps."""
    blob = ref + json.dumps(manifest, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class _DummyModel:
    """A model stand-in just concrete enough for `build_header` to introspect (it degrades every
    field it cannot compute to ``None`` rather than raising)."""


def _decision_receipt(
    *, dataset_ref: dict, posterior_ref: dict, claim: dict, decision: dict, substrate: Substrate
) -> dict:
    """Local stand-in for E7's Public API (`decision_receipt`, mixle-pde `reasoning` -- see module
    docstring). Walks data -> posterior -> claim -> decision, stamping every edge's
    ``provenance.content_hash`` with its parent's hash (Algorithm step 3), and returns the frozen IC-5
    envelope (Algorithm step 5): every number in ``decision`` traces to a step whose ``content_hash`` is
    recorded and re-derivable.
    """
    steps: list[dict[str, Any]] = [
        {
            "tool": "ingest_dataset",
            "args": {"dataset_ref": dataset_ref["name"]},
            "result": {"content_hash": dataset_ref["content_hash"], "n_records": dataset_ref["n_records"]},
            "model": None,
            "verdict": None,
        },
        {
            "tool": "run_inversion",
            "args": {"dataset_ref": dataset_ref["content_hash"], "modality": "gravity", "prior": "smooth"},
            "result": {
                "posterior_ref": posterior_ref["item_id"],
                "content_hash": posterior_ref["content_hash"],
                "item_hash": posterior_ref["item_hash"],
            },
            "model": None,
            "verdict": None,
            "provenance": {"content_hash": dataset_ref["content_hash"]},
        },
        {
            "tool": "query_posterior",
            "args": {"posterior_ref": posterior_ref["item_id"], "query": "region_mass"},
            "result": {"value": claim["value"], "content_hash": claim["content_hash"]},
            "model": claim["model"],
            "verdict": None,
            "provenance": {"content_hash": posterior_ref["content_hash"]},
        },
        {
            "tool": "decide",
            "args": {"claim_ref": claim["content_hash"]},
            "result": dict(decision, content_hash=decision["content_hash"]),
            "model": decision["model"],
            "verdict": decision.get("verdict"),
            "provenance": {"content_hash": claim["content_hash"]},
        },
    ]
    lineage = [
        {"stage": "data", "content_hash": dataset_ref["content_hash"]},
        {"stage": "inversion", "content_hash": posterior_ref["content_hash"], "parent": dataset_ref["content_hash"]},
        {"stage": "interpretation", "content_hash": claim["content_hash"], "parent": posterior_ref["content_hash"]},
        {"stage": "decision", "content_hash": decision["content_hash"], "parent": claim["content_hash"]},
    ]
    return {
        "prompt": f"drill decision for {dataset_ref['name']}",
        "steps": steps,
        "outcome": {k: v for k, v in decision.items() if k not in ("model", "verdict")},
        "provenance": {"lineage": lineage, "content_hash": decision["content_hash"]},
    }


def test_e7_provenance_chain(tmp_path):
    substrate = Substrate()

    # 1) data -- a synthetic gravity-survey dataset, fingerprinted with the existing (already-shipped)
    # dataset hashing utility `build_header` itself relies on.
    observations = [1.02, 0.98, 1.10, 0.87, 1.15, 1.01]
    data_hash = dataset_hash(observations)
    dataset_ref = {"name": "synthetic_gravity_survey", "content_hash": data_hash, "n_records": len(observations)}

    # 2) inversion -- fit a (synthetic) posterior, attach the dataset's content_hash to its provenance,
    # serialise it as an artifact, then `ingest_artifacts` it into the substrate (Algorithm steps 1-2):
    # lineage + retrieval work without copying arrays -- the arrays live behind the ref.
    mean = np.array([2670.0, 2680.0, 2705.0])
    cov_diag = np.array([12.0, 9.0, 15.0])
    registry_root = tmp_path / "registry"
    adir, posterior_hash = _write_posterior_artifact(
        registry_root, name="posterior_001", mean=mean, cov_diag=cov_diag, parent_content_hash=data_hash
    )
    manifest = json.loads((adir / "manifest.json").read_text())
    expected_item_hash = _item_hash(str(adir), manifest)

    ids = ingest_artifacts(substrate, str(registry_root))
    assert len(ids) == 1
    item = substrate.get(ids[0])
    assert item is not None
    assert item.payload["manifest"]["content_hash"] == posterior_hash  # IC-2 digest preserved as the artifact digest
    assert item.payload["ref"] == str(adir)
    assert (
        _item_hash(item.payload["ref"], item.payload["manifest"]) == expected_item_hash
    )  # M1a item hash re-derivable

    posterior_ref = {"item_id": ids[0], "content_hash": posterior_hash, "item_hash": expected_item_hash}

    # 3) interpretation -- read a decision quantity off the posterior and fingerprint it with the
    # existing (already-shipped) `build_header`, reused exactly as the work order calls for.
    region_mass = float(np.sum(mean) * 1e6)
    header = build_header(
        _DummyModel(),
        [region_mass],
        training={"query": "region_mass", "parent_content_hash": posterior_hash},
    )
    claim = {
        "value": region_mass,
        "content_hash": header.dataset_hash,
        "model": "physics-tools/query_posterior@v1",
    }
    assert header.training["parent_content_hash"] == posterior_hash

    # 4) decision -- a downstream call that consumes the claim; its own content_hash is derived from the
    # decision payload plus the parent claim hash, closing the data -> decision chain.
    decision_payload = {"drill": True, "expected_value": region_mass * 0.02, "risk": 0.18}
    decision_hash = dataset_hash(
        [decision_payload["expected_value"], decision_payload["risk"], claim["content_hash"]]
    )
    decision = dict(
        decision_payload,
        content_hash=decision_hash,
        model="drill-advisor/v1",
        verdict={"passed": True, "score": 0.9, "reasons": ["expected value positive"], "kind": "physical"},
    )

    receipt = _decision_receipt(
        dataset_ref=dataset_ref, posterior_ref=posterior_ref, claim=claim, decision=decision, substrate=substrate
    )

    # The frozen IC-5 envelope.
    validate_trace_record(receipt)
    assert set(receipt) >= set(TRACE_KEYS)
    for step in receipt["steps"]:
        assert set(STEP_KEYS) <= set(step)

    # Every scalar in the returned memo (the decision outcome) resolves to a hashed lineage edge, and
    # every recorded content_hash is re-derivable from the data that produced it -- not just asserted.
    lineage_by_stage = {edge["stage"]: edge for edge in receipt["provenance"]["lineage"]}
    assert {"data", "inversion", "interpretation", "decision"} <= set(lineage_by_stage)

    for scalar_key in ("expected_value", "risk", "drill"):
        assert scalar_key in receipt["outcome"]
    decision_edge = lineage_by_stage["decision"]
    assert decision_edge["content_hash"] == decision_hash
    assert decision_edge["parent"] == claim["content_hash"]

    # Re-derive every hash from scratch (idempotent: the same inputs produce the same digest).
    assert dataset_hash(observations) == data_hash
    assert _sha256_of_arrays({"mean": mean, "cov_diag": cov_diag}) == posterior_hash
    assert build_header(_DummyModel(), [region_mass]).dataset_hash == claim["content_hash"]
    assert (
        dataset_hash([decision_payload["expected_value"], decision_payload["risk"], claim["content_hash"]])
        == decision_hash
    )

    # And the chain is genuinely linked edge-to-edge: each stage's provenance names its parent's hash.
    assert lineage_by_stage["inversion"]["parent"] == data_hash
    assert lineage_by_stage["interpretation"]["parent"] == posterior_hash
    assert lineage_by_stage["decision"]["parent"] == claim["content_hash"]
