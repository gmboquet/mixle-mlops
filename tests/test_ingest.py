"""I7 — extraction verification + typed provenanced observations (the folded E9, the hard gate).

Uses a small monkeypatched IC-6 `Verifier` (a plain bounds check) rather than the real
`PhysicalVerifier`/`CalibrationVerifier` (E10): the point of I7's ingest gate is the orchestration
around verification — reject-nothing-written, accept-emits-typed-items, missing-value-opens-a-gap —
which must hold regardless of which verifier is plugged in, and it must hold even if E10's physics
verifier were not implemented yet.
"""

from __future__ import annotations

import copy

import pytest

from mixle_mlops.multimodal.ingest import TYPED_TABLE_SCHEMA, ingest_extraction
from mixle_mlops.verification.base import Verdict


class _BoundsVerifier:
    """Minimal IC-6 `Verifier`: `claim['value']` must fall within `context['bounds'] = (lo, hi)`."""

    def verify(self, claim: dict, context: dict) -> Verdict:
        lo, hi = context.get("bounds", (0.0, 100.0))
        value = claim.get("value")
        if value is None or not (lo <= value <= hi):
            return Verdict(
                passed=False,
                score=0.0,
                kind="physical",
                reasons=[f"value {value!r} out of bounds [{lo}, {hi}]"],
            )
        return Verdict(passed=True, score=1.0, kind="physical", reasons=[])


def _source_item() -> dict:
    from mixle_knowledge.contracts import KnowledgeItem

    return KnowledgeItem(
        id="table-src-1",
        kind="table",
        modality="table",
        schema_uri=TYPED_TABLE_SCHEMA,
        content_hash="1" * 64,
        payload={
            "primary_key": ["sample_id"],
            "columns": [
                {"name": "sample_id", "type": "string"},
                {"name": "cu_pct", "type": "float", "unit": "%"},
            ],
            "rows": [{"sample_id": "A-1", "cu_pct": 1.8}],
        },
    ).model_dump(mode="json")


def test_out_of_range_grade_rejected():
    claim = {
        "kind": "assay",
        "field": "cu_pct",
        "value": 142.0,  # an obviously impossible copper grade
        "unit": "%",
        "location": (500500.0, 4100500.0, -120.0),
        "crs": "EPSG:32611",
        "modality": "assay",
        "model": "vlm-extract-v3",
    }
    sink_calls: list = []

    result = ingest_extraction(
        claim,
        verifier=_BoundsVerifier(),
        context={"bounds": (0.0, 30.0)},
        sink=lambda obs, item: sink_calls.append((obs, item)) or "unused-id",
    )

    assert result.accepted is False
    assert result.observation is None
    assert result.knowledge_item is None
    assert result.verdict.passed is False
    assert result.provenance["status"] == "rejected"
    assert any("out of bounds" in r for r in result.verdict.reasons)
    assert sink_calls == []  # an impossible value writes nothing


def test_structure_preserved():
    from mixle_knowledge.contracts import KnowledgeItem

    source_item = _source_item()
    source_item_before = copy.deepcopy(source_item)

    claim = {
        "kind": "assay",
        "field": "cu_pct",
        "value": 1.8,
        "unit": "%",
        "location": (500500.0, 4100500.0, -120.0),
        "crs": "EPSG:32611",
        "modality": "assay",
        "noise_cov": 0.05,
        "model": "vlm-extract-v3",
    }
    sink_calls: list = []

    def _sink(obs, item):
        sink_calls.append((obs, item))
        return item["id"]

    result = ingest_extraction(
        claim,
        verifier=_BoundsVerifier(),
        context={"bounds": (0.0, 30.0)},
        sink=_sink,
        source_item=source_item,
    )

    assert result.accepted is True
    assert result.verdict.passed is True

    # --- IC-4 Observation: typed, georeferenced, carries the extracting model in provenance ---
    obs = result.observation
    assert obs.kind == "assay"
    assert obs.crs == "EPSG:32611"
    assert obs.modality == "assay"
    assert obs.units == "%"
    assert obs.value[0] == pytest.approx(1.8)
    assert obs.provenance["model"] == "vlm-extract-v3"

    # --- IC-13 KnowledgeItem: typed payload, canonical hash, verifier+model provenance, relation ---
    item = result.knowledge_item
    assert item["schema_uri"] == TYPED_TABLE_SCHEMA
    assert len(item["content_hash"]) == 64
    assert item["payload"]["rows"][0]["value"] == pytest.approx(1.8)
    relation = item["relations"][0]
    assert relation["predicate"] == "derived_from"
    assert relation["target_id"] == "table-src-1"
    assert any(p["uri"] == "model://vlm-extract-v3" for p in item["provenance"])
    assert any(p["uri"] == "verifier://physical" for p in item["provenance"])

    # --- deep round-trip: replaying the emitted item through the real pydantic model is a no-op ---
    reloaded = KnowledgeItem.model_validate(item)
    assert reloaded.model_dump(mode="json") == item

    # --- sink received both halves atomically; its returned id lands back in provenance ---
    assert sink_calls and sink_calls[0][0] is obs and sink_calls[0][1] is item
    assert result.provenance["sink_id"] == item["id"]

    # --- the source map/table/image item deep-round-trips completely unchanged ---
    assert source_item == source_item_before

    # --- a missing assay produces an OPEN typed gap rather than a fabricated value ---
    missing_claim = {"kind": "assay", "field": "au_ppm", "value": None, "model": "vlm-extract-v3"}
    gap_result = ingest_extraction(
        missing_claim,
        verifier=_BoundsVerifier(),
        context={"bounds": (0.0, 30.0)},
        source_item=source_item,
    )
    assert gap_result.accepted is False
    assert gap_result.observation is None
    assert gap_result.knowledge_item is None
    gap = gap_result.provenance["gap"]
    assert gap["status"] == "open"
    assert "au_ppm" in gap["question"]
    assert gap["required_schema"]["field"] == "au_ppm"
