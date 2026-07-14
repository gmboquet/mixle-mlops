"""I7 — extraction verification + typed, provenanced observations (the folded E9; work-plan §5).

``ingest_extraction`` is the single choke point between a frontier-model extraction claim (a
digitized map polygon, a log-curve read, an assay value, ...) and the physics layer: nothing becomes
an IC-4 :class:`~mixle_pde.observations.Observation` or an IC-13 ``KnowledgeItem`` without first
passing an IC-6 :class:`~mixle_mlops.verification.base.Verifier`. A rejected claim writes nothing —
no half-typed observation, no knowledge item — only a ``Verdict`` and its reasons, so a caller (M2a)
can log the failed attempt instead of silently laundering an implausible number into the substrate.
A claim that names a field with no value never invents one: it is turned into an OPEN IC-13
``KnowledgeGap`` instead, carrying the schema the missing evidence would need to satisfy.

Repo-boundary note: ``mixle_pde.observations.Observation`` (IC-4) and ``mixle_knowledge.contracts``
(IC-13) are both imported lazily, inside the functions that actually need them, never at module import
time — mirroring the existing convention in ``mixle_mlops/models/field_posterior.py`` and the IC-13
bridge in ``mixle_mlops/gateway/tool_registry.py``. Neither package is a hard dependency of
``mixle-mlops`` today, so a deployment without them still imports this module; only calling
``ingest_extraction`` on an accepted/gap claim needs them installed.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..verification.base import Verdict, Verifier

# Mirrors IC-13's frozen `mixle://schema/typed-table/1` schema uri (mixle-knowledge). A single typed
# observation/annotation is modeled as a one-row typed table — the same "record" mapping the M0b
# bridge in `gateway/tool_registry.py` already uses — so the emitted item gets real IC-13 structural
# validation (declared columns/types/primary key) rather than an unvalidated free-form payload.
TYPED_TABLE_SCHEMA = "mixle://schema/typed-table/1"

__all__ = ["IngestResult", "ingest_extraction"]


@dataclass
class IngestResult:
    """The outcome of one :func:`ingest_extraction` call.

    ``knowledge_item`` is the accepted claim's IC-13 item (a plain ``dict``, JSON round-trippable) or
    ``None`` when the claim was rejected or turned into a gap. ``provenance`` always carries
    ``status`` (``"accepted"`` | ``"rejected"`` | ``"gap"``) plus enough detail (reasons, verifier
    kind/score, the gap payload when relevant) for a caller to log a failed attempt (M2a) without
    re-deriving it from the verdict.
    """

    accepted: bool
    observation: Any | None
    knowledge_item: dict[str, Any] | None
    verdict: Verdict
    provenance: dict[str, Any] = field(default_factory=dict)


def ingest_extraction(
    claim: dict[str, Any],
    *,
    verifier: Verifier,
    context: dict[str, Any],
    sink: Callable[[Any, dict[str, Any]], str] | None = None,
    source_item: dict[str, Any] | None = None,
) -> IngestResult:
    """Verify ``claim`` against ``context``; on pass, mint a typed IC-4 ``Observation`` + IC-13
    ``KnowledgeItem`` pair and hand both to ``sink`` atomically.

    ``claim`` carries the extracted value plus enough shape to build an ``Observation``: ``value``,
    optionally ``location`` (an ``(x, y, z)`` triple, default mesh-origin), ``noise_cov`` (scalar or
    array, default ``1.0``), ``kind``/``field``, ``unit``/``units``, ``time``, ``crs``, ``modality``,
    and ``model`` (the extracting model's name, folded into provenance). ``source_item`` is the IC-13
    map/table/image/time-series artifact (a plain dict) this claim was extracted from; when given, the
    emitted item's ``derived_from`` relation targets ``source_item["id"]`` and ``source_item`` itself
    is only ever read, never mutated — the caller's copy deep-round-trips unchanged.

    If ``claim["value"]`` is missing (``None``), no verifier runs and no value is fabricated: the
    result is an OPEN IC-13 ``KnowledgeGap`` (in ``provenance["gap"]``) describing what evidence is
    still needed, with ``accepted=False`` and both ``observation``/``knowledge_item`` left ``None``.
    """
    field_name = str(claim.get("field") or claim.get("kind") or claim.get("modality") or "value")

    if claim.get("value") is None:
        return _gap_result(claim, field_name, source_item)

    verify_context = dict(context or {})
    if source_item is not None:
        verify_context.setdefault("source_item", source_item)
    verdict = verifier.verify(claim, verify_context)

    if not verdict.passed:
        return IngestResult(
            accepted=False,
            observation=None,
            knowledge_item=None,
            verdict=verdict,
            provenance={
                "status": "rejected",
                "field": field_name,
                "verifier_kind": verdict.kind,
                "score": verdict.score,
                "reasons": list(verdict.reasons),
            },
        )

    observation = _build_observation(claim, field_name)
    item_id, knowledge_item = _build_knowledge_item(claim, field_name, verdict, source_item)

    provenance = {
        "status": "accepted",
        "field": field_name,
        "verifier_kind": verdict.kind,
        "score": verdict.score,
        "model": claim.get("model"),
        "item_id": item_id,
        "source_item_id": (source_item or {}).get("id"),
    }
    result = IngestResult(
        accepted=True,
        observation=observation,
        knowledge_item=knowledge_item,
        verdict=verdict,
        provenance=provenance,
    )
    if sink is not None:
        result.provenance["sink_id"] = sink(observation, knowledge_item)
    return result


def _gap_result(claim: dict[str, Any], field_name: str, source_item: dict[str, Any] | None) -> IngestResult:
    """A missing value never gets fabricated: emit an OPEN IC-13 `KnowledgeGap` instead (algorithm step 6)."""
    try:
        from mixle_knowledge.contracts import KnowledgeGap, KnowledgeGapStatus
    except ImportError as exc:
        raise ImportError(
            "ingest_extraction(...) needs mixle_knowledge's structured-exchange contracts "
            "(mixle_knowledge.contracts.KnowledgeGap, IC-13); install/land the mixle-knowledge package."
        ) from exc

    gap = KnowledgeGap(
        id=f"gap-{uuid.uuid4().hex}",
        question=str(claim.get("question") or f"Find the missing {field_name} value"),
        required_schema=dict(claim.get("required_schema") or {"type": "number", "field": field_name}),
        acceptance_criteria=list(claim.get("acceptance_criteria") or ["verified extraction"]),
        status=KnowledgeGapStatus.OPEN,
    )
    verdict = Verdict(
        passed=False,
        score=0.0,
        kind="exact",
        reasons=[f"missing required field {field_name!r}; no value fabricated"],
    )
    return IngestResult(
        accepted=False,
        observation=None,
        knowledge_item=None,
        verdict=verdict,
        provenance={
            "status": "gap",
            "field": field_name,
            "gap": gap.model_dump(mode="json"),
            "source_item_id": (source_item or {}).get("id"),
        },
    )


def _build_observation(claim: dict[str, Any], field_name: str) -> Any:
    """Lift a passing ``claim`` into an IC-4 ``Observation`` (`mixle_pde.observations`, lazy import)."""
    try:
        from mixle_pde.observations import Observation
    except ImportError as exc:
        raise ImportError(
            "ingest_extraction(...) needs mixle_pde's typed-observation module "
            "(mixle_pde.observations.Observation, IC-4); install/land the mixle-pde package."
        ) from exc

    provenance = dict(claim.get("provenance") or {})
    provenance.setdefault("model", claim.get("model"))
    provenance.setdefault("field", field_name)

    noise_cov = np.atleast_1d(np.asarray(claim.get("noise_cov", claim.get("uncertainty", 1.0)), dtype=float))
    return Observation(
        kind=str(claim.get("kind") or field_name),
        location=claim.get("location", (0.0, 0.0, 0.0)),
        value=claim["value"],
        noise_cov=noise_cov,
        time=claim.get("time"),
        units=str(claim.get("unit") or claim.get("units") or ""),
        provenance=provenance,
        crs=claim.get("crs"),
        modality=str(claim.get("modality") or ""),
    )


def _build_knowledge_item(
    claim: dict[str, Any],
    field_name: str,
    verdict: Verdict,
    source_item: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Build the IC-13 typed-table `KnowledgeItem` (as a plain dict) for one accepted claim."""
    try:
        from mixle_knowledge.contracts import (
            KnowledgeItem,
            KnowledgeRelation,
            Modality,
            ResourceKind,
            SourceRef,
        )
    except ImportError as exc:
        raise ImportError(
            "ingest_extraction(...) needs mixle_knowledge's structured-exchange contracts "
            "(mixle_knowledge.contracts.KnowledgeItem, IC-13); install/land the mixle-knowledge package."
        ) from exc

    location = list(claim.get("location") or (0.0, 0.0, 0.0))
    x, y = float(location[0]), float(location[1])
    z = float(location[2]) if len(location) > 2 and location[2] is not None else None
    value = float(claim["value"])
    unit = claim.get("unit") or claim.get("units") or None
    crs = claim.get("crs")
    modality = str(claim.get("modality") or "") or None
    time_value = claim.get("time")

    item_id = f"obs-{uuid.uuid4().hex}"
    payload = {
        "primary_key": ["item_id"],
        "columns": [
            {"name": "item_id", "type": "string", "nullable": False},
            {"name": "field", "type": "string", "nullable": False},
            {"name": "value", "type": "float", "unit": unit, "nullable": False},
            {"name": "x", "type": "float", "nullable": False},
            {"name": "y", "type": "float", "nullable": False},
            {"name": "z", "type": "float", "nullable": True},
            {"name": "crs", "type": "string", "nullable": True},
            {"name": "modality", "type": "string", "nullable": True},
            {"name": "time", "type": "float", "nullable": True},
        ],
        "rows": [
            {
                "item_id": item_id,
                "field": field_name,
                "value": value,
                "x": x,
                "y": y,
                "z": z,
                "crs": crs,
                "modality": modality,
                "time": float(time_value) if time_value is not None else None,
            }
        ],
    }
    metadata: dict[str, Any] = {
        "verifier_kind": verdict.kind,
        "verifier_score": verdict.score,
        "field": field_name,
    }
    if source_item is not None and source_item.get("id") is not None:
        metadata["source_item_id"] = source_item["id"]

    content_hash = _content_hash(TYPED_TABLE_SCHEMA, "1.0.0", payload, None, metadata)

    provenance: list[Any] = []
    model_name = claim.get("model")
    if model_name:
        provenance.append(SourceRef(uri=f"model://{model_name}"))
    provenance.append(SourceRef(uri=f"verifier://{verdict.kind}"))

    relations: list[Any] = []
    if source_item is not None and source_item.get("id") is not None:
        relations.append(KnowledgeRelation(predicate="derived_from", target_id=str(source_item["id"])))

    item = KnowledgeItem(
        id=item_id,
        kind=ResourceKind.TABLE,
        modality=Modality.TABLE,
        schema_uri=TYPED_TABLE_SCHEMA,
        content_hash=content_hash,
        payload=payload,
        text_surface=f"{field_name} = {value:g}" + (f" {unit}" if unit else ""),
        provenance=provenance,
        relations=relations,
        uncertainty={"noise_cov": claim.get("noise_cov"), "verifier_score": verdict.score},
        metadata=metadata,
    )
    return item_id, item.model_dump(mode="json")


def _content_hash(
    schema_uri: str,
    schema_version: str,
    payload: Any,
    artifact_ref: str | None,
    metadata: dict[str, Any],
) -> str:
    """IC-13's frozen hash rule: sha256 of the canonical JSON of {schema_uri, schema_version, payload,
    artifact_ref, metadata} — the same rule `KnowledgeItem.content_hash` documents at the contract's
    foot, applied here rather than through `mixle_pde.io.artifacts` (IC-2's own `content_hash`/
    `sha256_of_arrays` target that module's *array* artifacts, not a JSON claim payload, and — per
    E7's own repo-boundary note — that module has not landed in `mixle-pde` as of this PR)."""
    canonical = json.dumps(
        {
            "schema_uri": schema_uri,
            "schema_version": schema_version,
            "payload": payload,
            "artifact_ref": artifact_ref,
            "metadata": metadata,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
