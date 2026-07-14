"""M7 -- the self-improvement loop: one usage -> retrain -> verify -> promote cycle for a hosted model
(work-plan M7, contracts IC-5/IC-6/IC-10/IC-13).

``improve_once`` is the single orchestration entry point the autonomous 0.8.0 compounding loop drives:

1. It mines usage recorded against a model -- an IC-5 trace envelope, optionally carrying the IC-13
   bundle/delta that usage produced or resolved.
2. Only *verified* transitions on a *policy-allowed* canonical ref are mined (the M4 dataset foundry's
   job, once it lands; this module tries the real foundry first and falls back to its own miner).
3. The mined observations are handed to the existing verify-gated ``evolve.worker.EvolutionWorker`` to
   retrain and provisionally promote a challenger -- M5's retrain surface, once it lands; until then this
   reuses ``EvolutionWorker`` (which already wraps ``mixle.evolve.improve``), per the M7 non-goal of adding
   no new promotion infrastructure.
4. A provisional promotion is re-checked by an injected IC-6 ``Verifier`` (the challenger's
   physical/calibration claim) and by validating every mined IC-13 delta's identity shape.
5. It is re-checked again by an injected M8 harness: end-to-end task success must not drop, and the
   compounding-loop safety checks (structure fidelity, gap/conflict safety, access isolation) must all
   still hold. A reported leak, an unexpected content-hash mutation, or a silent overwrite hard-blocks the
   promotion exactly like a harness regression does.
6. Either gate failing rolls the promotion back via the worker's existing ``rollback``, and the run's
   model + mined-dataset size + bundle/delta lineage is folded into the existing ``EvolutionRecord``
   lineage trail (``evolve.lineage.record_run``) regardless of outcome.

M4 (dataset foundry), M5 (retrain) and M8 (the harness itself) are separate, still-evolving packages; this
module never hard-imports them at module scope. ``verifier`` and ``harness`` are structurally-typed
(duck-typed) injected collaborators -- IC-6's ``Verifier.verify(claim, context) -> Verdict`` and a harness
exposing ``evaluate(model_id) -> HarnessResult`` -- so this loop is fully exercisable today with fakes, and
picks up the real foundry/retrain/harness the moment those packages exist, with zero code change here.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence, runtime_checkable

from sqlmodel import Field, Session, SQLModel, select

from ..core.registry import ModelRegistry
from .lineage import record_run
from .policy import EvolutionPolicy
from .worker import EvolutionRun, EvolutionWorker

# IC-5's frozen envelope keys, mirrored here so this module has no hard import on
# ``mixle.task.trace_record`` -- the real validator is used automatically once that package lands.
TRACE_KEYS = ("prompt", "steps", "outcome", "provenance")

# IC-13's frozen delta identity fields (the subset required on every ``KnowledgeDelta``), mirrored the
# same way against ``mixle_knowledge.contracts.KnowledgeDelta``.
DELTA_REQUIRED_KEYS = ("base_bundle_id", "base_revision", "produced_by")

__all__ = [
    "TRACE_KEYS",
    "DELTA_REQUIRED_KEYS",
    "Verifier",
    "Harness",
    "HarnessResult",
    "UsageTraceRecord",
    "record_usage",
    "improve_once",
]


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_trace(trace: dict) -> None:
    """Enforce the IC-5 envelope shape; defers to the real validator once ``mixle.task.trace_record`` lands."""
    try:
        from mixle.task.trace_record import validate_trace_record
    except ImportError:
        missing = [k for k in TRACE_KEYS if k not in trace]
        if missing:
            raise ValueError(f"usage trace missing frozen IC-5 keys: {missing}")
    else:
        validate_trace_record(trace)


def _validate_delta(delta: dict) -> None:
    """Enforce the IC-13 delta identity shape; defers to the real pydantic model once importable."""
    try:
        from mixle_knowledge.contracts import KnowledgeDelta
    except ImportError:
        missing = [k for k in DELTA_REQUIRED_KEYS if k not in delta]
        if missing:
            raise ValueError(f"usage delta missing frozen IC-13 keys: {missing}")
    else:
        KnowledgeDelta.model_validate(delta)


@runtime_checkable
class Verifier(Protocol):
    """Structural mirror of IC-6 -- ``mixle_mlops.verification.base.Verifier`` once that package lands."""

    def verify(self, claim: dict, context: dict) -> Any: ...  # returns an IC-6 Verdict (passed, score, kind, ...)


@dataclass
class HarnessResult:
    """One M8 evaluation of a served model: end-to-end task success plus the compounding-loop safety gates
    the M7 algorithm names explicitly -- structure fidelity, gap/conflict safety, access isolation -- and the
    three hard-block conditions (leak, unexpected hash mutation, silent overwrite)."""

    success: float
    regressed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    leak_detected: bool = False
    hash_mutation_detected: bool = False
    silent_overwrite_detected: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Harness(Protocol):
    """Structural mirror of the M8 meta-skill harness this loop's promotion is gated on."""

    def evaluate(self, model_id: str) -> HarnessResult: ...


class UsageTraceRecord(SQLModel, table=True):
    """One recorded usage transition for a hosted model: an IC-5 trace, plus the IC-13 bundle/delta it
    produced or resolved, if any. ``improve_once`` mines only ``verified=True`` rows (a verified transition)
    on an allowed ``canonical_ref`` (an IC-10 catalog entry id) -- the M7 algorithm's "policy-allowed
    canonical refs and verified transitions" mining rule."""

    __tablename__ = "evolution_usage_trace"

    id: str = Field(default_factory=_uuid, primary_key=True)
    model_id: str = Field(index=True)
    trace_json: str
    outcome_value: float | None = None
    bundle_id: str | None = Field(default=None, index=True)
    delta_json: str | None = None
    canonical_ref: str | None = Field(default=None, index=True)
    verified: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=_now, index=True)


def _ensure_schema(session: Session) -> None:
    """Create ``UsageTraceRecord``'s table if missing. Self-contained: this module is new and isn't on
    ``storage.db.init_db``'s table-import list, so it creates its own table on first use instead of
    requiring that unrelated file to be touched."""
    bind = session.get_bind()
    UsageTraceRecord.metadata.create_all(bind, tables=[UsageTraceRecord.__table__], checkfirst=True)


def record_usage(
    session: Session,
    model_id: str,
    trace: dict,
    *,
    verified: bool = False,
    bundle_id: str | None = None,
    delta: dict | None = None,
    canonical_ref: str | None = None,
) -> UsageTraceRecord:
    """Persist one usage transition (an IC-5 trace, plus an optional IC-13 bundle/delta) for later mining."""
    _validate_trace(trace)
    if delta is not None:
        _validate_delta(delta)
    outcome = trace.get("outcome")
    is_number = isinstance(outcome, (int, float)) and not isinstance(outcome, bool)
    _ensure_schema(session)
    rec = UsageTraceRecord(
        model_id=model_id,
        trace_json=json.dumps(trace),
        outcome_value=float(outcome) if is_number else None,
        bundle_id=bundle_id,
        delta_json=json.dumps(delta) if delta is not None else None,
        canonical_ref=canonical_ref,
        verified=verified,
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def _mine_usage(
    session: Session,
    model_id: str,
    *,
    since: datetime | None = None,
    allowed_canonical_refs: Sequence[str] | None = None,
) -> tuple[list[Any], list[dict]]:
    """Steps (1)+(2) of the M7 algorithm: pull IC-5/IC-13 usage and keep only verified transitions on a
    policy-allowed canonical ref. Tries the real dataset foundry (M4, ``mixle_aifactory.foundry``) first so
    a landed foundry is used automatically; falls back to this module's own miner otherwise."""
    _ensure_schema(session)
    stmt = select(UsageTraceRecord).where(UsageTraceRecord.model_id == model_id, UsageTraceRecord.verified.is_(True))
    if since is not None:
        stmt = stmt.where(UsageTraceRecord.created_at >= since)
    stmt = stmt.order_by(UsageTraceRecord.created_at.asc())
    rows = list(session.exec(stmt).all())

    try:
        from mixle_aifactory.foundry import mine_examples  # the real M4 dataset foundry, once it lands
    except ImportError:
        mine_examples = None

    data: list[Any] = []
    bundle_lineage: list[dict] = []
    for row in rows:
        if allowed_canonical_refs is not None and row.canonical_ref not in allowed_canonical_refs:
            continue
        if mine_examples is not None:
            data.extend(mine_examples(json.loads(row.trace_json)))
        elif row.outcome_value is not None:
            data.append(row.outcome_value)
        if row.delta_json:
            bundle_lineage.append({"bundle_id": row.bundle_id, "delta": json.loads(row.delta_json)})
    return data, bundle_lineage


def _retrain(worker: EvolutionWorker, model_id: str, data: list[Any], policy: EvolutionPolicy) -> EvolutionRun:
    """Step (3) of the M7 algorithm: retrain via M5. ``mixle_aifactory.train`` is the real M5 retrain
    surface, once it lands; until then this reuses ``EvolutionWorker`` (which already wraps
    ``mixle.evolve.improve``) -- the M7 non-goal of adding no new promotion infrastructure."""
    try:
        from mixle_aifactory.train import retrain  # the real M5 retrain entrypoint, once it lands
    except ImportError:
        return worker.run(model_id, data, policy, promote=True)
    return retrain(worker, model_id, data, policy)


def improve_once(
    session: Session,
    *,
    model_id: str,
    verifier: Verifier,
    harness: Harness,
    registry: ModelRegistry | None = None,
    policy: EvolutionPolicy | None = None,
    since: datetime | None = None,
    allowed_canonical_refs: Sequence[str] | None = None,
) -> EvolutionRun:
    """One usage -> retrain -> verify -> promote cycle for ``model_id`` (M7's Public API).

    ``registry``, ``policy``, ``since`` and ``allowed_canonical_refs`` are additive, optional keyword
    arguments: the frozen call shape ``improve_once(session, model_id=..., verifier=..., harness=...)``
    still runs with library defaults for all of them -- an empty ``ModelRegistry`` (callers host their
    model in one they build and pass in) and the default ``EvolutionPolicy``.
    """
    registry = registry if registry is not None else ModelRegistry()
    policy = policy if policy is not None else EvolutionPolicy()
    worker = EvolutionWorker(registry)

    before = harness.evaluate(model_id)
    data, bundle_lineage = _mine_usage(session, model_id, since=since, allowed_canonical_refs=allowed_canonical_refs)
    run = _retrain(worker, model_id, data, policy)

    lineage: dict[str, Any] = {
        "usage_mined": len(data),
        "bundle_lineage": bundle_lineage,
        "harness_before": {"success": before.success},
    }

    if run.promoted:
        for entry in bundle_lineage:  # (4) verify delta schemas
            _validate_delta(entry["delta"])
        claim = {"model_id": model_id, "operator": run.operator, "delta": run.delta, "objective": run.objective}
        context = {"bundle_lineage": bundle_lineage, "n_data": run.n_data}
        verdict = verifier.verify(claim, context)  # (4) verify physical/calibration
        after = harness.evaluate(model_id)  # (5) the M8 gate

        blocked = (
            not getattr(verdict, "passed", False)
            or after.regressed
            or after.leak_detected
            or after.hash_mutation_detected
            or after.silent_overwrite_detected
            or after.success < before.success
            or not all(after.checks.values())
        )
        lineage["verifier_verdict"] = {
            "passed": getattr(verdict, "passed", None),
            "score": getattr(verdict, "score", None),
            "kind": getattr(verdict, "kind", None),
        }
        lineage["harness_after"] = {
            "success": after.success,
            "regressed": after.regressed,
            "checks": dict(after.checks),
        }
        if blocked:
            worker.rollback(model_id)
            run.promoted = False
            run.error = run.error or "blocked by IC-6 verifier / M8 harness gate; rolled back to champion"

    run.verdict = {**(run.verdict or {}), **lineage}  # (6) model + dataset + bundle/delta lineage
    record_run(session, run, user_id=None)
    return run
