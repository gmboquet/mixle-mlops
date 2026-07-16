"""Durable, separately authorized architecture migration and rollback."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ArchitectureTransitionError(ValueError):
    """Raised when a transition would violate identity or authority boundaries."""


class TransitionState(StrEnum):
    SUBMITTED = "submitted"
    REJECTED = "rejected"
    AUTHORIZED = "authorized"
    DUAL_RUNNING = "dual_running"
    PROMOTED = "promoted"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


def _nonempty(value: str, label: str) -> None:
    if not value or not value.strip():
        raise ArchitectureTransitionError(f"{label} must be non-empty")


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ProviderContract:
    id: str
    provider: str
    version: str
    input_schema: str
    output_schema: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "provider contract id"),
            (self.provider, "provider"),
            (self.version, "provider contract version"),
            (self.input_schema, "provider input schema"),
            (self.output_schema, "provider output schema"),
        ):
            _nonempty(value, label)


@dataclass(frozen=True)
class ConsumerMigration:
    consumer_id: str
    from_contract_id: str
    to_contract_id: str
    compatibility_mode: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.consumer_id, "consumer id"),
            (self.from_contract_id, "source contract id"),
            (self.to_contract_id, "target contract id"),
            (self.compatibility_mode, "compatibility mode"),
        ):
            _nonempty(value, label)
        if self.from_contract_id == self.to_contract_id:
            raise ArchitectureTransitionError("consumer migration must change provider contracts")


@dataclass(frozen=True)
class DualRunPolicy:
    comparison_metrics: tuple[str, ...]
    max_runs: int
    minimum_evidence: int

    def __post_init__(self) -> None:
        if not self.comparison_metrics or len(self.comparison_metrics) != len(set(self.comparison_metrics)):
            raise ArchitectureTransitionError("dual-run comparison metrics must be non-empty and unique")
        if self.max_runs < 1 or self.minimum_evidence < 1:
            raise ArchitectureTransitionError("dual-run bounds and evidence requirements must be positive")


@dataclass(frozen=True)
class RollbackPlan:
    triggers: tuple[str, ...]
    restore_contract_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.triggers or not self.restore_contract_ids:
            raise ArchitectureTransitionError("rollback triggers and restoration contracts are required")


@dataclass(frozen=True)
class MigrationPlan:
    id: str
    proposal_id: str
    governance_epoch: str
    provider_contracts: tuple[ProviderContract, ...]
    consumer_migrations: tuple[ConsumerMigration, ...]
    dual_run: DualRunPolicy
    rollback: RollbackPlan
    change_targets: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "migration plan id"),
            (self.proposal_id, "architecture proposal id"),
            (self.governance_epoch, "governance epoch"),
        ):
            _nonempty(value, label)
        if not self.provider_contracts or not self.consumer_migrations:
            raise ArchitectureTransitionError("migration requires provider contracts and consumer migrations")
        contract_ids = [contract.id for contract in self.provider_contracts]
        if len(contract_ids) != len(set(contract_ids)):
            raise ArchitectureTransitionError("provider contract ids must be unique")
        known = set(contract_ids)
        if any(migration.to_contract_id not in known for migration in self.consumer_migrations):
            raise ArchitectureTransitionError("consumer migration targets an undeclared provider contract")
        if not set(self.rollback.restore_contract_ids) <= {
            migration.from_contract_id for migration in self.consumer_migrations
        }:
            raise ArchitectureTransitionError("rollback must restore a declared prior consumer contract")


@dataclass(frozen=True)
class TransitionAuthorization:
    id: str
    proposal_id: str
    evaluation_id: str
    governance_epoch: str
    issuer: str
    evaluation_accepted: bool
    allowed_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "transition authorization id"),
            (self.proposal_id, "authorized proposal id"),
            (self.evaluation_id, "independent evaluation id"),
            (self.governance_epoch, "authorization governance epoch"),
            (self.issuer, "authorization issuer"),
        ):
            _nonempty(value, label)


@dataclass(frozen=True)
class TransitionReceipt:
    sequence: int
    event_id: str
    plan_id: str
    request_action: str
    recorded_action: str
    state: TransitionState
    evidence_refs: tuple[str, ...]
    detail: str


_PROTECTED_TARGETS = frozenset({"evaluator", "governance", "rollout_controller"})
_REQUIRED_AUTHORITY = frozenset({"dual_run", "promote", "rollback"})


class ArchitectureTransitionController:
    """Atomic local ledger for reversible migration; it never edits code or rules."""

    def __init__(
        self,
        root: str | Path,
        *,
        governance_epoch: str,
        trusted_authorization_issuers: tuple[str, ...],
    ) -> None:
        _nonempty(governance_epoch, "controller governance epoch")
        if not trusted_authorization_issuers or len(trusted_authorization_issuers) != len(
            set(trusted_authorization_issuers)
        ):
            raise ArchitectureTransitionError("trusted authorization issuers must be non-empty and unique")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "architecture-transitions.json"
        self.governance_epoch = governance_epoch
        self.trusted_authorization_issuers = trusted_authorization_issuers
        self._plans: dict[str, MigrationPlan] = {}
        self._authorizations: dict[str, TransitionAuthorization] = {}
        self._states: dict[str, TransitionState] = {}
        self._receipts: list[TransitionReceipt] = []
        self._events: dict[str, TransitionReceipt] = {}
        self._load()

    @staticmethod
    def _plan(value: dict[str, Any]) -> MigrationPlan:
        return MigrationPlan(
            id=value["id"],
            proposal_id=value["proposal_id"],
            governance_epoch=value["governance_epoch"],
            provider_contracts=tuple(ProviderContract(**item) for item in value["provider_contracts"]),
            consumer_migrations=tuple(ConsumerMigration(**item) for item in value["consumer_migrations"]),
            dual_run=DualRunPolicy(
                comparison_metrics=tuple(value["dual_run"]["comparison_metrics"]),
                max_runs=value["dual_run"]["max_runs"],
                minimum_evidence=value["dual_run"]["minimum_evidence"],
            ),
            rollback=RollbackPlan(
                triggers=tuple(value["rollback"]["triggers"]),
                restore_contract_ids=tuple(value["rollback"]["restore_contract_ids"]),
            ),
            change_targets=tuple(value["change_targets"]),
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        if self.path.is_symlink():
            raise ArchitectureTransitionError("transition ledger cannot be a symlink")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("governance_epoch") != self.governance_epoch:
            raise ArchitectureTransitionError("transition ledger governance epoch does not match the controller")
        if tuple(value.get("trusted_authorization_issuers", ())) != self.trusted_authorization_issuers:
            raise ArchitectureTransitionError("transition ledger authorization issuers do not match the controller")
        self._plans = {identifier: self._plan(item) for identifier, item in value.get("plans", {}).items()}
        self._authorizations = {
            identifier: TransitionAuthorization(**{**item, "allowed_actions": tuple(item["allowed_actions"])})
            for identifier, item in value.get("authorizations", {}).items()
        }
        self._states = {identifier: TransitionState(state) for identifier, state in value.get("states", {}).items()}
        self._receipts = [
            TransitionReceipt(
                **{
                    **item,
                    "state": TransitionState(item["state"]),
                    "evidence_refs": tuple(item["evidence_refs"]),
                }
            )
            for item in value.get("receipts", [])
        ]
        self._events = {receipt.event_id: receipt for receipt in self._receipts}

    def _write(self) -> None:
        value = {
            "schema_version": "1.0.0",
            "governance_epoch": self.governance_epoch,
            "trusted_authorization_issuers": self.trusted_authorization_issuers,
            "plans": {identifier: asdict(plan) for identifier, plan in self._plans.items()},
            "authorizations": {
                identifier: asdict(authorization) for identifier, authorization in self._authorizations.items()
            },
            "states": {identifier: state.value for identifier, state in self._states.items()},
            "receipts": [{**asdict(receipt), "state": receipt.state.value} for receipt in self._receipts],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def _replay(self, event_id: str, plan_id: str, request_action: str) -> TransitionReceipt | None:
        existing = self._events.get(event_id)
        if existing is None:
            return None
        if (existing.plan_id, existing.request_action) != (plan_id, request_action):
            raise ArchitectureTransitionError("replayed transition event has different semantics")
        return existing

    def _record(
        self,
        *,
        event_id: str,
        plan_id: str,
        request_action: str,
        recorded_action: str,
        state: TransitionState,
        evidence_refs: tuple[str, ...] = (),
        detail: str,
    ) -> TransitionReceipt:
        _nonempty(event_id, "transition event id")
        if event_id in self._events:
            raise ArchitectureTransitionError("transition event was not replay-checked")
        receipt = TransitionReceipt(
            len(self._receipts),
            event_id,
            plan_id,
            request_action,
            recorded_action,
            state,
            evidence_refs,
            detail,
        )
        self._receipts.append(receipt)
        self._events[event_id] = receipt
        self._states[plan_id] = state
        self._write()
        return receipt

    def submit(self, plan: MigrationPlan, *, event_id: str) -> TransitionReceipt:
        replay = self._replay(event_id, plan.id, "submit")
        if replay is not None:
            if self._plans.get(plan.id) != plan:
                raise ArchitectureTransitionError("replayed submission contains a different plan")
            return replay
        existing = self._plans.get(plan.id)
        if existing is not None and existing != plan:
            raise ArchitectureTransitionError("immutable migration plan id collision")
        self._plans[plan.id] = plan
        issues: list[str] = []
        if plan.governance_epoch != self.governance_epoch:
            issues.append("governance epoch mismatch")
        if _PROTECTED_TARGETS.intersection(plan.change_targets):
            issues.append("plan targets a protected evaluator, governance, or rollout control")
        state = TransitionState.REJECTED if issues else TransitionState.SUBMITTED
        return self._record(
            event_id=event_id,
            plan_id=plan.id,
            request_action="submit",
            recorded_action="rejected" if issues else "submitted",
            state=state,
            detail="; ".join(issues) if issues else "plan retained pending separate authorization",
        )

    def authorize(
        self,
        plan_id: str,
        authorization: TransitionAuthorization,
        *,
        event_id: str,
    ) -> TransitionReceipt:
        replay = self._replay(event_id, plan_id, "authorize")
        authorization_evidence = (
            authorization.id,
            authorization.evaluation_id,
            _semantic_sha256(asdict(authorization)),
        )
        if replay is not None:
            if replay.evidence_refs != authorization_evidence:
                raise ArchitectureTransitionError("replayed authorization has different semantics")
            return replay
        plan = self._plans[plan_id]
        if self._states[plan_id] is not TransitionState.SUBMITTED:
            return self._record(
                event_id=event_id,
                plan_id=plan_id,
                request_action="authorize",
                recorded_action="ignored_out_of_order",
                state=self._states[plan_id],
                evidence_refs=authorization_evidence,
                detail="authorization requires a non-rejected submitted plan",
            )
        issues: list[str] = []
        if authorization.proposal_id != plan.proposal_id:
            issues.append("authorization targets a different proposal")
        if authorization.governance_epoch != self.governance_epoch:
            issues.append("authorization governance epoch mismatch")
        if not authorization.evaluation_accepted:
            issues.append("independent evaluation did not accept the proposal")
        if authorization.issuer not in self.trusted_authorization_issuers:
            issues.append("authorization issuer is not trusted")
        if not _REQUIRED_AUTHORITY <= set(authorization.allowed_actions):
            issues.append("authorization does not cover dual run, promotion, and rollback")
        if issues:
            return self._record(
                event_id=event_id,
                plan_id=plan_id,
                request_action="authorize",
                recorded_action="rejected",
                state=TransitionState.REJECTED,
                evidence_refs=authorization_evidence,
                detail="; ".join(issues),
            )
        self._authorizations[plan_id] = authorization
        return self._record(
            event_id=event_id,
            plan_id=plan_id,
            request_action="authorize",
            recorded_action="authorized",
            state=TransitionState.AUTHORIZED,
            evidence_refs=authorization_evidence,
            detail="separate rollout authority verified",
        )

    def start_dual_run(self, plan_id: str, *, event_id: str) -> TransitionReceipt:
        return self._transition(
            plan_id,
            event_id=event_id,
            request_action="start_dual_run",
            required_state=TransitionState.AUTHORIZED,
            next_state=TransitionState.DUAL_RUNNING,
            evidence_refs=(self._authorizations[plan_id].id,) if plan_id in self._authorizations else (),
            detail="bounded dual run started",
        )

    def complete_dual_run(
        self,
        plan_id: str,
        *,
        event_id: str,
        passed: bool,
        evidence_refs: tuple[str, ...],
    ) -> TransitionReceipt:
        policy = self._plans[plan_id].dual_run
        if len(evidence_refs) < policy.minimum_evidence or len(evidence_refs) != len(set(evidence_refs)):
            raise ArchitectureTransitionError(
                "dual-run completion requires unique comparison evidence at its minimum"
            )
        next_state = TransitionState.PROMOTED if passed else TransitionState.FAILED
        return self._transition(
            plan_id,
            event_id=event_id,
            request_action=f"complete_dual_run:{str(passed).lower()}",
            required_state=TransitionState.DUAL_RUNNING,
            next_state=next_state,
            evidence_refs=evidence_refs,
            detail="dual-run comparison passed" if passed else "dual-run comparison failed",
        )

    def rollback(
        self,
        plan_id: str,
        *,
        event_id: str,
        reason: str,
        evidence_refs: tuple[str, ...],
    ) -> TransitionReceipt:
        _nonempty(reason, "rollback reason")
        authorization = self._authorizations.get(plan_id)
        expected_evidence = (authorization.id, *evidence_refs) if authorization is not None else evidence_refs
        replay = self._replay(event_id, plan_id, "rollback")
        if replay is not None:
            if replay.evidence_refs != expected_evidence or (
                replay.recorded_action != "ignored_out_of_order" and replay.detail != reason
            ):
                raise ArchitectureTransitionError("replayed rollback has different semantics")
            return replay
        state = self._states[plan_id]
        allowed = {
            TransitionState.AUTHORIZED,
            TransitionState.DUAL_RUNNING,
            TransitionState.PROMOTED,
            TransitionState.FAILED,
        }
        if state not in allowed:
            return self._record(
                event_id=event_id,
                plan_id=plan_id,
                request_action="rollback",
                recorded_action="ignored_out_of_order",
                state=state,
                evidence_refs=expected_evidence,
                detail="rollback is not valid from the retained state",
            )
        if authorization is None or "rollback" not in authorization.allowed_actions:
            raise ArchitectureTransitionError("rollback requires retained explicit authority")
        return self._record(
            event_id=event_id,
            plan_id=plan_id,
            request_action="rollback",
            recorded_action="rolled_back",
            state=TransitionState.ROLLED_BACK,
            evidence_refs=(authorization.id, *evidence_refs),
            detail=reason,
        )

    def _transition(
        self,
        plan_id: str,
        *,
        event_id: str,
        request_action: str,
        required_state: TransitionState,
        next_state: TransitionState,
        evidence_refs: tuple[str, ...],
        detail: str,
    ) -> TransitionReceipt:
        replay = self._replay(event_id, plan_id, request_action)
        if replay is not None:
            if replay.evidence_refs != evidence_refs or (
                replay.recorded_action != "ignored_out_of_order" and replay.detail != detail
            ):
                raise ArchitectureTransitionError("replayed transition event has different semantics")
            return replay
        current = self._states[plan_id]
        if current is not required_state:
            return self._record(
                event_id=event_id,
                plan_id=plan_id,
                request_action=request_action,
                recorded_action="ignored_out_of_order",
                state=current,
                evidence_refs=evidence_refs,
                detail=f"expected {required_state.value}, retained {current.value}",
            )
        return self._record(
            event_id=event_id,
            plan_id=plan_id,
            request_action=request_action,
            recorded_action=next_state.value,
            state=next_state,
            evidence_refs=evidence_refs,
            detail=detail,
        )

    def state(self, plan_id: str) -> TransitionState:
        return self._states[plan_id]

    @property
    def receipts(self) -> tuple[TransitionReceipt, ...]:
        return tuple(self._receipts)
