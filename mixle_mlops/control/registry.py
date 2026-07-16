"""Immutable model candidates with evidence-gated promotion and auditable rollback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import (
    ArtifactRef,
    EvidenceKind,
    EvidenceReceipt,
    ModelCandidate,
    OperationalError,
    OwnerScope,
    PromotionPolicy,
    canonical_json,
)


@dataclass(frozen=True)
class DeploymentReceipt:
    sequence: int
    action: str
    alias: str
    candidate_id: str | None
    previous_candidate_id: str | None
    actor: str
    evidence_receipt_ids: tuple[str, ...]
    reason: str | None = None
    assessment_id: str | None = None
    policy_id: str | None = None
    authorization_id: str | None = None


@dataclass(frozen=True)
class QuarantineRecord:
    candidate_id: str
    deployment_sequence: int
    actor: str
    reason: str
    incident_id: str
    policy_id: str
    authorization_id: str
    quarantined_at: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "quarantine candidate id"),
            (self.actor, "quarantine actor"),
            (self.reason, "quarantine reason"),
            (self.incident_id, "quarantine incident id"),
            (self.policy_id, "quarantine policy id"),
            (self.authorization_id, "quarantine authorization id"),
            (self.quarantined_at, "quarantine timestamp"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise OperationalError(f"{label} must be non-empty")
        if (
            not isinstance(self.deployment_sequence, int)
            or isinstance(self.deployment_sequence, bool)
            or self.deployment_sequence < 0
        ):
            raise OperationalError("quarantine deployment sequence must be a non-negative integer")
        try:
            timestamp = datetime.fromisoformat(self.quarantined_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise OperationalError("quarantine timestamp must be RFC 3339") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise OperationalError("quarantine timestamp must include a timezone")


class DeploymentRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "deployments.json"
        self._candidates: dict[str, ModelCandidate] = {}
        self._aliases: dict[str, str] = {}
        self._previous: dict[str, str] = {}
        self._receipts: list[DeploymentReceipt] = []
        self._quarantines: dict[str, QuarantineRecord] = {}
        self._load()

    @staticmethod
    def _artifact(value: dict[str, Any]) -> ArtifactRef:
        return ArtifactRef(
            owner=OwnerScope(**value["owner"]), **{key: item for key, item in value.items() if key != "owner"}
        )

    @classmethod
    def _candidate(cls, value: dict[str, Any]) -> ModelCandidate:
        return ModelCandidate(
            id=value["id"],
            model_id=value["model_id"],
            version=value["version"],
            artifact=cls._artifact(value["artifact"]),
            factory_receipt_id=value["factory_receipt_id"],
            harness_receipt_ids=tuple(value["harness_receipt_ids"]),
            metadata=dict(value.get("metadata", {})),
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        if self.path.is_symlink():
            raise OperationalError("deployment registry cannot be a symlink")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        self._candidates = {key: self._candidate(item) for key, item in value.get("candidates", {}).items()}
        self._aliases = dict(value.get("aliases", {}))
        self._previous = dict(value.get("previous", {}))
        self._receipts = [
            DeploymentReceipt(**{**item, "evidence_receipt_ids": tuple(item["evidence_receipt_ids"])})
            for item in value.get("receipts", [])
        ]
        self._quarantines = {key: QuarantineRecord(**item) for key, item in value.get("quarantines", {}).items()}
        if any(key != item.candidate_id or key not in self._candidates for key, item in self._quarantines.items()):
            raise OperationalError("deployment registry contains an invalid quarantine record")

    def _write(self) -> None:
        rendered = (
            canonical_json(
                {
                    "schema_version": "1.0.0",
                    "candidates": self._candidates,
                    "aliases": self._aliases,
                    "previous": self._previous,
                    "receipts": self._receipts,
                    "quarantines": self._quarantines,
                }
            )
            + "\n"
        )
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def register(self, candidate: ModelCandidate) -> ModelCandidate:
        existing = self._candidates.get(candidate.id)
        if existing is not None and existing != candidate:
            raise OperationalError("immutable candidate id collision")
        self._candidates[candidate.id] = candidate
        self._write()
        return candidate

    def candidate(self, identifier: str) -> ModelCandidate:
        try:
            return self._candidates[identifier]
        except KeyError as exc:
            raise KeyError(identifier) from exc

    def resolve(self, alias: str) -> ModelCandidate:
        try:
            identifier = self._aliases[alias]
        except KeyError as exc:
            raise KeyError(alias) from exc
        if identifier in self._quarantines:
            raise OperationalError(f"candidate {identifier!r} is quarantined")
        return self.candidate(identifier)

    def deployment(self, alias: str, *, allow_quarantined: bool = False) -> DeploymentReceipt:
        try:
            candidate_id = self._aliases[alias]
        except KeyError as exc:
            raise KeyError(alias) from exc
        if not allow_quarantined and candidate_id in self._quarantines:
            raise OperationalError(f"candidate {candidate_id!r} is quarantined")
        try:
            return next(
                item
                for item in reversed(self._receipts)
                if item.alias == alias
                and item.candidate_id == candidate_id
                and item.action in {"promote", "rollback"}
            )
        except StopIteration as exc:
            raise OperationalError("alias has no deployment receipt") from exc

    def rollback_target(self, alias: str) -> ModelCandidate:
        current = self._aliases.get(alias)
        previous = self._previous.get(alias)
        if current is None or previous is None:
            raise OperationalError("alias has no rollback candidate")
        if previous in self._quarantines:
            raise OperationalError("rollback candidate is quarantined")
        return self.candidate(previous)

    @staticmethod
    def _verify(
        candidate: ModelCandidate,
        evidence: tuple[EvidenceReceipt, ...],
        policy: PromotionPolicy,
    ) -> tuple[str, ...]:
        by_id = {item.id: item for item in evidence}
        required_ids = {candidate.factory_receipt_id, *candidate.harness_receipt_ids}
        if not required_ids <= by_id.keys():
            raise OperationalError("promotion evidence is incomplete")
        selected = [by_id[identifier] for identifier in sorted(required_ids)]
        if any(not item.passed or item.subject_sha256 != candidate.artifact.sha256 for item in selected):
            raise OperationalError("promotion evidence failed or targets a different artifact")
        factory = [item for item in selected if item.kind is EvidenceKind.FACTORY]
        harness = [item for item in selected if item.kind is EvidenceKind.HARNESS]
        if not factory or any(item.issuer not in policy.trusted_factory_issuers for item in factory):
            raise OperationalError("factory evidence issuer is not trusted")
        if not harness or any(item.issuer not in policy.trusted_harness_issuers for item in harness):
            raise OperationalError("harness evidence issuer is not trusted")
        suites = {suite for item in harness for suite in item.suites}
        if not set(policy.required_suites) <= suites:
            raise OperationalError("required harness suites are missing")
        return tuple(item.id for item in selected)

    def promote(
        self,
        candidate_id: str,
        alias: str,
        evidence: tuple[EvidenceReceipt, ...],
        policy: PromotionPolicy,
        *,
        actor: str,
    ) -> DeploymentReceipt:
        if alias not in policy.aliases:
            raise OperationalError("alias is not allowed by promotion policy")
        candidate = self.candidate(candidate_id)
        if candidate_id in self._quarantines:
            raise OperationalError("quarantined candidate cannot be promoted")
        receipt_ids = self._verify(candidate, evidence, policy)
        previous = self._aliases.get(alias)
        if previous == candidate_id:
            return next(
                item
                for item in reversed(self._receipts)
                if item.action == "promote" and item.alias == alias and item.candidate_id == candidate_id
            )
        if previous is not None:
            self._previous[alias] = previous
        self._aliases[alias] = candidate_id
        receipt = DeploymentReceipt(
            len(self._receipts),
            "promote",
            alias,
            candidate_id,
            previous,
            actor,
            receipt_ids,
        )
        self._receipts.append(receipt)
        self._write()
        return receipt

    def rollback(
        self,
        alias: str,
        *,
        actor: str,
        reason: str,
        assessment_id: str | None = None,
        policy_id: str | None = None,
        authorization_id: str | None = None,
    ) -> DeploymentReceipt:
        current = self._aliases.get(alias)
        previous = self.rollback_target(alias).id
        self._aliases[alias] = previous
        self._previous[alias] = current
        receipt = DeploymentReceipt(
            len(self._receipts),
            "rollback",
            alias,
            previous,
            current,
            actor,
            (),
            reason,
            assessment_id,
            policy_id,
            authorization_id,
        )
        self._receipts.append(receipt)
        self._write()
        return receipt

    def mark_unhealthy(self, alias: str, *, actor: str, incident_id: str) -> DeploymentReceipt:
        return self.rollback(alias, actor=actor, reason=f"automatic bounded rollback for incident {incident_id}")

    def quarantine(
        self,
        candidate_id: str,
        *,
        deployment_sequence: int,
        actor: str,
        reason: str,
        incident_id: str,
        policy_id: str,
        authorization_id: str,
        quarantined_at: str,
    ) -> QuarantineRecord:
        self.candidate(candidate_id)
        for value, label in (
            (actor, "quarantine actor"),
            (reason, "quarantine reason"),
            (incident_id, "incident id"),
            (policy_id, "monitoring policy id"),
            (authorization_id, "monitoring authorization id"),
            (quarantined_at, "quarantine timestamp"),
        ):
            if not value or not value.strip():
                raise OperationalError(f"{label} must be non-empty")
        if (
            not isinstance(deployment_sequence, int)
            or isinstance(deployment_sequence, bool)
            or deployment_sequence < 0
        ):
            raise OperationalError("deployment sequence must be a non-negative integer")
        if not any(
            item.sequence == deployment_sequence and item.candidate_id == candidate_id for item in self._receipts
        ):
            raise OperationalError("quarantine does not identify a registered deployment receipt")
        record = QuarantineRecord(
            candidate_id,
            deployment_sequence,
            actor,
            reason,
            incident_id,
            policy_id,
            authorization_id,
            quarantined_at,
        )
        existing = self._quarantines.get(candidate_id)
        if existing is not None:
            if (
                existing.deployment_sequence != deployment_sequence
                or existing.incident_id != incident_id
                or existing.policy_id != policy_id
                or existing.authorization_id != authorization_id
            ):
                raise OperationalError("candidate is already quarantined by a different incident")
            return existing
        self._quarantines[candidate_id] = record
        self._write()
        return record

    def quarantine_record(self, candidate_id: str) -> QuarantineRecord | None:
        return self._quarantines.get(candidate_id)

    def receipt_for_assessment(self, assessment_id: str) -> DeploymentReceipt | None:
        return next(
            (item for item in reversed(self._receipts) if item.assessment_id == assessment_id),
            None,
        )

    @property
    def receipts(self) -> tuple[DeploymentReceipt, ...]:
        return tuple(self._receipts)


__all__ = ["DeploymentReceipt", "DeploymentRegistry", "QuarantineRecord"]
