"""Immutable model candidates with evidence-gated promotion and auditable rollback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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


class DeploymentRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "deployments.json"
        self._candidates: dict[str, ModelCandidate] = {}
        self._aliases: dict[str, str] = {}
        self._previous: dict[str, str] = {}
        self._receipts: list[DeploymentReceipt] = []
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

    def _write(self) -> None:
        rendered = (
            canonical_json(
                {
                    "schema_version": "1.0.0",
                    "candidates": self._candidates,
                    "aliases": self._aliases,
                    "previous": self._previous,
                    "receipts": self._receipts,
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
            return self.candidate(self._aliases[alias])
        except KeyError as exc:
            raise KeyError(alias) from exc

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

    def rollback(self, alias: str, *, actor: str, reason: str) -> DeploymentReceipt:
        current = self._aliases.get(alias)
        previous = self._previous.get(alias)
        if current is None or previous is None:
            raise OperationalError("alias has no rollback candidate")
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
        )
        self._receipts.append(receipt)
        self._write()
        return receipt

    def mark_unhealthy(self, alias: str, *, actor: str, incident_id: str) -> DeploymentReceipt:
        return self.rollback(alias, actor=actor, reason=f"automatic bounded rollback for incident {incident_id}")

    @property
    def receipts(self) -> tuple[DeploymentReceipt, ...]:
        return tuple(self._receipts)


__all__ = ["DeploymentReceipt", "DeploymentRegistry"]
