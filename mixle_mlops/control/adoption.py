"""Authorization- and architecture-epoch-gated operational adoption."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import EvidenceReceipt, OperationalError, PromotionPolicy, canonical_json
from .registry import DeploymentReceipt, DeploymentRegistry


def _time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise OperationalError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OperationalError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class ArchitectureEpochPin:
    id: str
    revision: int
    architecture_digest: str
    approved_by: str
    effective_at: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.approved_by.strip() or self.revision < 0:
            raise OperationalError("architecture epoch identity, revision, and approver are required")
        if len(self.architecture_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.architecture_digest.lower()
        ):
            raise OperationalError("architecture epoch digest must be a sha256 digest")
        _time(self.effective_at, "architecture effective_at")


@dataclass(frozen=True)
class EvaluationAttestation:
    id: str
    candidate_id: str
    artifact_digest: str
    recommendation: str
    evaluator_project: str
    builder_project: str
    signature: str
    authorization_status: str = "not_requested"

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.id, self.candidate_id, self.evaluator_project, self.builder_project, self.signature)
        ):
            raise OperationalError("evaluation attestation identity fields are required")
        if len(self.artifact_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.artifact_digest.lower()
        ):
            raise OperationalError("evaluation artifact digest must be a sha256 digest")


@dataclass(frozen=True)
class LifecycleAuthorization:
    """Operational view of ``mixle.capability_lifecycle.AuthorizationDecision.as_dict``."""

    decision_id: str
    capability_id: str
    capability_version: str
    capability_digest: str | None
    outcome: str
    issued_by: str
    scopes: tuple[str, ...]
    decided_at: str
    expires_at: str | None = None
    revoked_at: str | None = None
    revoked_by: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LifecycleAuthorization:
        capability = value.get("capability")
        if not isinstance(capability, Mapping):
            raise OperationalError("authorization requires a capability identity")
        return cls(
            decision_id=str(value.get("decision_id", "")),
            capability_id=str(capability.get("capability_id", "")),
            capability_version=str(capability.get("version", "")),
            capability_digest=None if capability.get("digest") is None else str(capability["digest"]),
            outcome=str(value.get("outcome", "")),
            issued_by=str(value.get("issued_by", "")),
            scopes=tuple(str(scope) for scope in value.get("scopes", ())),
            decided_at=str(value.get("decided_at", "")),
            expires_at=None if value.get("expires_at") is None else str(value["expires_at"]),
            revoked_at=None if value.get("revoked_at") is None else str(value["revoked_at"]),
            revoked_by=None if value.get("revoked_by") is None else str(value["revoked_by"]),
        )

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.decision_id, self.capability_id, self.capability_version, self.issued_by)
        ):
            raise OperationalError("authorization identity and issuer fields are required")
        if not self.scopes or any(not scope.strip() for scope in self.scopes):
            raise OperationalError("authorization requires non-empty scopes")
        _time(self.decided_at, "authorization decided_at")
        if self.expires_at is not None:
            _time(self.expires_at, "authorization expires_at")
        if self.revoked_at is not None:
            _time(self.revoked_at, "authorization revoked_at")

    def allows(self, scope: str, *, at: str) -> bool:
        instant = _time(at, "adoption time")
        if self.outcome != "granted" or instant < _time(self.decided_at, "authorization decided_at"):
            return False
        if self.revoked_at is not None and instant >= _time(self.revoked_at, "authorization revoked_at"):
            return False
        if self.expires_at is not None and instant >= _time(self.expires_at, "authorization expires_at"):
            return False
        return scope in self.scopes or "*" in self.scopes


@dataclass(frozen=True)
class GovernedAdoptionPolicy:
    promotion: PromotionPolicy
    authorization_scope: str
    trusted_evaluator_projects: tuple[str, ...]
    trusted_authorization_issuers: tuple[str, ...]
    trusted_epoch_approvers: tuple[str, ...]


@dataclass(frozen=True)
class GovernedAdoptionReceipt:
    sequence: int
    deployment: DeploymentReceipt
    evaluation_id: str
    authorization_id: str
    architecture_epoch_id: str
    architecture_digest: str
    adopted_at: str


class GovernedDeploymentRegistry:
    """Adds independent evaluation, authorization, and epoch gates to deployment."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.deployments = DeploymentRegistry(self.root / "deployments")
        self.path = self.root / "governed-adoptions.json"
        self._receipts: list[GovernedAdoptionReceipt] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        if self.path.is_symlink():
            raise OperationalError("governed adoption registry cannot be a symlink")
        for item in json.loads(self.path.read_text()).get("receipts", []):
            self._receipts.append(
                GovernedAdoptionReceipt(
                    sequence=int(item["sequence"]),
                    deployment=DeploymentReceipt(
                        **{
                            **item["deployment"],
                            "evidence_receipt_ids": tuple(item["deployment"]["evidence_receipt_ids"]),
                        }
                    ),
                    evaluation_id=item["evaluation_id"],
                    authorization_id=item["authorization_id"],
                    architecture_epoch_id=item["architecture_epoch_id"],
                    architecture_digest=item["architecture_digest"],
                    adopted_at=item["adopted_at"],
                )
            )

    def _write(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(canonical_json({"schema_version": "1.0.0", "receipts": self._receipts}) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def adopt(
        self,
        candidate_id: str,
        alias: str,
        evidence: tuple[EvidenceReceipt, ...],
        evaluation: EvaluationAttestation,
        authorization: LifecycleAuthorization | Mapping[str, Any],
        epoch: ArchitectureEpochPin,
        policy: GovernedAdoptionPolicy,
        *,
        actor: str,
        verify_evaluation: Callable[[EvaluationAttestation], bool],
        adopted_at: str,
    ) -> GovernedAdoptionReceipt:
        candidate = self.deployments.candidate(candidate_id)
        authorization = (
            authorization
            if isinstance(authorization, LifecycleAuthorization)
            else LifecycleAuthorization.from_dict(authorization)
        )
        if evaluation.recommendation != "accept":
            raise OperationalError("independent evaluation did not recommend acceptance")
        if evaluation.builder_project == evaluation.evaluator_project:
            raise OperationalError("builder and evaluator projects must be distinct")
        if evaluation.evaluator_project not in policy.trusted_evaluator_projects:
            raise OperationalError("evaluation project is not trusted")
        if evaluation.candidate_id != candidate.id or evaluation.artifact_digest != candidate.artifact.sha256:
            raise OperationalError("evaluation targets a different immutable candidate")
        if not verify_evaluation(evaluation):
            raise OperationalError("evaluation signature verification failed")
        if evaluation.authorization_status != "not_requested":
            raise OperationalError("evaluation receipt must not claim operational authorization")
        if authorization.issued_by not in policy.trusted_authorization_issuers:
            raise OperationalError("authorization issuer is not trusted")
        if authorization.capability_id != candidate.model_id or authorization.capability_version != candidate.version:
            raise OperationalError("authorization targets a different capability version")
        if authorization.capability_digest not in {None, candidate.artifact.sha256}:
            raise OperationalError("authorization targets different artifact bytes")
        if not authorization.allows(policy.authorization_scope, at=adopted_at):
            raise OperationalError("authorization is absent, expired, revoked, denied, or out of scope")
        if epoch.approved_by not in policy.trusted_epoch_approvers:
            raise OperationalError("architecture epoch approver is not trusted")
        if _time(epoch.effective_at, "architecture effective_at") > _time(adopted_at, "adoption time"):
            raise OperationalError("architecture epoch is not effective at adoption time")
        if candidate.metadata.get("architecture_epoch_id") != epoch.id:
            raise OperationalError("candidate is not pinned to the selected architecture epoch")

        deployment = self.deployments.promote(candidate_id, alias, evidence, policy.promotion, actor=actor)
        existing = next(
            (
                receipt
                for receipt in reversed(self._receipts)
                if receipt.deployment == deployment
                and receipt.evaluation_id == evaluation.id
                and receipt.authorization_id == authorization.decision_id
                and receipt.architecture_epoch_id == epoch.id
            ),
            None,
        )
        if existing is not None:
            return existing
        receipt = GovernedAdoptionReceipt(
            sequence=len(self._receipts),
            deployment=deployment,
            evaluation_id=evaluation.id,
            authorization_id=authorization.decision_id,
            architecture_epoch_id=epoch.id,
            architecture_digest=epoch.architecture_digest,
            adopted_at=adopted_at,
        )
        self._receipts.append(receipt)
        self._write()
        return receipt

    @property
    def receipts(self) -> tuple[GovernedAdoptionReceipt, ...]:
        return tuple(self._receipts)


__all__ = [
    "ArchitectureEpochPin",
    "EvaluationAttestation",
    "GovernedAdoptionPolicy",
    "GovernedAdoptionReceipt",
    "GovernedDeploymentRegistry",
    "LifecycleAuthorization",
]
