"""Provider-neutral identities and state-machine contracts for operational work."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "1.0.0"


class OperationalError(ValueError):
    pass


class JobState(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvidenceKind(StrEnum):
    FACTORY = "factory"
    HARNESS = "harness"


def _json(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {key: _json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _nonempty(value: str, label: str) -> None:
    if not value or not value.strip():
        raise OperationalError(f"{label} must be non-empty")


def _digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise OperationalError(f"{label} must contain 64 hexadecimal characters")


@dataclass(frozen=True)
class OwnerScope:
    organization_id: str
    project_id: str

    def __post_init__(self) -> None:
        _nonempty(self.organization_id, "organization id")
        _nonempty(self.project_id, "project id")

    @property
    def key(self) -> str:
        return f"{self.organization_id}/{self.project_id}"


@dataclass(frozen=True)
class CapabilityRef:
    id: str
    version: str
    input_schema: str
    output_schema: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "capability id"),
            (self.version, "capability version"),
            (self.input_schema, "input schema"),
            (self.output_schema, "output schema"),
        ):
            _nonempty(value, label)


@dataclass(frozen=True)
class ArtifactRef:
    owner: OwnerScope
    sha256: str
    size_bytes: int
    media_type: str
    uri: str
    semantic_type: str

    def __post_init__(self) -> None:
        _digest(self.sha256, "artifact sha256")
        if self.size_bytes < 0:
            raise OperationalError("artifact size cannot be negative")
        _nonempty(self.media_type, "artifact media type")
        _nonempty(self.uri, "artifact uri")
        _nonempty(self.semantic_type, "artifact semantic type")


@dataclass(frozen=True)
class InvocationSpec:
    capability: CapabilityRef
    inputs: tuple[ArtifactRef, ...]
    parameters: dict[str, Any]
    knowledge_snapshot_id: str | None = None
    seed: int | None = None
    policy_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    @property
    def semantic_id(self) -> str:
        return semantic_hash(self)


@dataclass(frozen=True)
class ResourceLimits:
    timeout_seconds: float
    memory_bytes: int
    cpu_seconds: float
    output_bytes: int
    event_count: int = 1_000

    def __post_init__(self) -> None:
        if min(self.timeout_seconds, self.memory_bytes, self.cpu_seconds, self.output_bytes, self.event_count) <= 0:
            raise OperationalError("resource limits must be positive")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retryable_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise OperationalError("max attempts must be positive")


@dataclass(frozen=True)
class JobSpec:
    id: str
    owner: OwnerScope
    invocation: InvocationSpec
    resources: ResourceLimits
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    idempotency_key: str | None = None
    priority: int = 0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.id, "job id")
        if any(artifact.owner != self.owner for artifact in self.invocation.inputs):
            raise OperationalError("job inputs must belong to the job owner scope")

    @property
    def semantic_id(self) -> str:
        return self.invocation.semantic_id

    def as_dict(self) -> dict[str, Any]:
        return _json(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JobSpec:
        owner = OwnerScope(**value["owner"])
        capability = CapabilityRef(**value["invocation"]["capability"])
        invocation = InvocationSpec(
            capability=capability,
            inputs=tuple(
                ArtifactRef(
                    owner=OwnerScope(**item["owner"]), **{key: entry for key, entry in item.items() if key != "owner"}
                )
                for item in value["invocation"].get("inputs", [])
            ),
            parameters=dict(value["invocation"].get("parameters", {})),
            knowledge_snapshot_id=value["invocation"].get("knowledge_snapshot_id"),
            seed=value["invocation"].get("seed"),
            policy_id=value["invocation"].get("policy_id"),
            schema_version=value["invocation"].get("schema_version", SCHEMA_VERSION),
        )
        return cls(
            id=value["id"],
            owner=owner,
            invocation=invocation,
            resources=ResourceLimits(**value["resources"]),
            retry=RetryPolicy(**value.get("retry", {})),
            idempotency_key=value.get("idempotency_key"),
            priority=int(value.get("priority", 0)),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class EvidenceReceipt:
    id: str
    kind: EvidenceKind
    issuer: str
    subject_sha256: str
    passed: bool
    suites: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.id, "evidence receipt id")
        _nonempty(self.issuer, "evidence issuer")
        _digest(self.subject_sha256, "evidence subject sha256")


@dataclass(frozen=True)
class ModelCandidate:
    id: str
    model_id: str
    version: str
    artifact: ArtifactRef
    factory_receipt_id: str
    harness_receipt_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "candidate id"),
            (self.model_id, "model id"),
            (self.version, "model version"),
            (self.factory_receipt_id, "factory receipt id"),
        ):
            _nonempty(value, label)
        if not self.harness_receipt_ids:
            raise OperationalError("candidate requires at least one harness receipt id")


@dataclass(frozen=True)
class PromotionPolicy:
    trusted_factory_issuers: tuple[str, ...]
    trusted_harness_issuers: tuple[str, ...]
    required_suites: tuple[str, ...]
    aliases: tuple[str, ...] = ("stage", "production")


__all__ = [
    "SCHEMA_VERSION",
    "ArtifactRef",
    "CapabilityRef",
    "EvidenceKind",
    "EvidenceReceipt",
    "InvocationSpec",
    "JobSpec",
    "JobState",
    "ModelCandidate",
    "OperationalError",
    "OwnerScope",
    "PromotionPolicy",
    "ResourceLimits",
    "RetryPolicy",
    "canonical_json",
    "semantic_hash",
]
