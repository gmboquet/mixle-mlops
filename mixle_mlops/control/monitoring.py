"""Deployment-bound operational monitoring with durable, stale-safe enforcement."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .contracts import OperationalError, canonical_json, semantic_hash
from .registry import DeploymentRegistry

MONITORING_SCHEMA_VERSION = "1.0.0"


class _FrozenMetrics(dict[str, float]):
    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("health observation metrics are immutable")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OperationalError(f"{label} must be non-empty")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise OperationalError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OperationalError(f"{label} must be a non-negative integer")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OperationalError(f"{label} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OperationalError(f"{label} must include a timezone")
    return parsed


def _strict_json(raw: str) -> dict[str, Any]:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise OperationalError(f"monitoring ledger contains duplicate field {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value!r}")

    try:
        value = json.loads(raw, object_pairs_hook=object_from_pairs, parse_constant=reject_constant)
    except (RecursionError, TypeError, ValueError) as exc:
        raise OperationalError(f"monitoring ledger is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise OperationalError("monitoring ledger must contain an object")
    return value


class ThresholdDirection(StrEnum):
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


class HealthState(StrEnum):
    INSUFFICIENT = "insufficient"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class EnforcementAction(StrEnum):
    NONE = "none"
    QUARANTINE = "quarantine"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class MetricThreshold:
    name: str
    direction: ThresholdDirection
    bound: float

    def __post_init__(self) -> None:
        _nonempty(self.name, "metric name")
        if not isinstance(self.direction, ThresholdDirection):
            raise OperationalError("metric direction must be a ThresholdDirection")
        if not isinstance(self.bound, (int, float)) or isinstance(self.bound, bool) or not math.isfinite(self.bound):
            raise OperationalError("metric bound must be finite")

    def breached(self, value: float) -> bool:
        if self.direction is ThresholdDirection.MAXIMUM:
            return value > self.bound
        return value < self.bound


@dataclass(frozen=True)
class MonitoringPolicy:
    id: str
    version: str
    thresholds: tuple[MetricThreshold, ...]
    authorized_by: str
    aliases: tuple[str, ...] = ("production",)
    window_size: int = 20
    minimum_samples: int = 5
    max_observation_age_seconds: float = 300.0
    max_breach_fraction: float = 0.0
    action: EnforcementAction = EnforcementAction.ROLLBACK
    schema_version: str = MONITORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.id, "monitoring policy id")
        _nonempty(self.version, "monitoring policy version")
        _nonempty(self.authorized_by, "monitoring authorization id")
        if self.schema_version != MONITORING_SCHEMA_VERSION:
            raise OperationalError(f"unsupported monitoring schema {self.schema_version!r}")
        if not isinstance(self.thresholds, (list, tuple)) or not self.thresholds:
            raise OperationalError("monitoring policy requires at least one threshold")
        if not all(isinstance(item, MetricThreshold) for item in self.thresholds):
            raise OperationalError("monitoring thresholds must be MetricThreshold values")
        names = [item.name for item in self.thresholds]
        if len(names) != len(set(names)):
            raise OperationalError("monitoring threshold names must be unique")
        if not isinstance(self.aliases, (list, tuple)) or not self.aliases:
            raise OperationalError("monitoring policy requires at least one alias")
        if any(not isinstance(item, str) or not item.strip() for item in self.aliases):
            raise OperationalError("monitoring aliases must be non-empty strings")
        if len(self.aliases) != len(set(self.aliases)):
            raise OperationalError("monitoring aliases must be unique")
        _positive_integer(self.window_size, "monitoring window size")
        _positive_integer(self.minimum_samples, "monitoring minimum samples")
        if self.minimum_samples > self.window_size:
            raise OperationalError("monitoring minimum samples cannot exceed the window size")
        if (
            not isinstance(self.max_observation_age_seconds, (int, float))
            or isinstance(self.max_observation_age_seconds, bool)
            or not math.isfinite(self.max_observation_age_seconds)
            or self.max_observation_age_seconds <= 0
        ):
            raise OperationalError("maximum observation age must be finite and positive")
        if (
            not isinstance(self.max_breach_fraction, (int, float))
            or isinstance(self.max_breach_fraction, bool)
            or not math.isfinite(self.max_breach_fraction)
            or not 0 <= self.max_breach_fraction <= 1
        ):
            raise OperationalError("maximum breach fraction must be finite and between zero and one")
        if self.action not in {EnforcementAction.QUARANTINE, EnforcementAction.ROLLBACK}:
            raise OperationalError("an unhealthy monitoring policy must quarantine or roll back")
        object.__setattr__(self, "thresholds", tuple(sorted(self.thresholds, key=lambda item: item.name)))
        object.__setattr__(self, "aliases", tuple(sorted(self.aliases)))

    @property
    def semantic_id(self) -> str:
        return semantic_hash(self)


@dataclass(frozen=True)
class HealthObservation:
    id: str
    alias: str
    candidate_id: str
    deployment_sequence: int
    observed_at: str
    source: str
    metrics: dict[str, float]
    schema_version: str = MONITORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "observation id"),
            (self.alias, "deployment alias"),
            (self.candidate_id, "candidate id"),
            (self.source, "observation source"),
        ):
            _nonempty(value, label)
        _nonnegative_integer(self.deployment_sequence, "deployment sequence")
        _timestamp(self.observed_at, "observation timestamp")
        if self.schema_version != MONITORING_SCHEMA_VERSION:
            raise OperationalError(f"unsupported monitoring schema {self.schema_version!r}")
        if not isinstance(self.metrics, dict) or not self.metrics:
            raise OperationalError("health observation requires metrics")
        for name, value in self.metrics.items():
            _nonempty(name, "metric name")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise OperationalError(f"metric {name!r} must be finite")
        object.__setattr__(self, "metrics", _FrozenMetrics(sorted(self.metrics.items())))

    @property
    def semantic_id(self) -> str:
        return semantic_hash(self)


@dataclass(frozen=True)
class MetricAssessment:
    name: str
    direction: ThresholdDirection
    bound: float
    samples: int
    breaches: int
    missing: int
    breach_fraction: float

    def __post_init__(self) -> None:
        _nonempty(self.name, "assessment metric name")
        if not isinstance(self.direction, ThresholdDirection):
            raise OperationalError("assessment metric direction must be a ThresholdDirection")
        if not isinstance(self.bound, (int, float)) or isinstance(self.bound, bool) or not math.isfinite(self.bound):
            raise OperationalError("assessment metric bound must be finite")
        _nonnegative_integer(self.samples, "assessment sample count")
        _nonnegative_integer(self.breaches, "assessment breach count")
        _nonnegative_integer(self.missing, "assessment missing count")
        if self.breaches > self.samples or self.missing > self.breaches:
            raise OperationalError("assessment metric counts are inconsistent")
        if (
            not isinstance(self.breach_fraction, (int, float))
            or isinstance(self.breach_fraction, bool)
            or not math.isfinite(self.breach_fraction)
            or not 0 <= self.breach_fraction <= 1
        ):
            raise OperationalError("assessment breach fraction must be finite and between zero and one")
        expected = self.breaches / self.samples if self.samples else 0.0
        if not math.isclose(self.breach_fraction, expected, rel_tol=0.0, abs_tol=1e-15):
            raise OperationalError("assessment breach fraction does not match its counts")


@dataclass(frozen=True)
class HealthAssessment:
    alias: str
    candidate_id: str
    deployment_sequence: int
    policy_id: str
    policy_version: str
    policy_sha256: str
    authorization_id: str
    policy: MonitoringPolicy
    state: HealthState
    action: EnforcementAction
    observation_ids: tuple[str, ...]
    observation_sha256s: tuple[str, ...]
    metrics: tuple[MetricAssessment, ...]
    created_at: str
    schema_version: str = MONITORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, label in (
            (self.alias, "assessment alias"),
            (self.candidate_id, "assessment candidate id"),
            (self.policy_id, "assessment policy id"),
            (self.policy_version, "assessment policy version"),
            (self.authorization_id, "assessment authorization id"),
        ):
            _nonempty(value, label)
        _nonnegative_integer(self.deployment_sequence, "assessment deployment sequence")
        if len(self.policy_sha256) != 64 or any(
            item not in "0123456789abcdef" for item in self.policy_sha256.lower()
        ):
            raise OperationalError("assessment policy sha256 must contain 64 hexadecimal characters")
        if not isinstance(self.policy, MonitoringPolicy):
            raise OperationalError("assessment must embed its exact MonitoringPolicy")
        if (
            self.policy.id != self.policy_id
            or self.policy.version != self.policy_version
            or self.policy.semantic_id != self.policy_sha256
            or self.policy.authorized_by != self.authorization_id
            or self.alias not in self.policy.aliases
        ):
            raise OperationalError("assessment policy identity is inconsistent")
        _timestamp(self.created_at, "assessment timestamp")
        if not isinstance(self.observation_ids, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in self.observation_ids
        ):
            raise OperationalError("assessment observation ids must be non-empty strings")
        if len(self.observation_ids) != len(set(self.observation_ids)):
            raise OperationalError("assessment observation ids must be unique")
        if not isinstance(self.observation_sha256s, (list, tuple)) or len(self.observation_sha256s) != len(
            self.observation_ids
        ):
            raise OperationalError("assessment observation hashes must align with observation ids")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value.lower())
            for value in self.observation_sha256s
        ):
            raise OperationalError("assessment observation hashes must contain 64 hexadecimal characters")
        if not isinstance(self.metrics, (list, tuple)) or not all(
            isinstance(item, MetricAssessment) for item in self.metrics
        ):
            raise OperationalError("assessment metrics must be MetricAssessment values")
        if len({item.name for item in self.metrics}) != len(self.metrics):
            raise OperationalError("assessment metric names must be unique")
        if {item.name for item in self.metrics} != {item.name for item in self.policy.thresholds}:
            raise OperationalError("assessment metrics do not match the embedded policy")
        if any(item.samples != len(self.observation_ids) for item in self.metrics):
            raise OperationalError("assessment metric sample counts do not match its observations")
        if not isinstance(self.state, HealthState) or not isinstance(self.action, EnforcementAction):
            raise OperationalError("assessment state and action must use monitoring enums")
        if len(self.observation_ids) < self.policy.minimum_samples:
            expected_state = HealthState.INSUFFICIENT
        elif any(item.breach_fraction > self.policy.max_breach_fraction for item in self.metrics):
            expected_state = HealthState.UNHEALTHY
        else:
            expected_state = HealthState.HEALTHY
        if self.state is not expected_state:
            raise OperationalError("assessment state does not match its policy and metric evidence")
        expected_action = self.policy.action if self.state is HealthState.UNHEALTHY else EnforcementAction.NONE
        if self.action is not expected_action:
            raise OperationalError("assessment action does not match its policy and state")
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))
        object.__setattr__(self, "observation_sha256s", tuple(self.observation_sha256s))
        object.__setattr__(self, "metrics", tuple(self.metrics))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "alias": self.alias,
            "candidate_id": self.candidate_id,
            "deployment_sequence": self.deployment_sequence,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "authorization_id": self.authorization_id,
            "policy": self.policy,
            "state": self.state,
            "action": self.action,
            "observation_ids": self.observation_ids,
            "observation_sha256s": self.observation_sha256s,
            "metrics": self.metrics,
        }

    @property
    def semantic_id(self) -> str:
        return semantic_hash(self._identity_payload())

    @property
    def id(self) -> str:
        return f"health-{self.semantic_id}"


@dataclass(frozen=True)
class EnforcementReceipt:
    assessment_id: str
    assessment_sha256: str
    alias: str
    unhealthy_candidate_id: str
    action: EnforcementAction
    quarantine_incident_id: str
    policy_id: str
    authorization_id: str
    actor: str
    enforced_at: str
    rollback_sequence: int | None = None
    rollback_candidate_id: str | None = None
    schema_version: str = MONITORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, label in (
            (self.assessment_id, "enforcement assessment id"),
            (self.alias, "enforcement alias"),
            (self.unhealthy_candidate_id, "enforcement candidate id"),
            (self.quarantine_incident_id, "enforcement incident id"),
            (self.policy_id, "enforcement policy id"),
            (self.authorization_id, "enforcement authorization id"),
            (self.actor, "enforcement actor"),
        ):
            _nonempty(value, label)
        if len(self.assessment_sha256) != 64 or any(
            item not in "0123456789abcdef" for item in self.assessment_sha256.lower()
        ):
            raise OperationalError("enforcement assessment sha256 must contain 64 hexadecimal characters")
        if self.action not in {EnforcementAction.QUARANTINE, EnforcementAction.ROLLBACK}:
            raise OperationalError("enforcement action must quarantine or roll back")
        if self.schema_version != MONITORING_SCHEMA_VERSION:
            raise OperationalError(f"unsupported monitoring schema {self.schema_version!r}")
        _timestamp(self.enforced_at, "enforcement timestamp")
        if self.rollback_sequence is not None:
            _nonnegative_integer(self.rollback_sequence, "rollback sequence")
        if (self.rollback_sequence is None) != (self.rollback_candidate_id is None):
            raise OperationalError("rollback sequence and candidate must be present together")
        if self.action is EnforcementAction.ROLLBACK and self.rollback_sequence is None:
            raise OperationalError("rollback enforcement requires a deployment receipt")
        if self.action is EnforcementAction.QUARANTINE and self.rollback_sequence is not None:
            raise OperationalError("quarantine-only enforcement cannot contain a rollback receipt")

    @property
    def semantic_id(self) -> str:
        return semantic_hash(self)


class DeploymentMonitor:
    """Atomic local ledger; telemetry collection and scientific interpretation stay outside this class."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "monitoring.json"
        self._observations: dict[str, HealthObservation] = {}
        self._assessments: dict[str, HealthAssessment] = {}
        self._enforcements: dict[str, EnforcementReceipt] = {}
        self._load()

    @staticmethod
    def _policy(value: dict[str, Any]) -> MonitoringPolicy:
        return MonitoringPolicy(
            **{
                **value,
                "thresholds": tuple(
                    MetricThreshold(**{**item, "direction": ThresholdDirection(item["direction"])})
                    for item in value["thresholds"]
                ),
                "aliases": tuple(value.get("aliases", [])),
                "action": EnforcementAction(value["action"]),
            }
        )

    @staticmethod
    def _observation(value: dict[str, Any]) -> HealthObservation:
        return HealthObservation(**value)

    @staticmethod
    def _assessment(value: dict[str, Any]) -> HealthAssessment:
        return HealthAssessment(
            **{
                **value,
                "state": HealthState(value["state"]),
                "action": EnforcementAction(value["action"]),
                "policy": DeploymentMonitor._policy(value["policy"]),
                "observation_ids": tuple(value.get("observation_ids", [])),
                "observation_sha256s": tuple(value.get("observation_sha256s", [])),
                "metrics": tuple(
                    MetricAssessment(**{**item, "direction": ThresholdDirection(item["direction"])})
                    for item in value.get("metrics", [])
                ),
            }
        )

    @staticmethod
    def _enforcement(value: dict[str, Any]) -> EnforcementReceipt:
        return EnforcementReceipt(**{**value, "action": EnforcementAction(value["action"])})

    def _load(self) -> None:
        if not self.path.exists():
            return
        if self.path.is_symlink() or not self.path.is_file():
            raise OperationalError("monitoring ledger must be a regular file")
        value = _strict_json(self.path.read_text(encoding="utf-8"))
        if value.get("schema_version") != MONITORING_SCHEMA_VERSION:
            raise OperationalError(f"unsupported monitoring ledger schema {value.get('schema_version')!r}")
        unknown = value.keys() - {"schema_version", "observations", "assessments", "enforcements"}
        if unknown:
            raise OperationalError(f"monitoring ledger contains unknown fields: {sorted(unknown)}")
        try:
            raw_observations = value.get("observations", {})
            raw_assessments = value.get("assessments", {})
            raw_enforcements = value.get("enforcements", {})
            if not all(isinstance(item, dict) for item in (raw_observations, raw_assessments, raw_enforcements)):
                raise TypeError("monitoring record collections must be objects")
            self._observations = {key: self._observation(item) for key, item in raw_observations.items()}
            self._assessments = {key: self._assessment(item) for key, item in raw_assessments.items()}
            self._enforcements = {key: self._enforcement(item) for key, item in raw_enforcements.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationalError(f"monitoring ledger contains invalid records: {exc}") from exc
        if any(key != item.id for key, item in self._observations.items()):
            raise OperationalError("monitoring ledger observation identity mismatch")
        if any(key != item.id for key, item in self._assessments.items()):
            raise OperationalError("monitoring ledger assessment identity mismatch")
        if any(key != item.assessment_id for key, item in self._enforcements.items()):
            raise OperationalError("monitoring ledger enforcement identity mismatch")
        for assessment in self._assessments.values():
            try:
                referenced = [self._observations[identifier] for identifier in assessment.observation_ids]
            except KeyError as exc:
                raise OperationalError("monitoring assessment references an unknown observation") from exc
            if tuple(item.semantic_id for item in referenced) != assessment.observation_sha256s:
                raise OperationalError("monitoring assessment observation identity mismatch")
            created = _timestamp(assessment.created_at, "assessment timestamp")
            if any(
                not 0
                <= (created - _timestamp(item.observed_at, "observation timestamp")).total_seconds()
                <= assessment.policy.max_observation_age_seconds
                for item in referenced
            ):
                raise OperationalError("monitoring assessment contains an observation outside its policy age bound")

    def _write(self) -> None:
        rendered = canonical_json(
            {
                "schema_version": MONITORING_SCHEMA_VERSION,
                "observations": self._observations,
                "assessments": self._assessments,
                "enforcements": self._enforcements,
            }
        )
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def observe(self, observation: HealthObservation, registry: DeploymentRegistry) -> HealthObservation:
        deployment = registry.deployment(observation.alias)
        if (
            deployment.candidate_id != observation.candidate_id
            or deployment.sequence != observation.deployment_sequence
        ):
            raise OperationalError("health observation targets a stale or different deployment")
        existing = self._observations.get(observation.id)
        if existing is not None:
            if existing != observation:
                raise OperationalError("immutable observation id collision")
            return existing
        self._observations[observation.id] = observation
        self._write()
        return observation

    def assess(
        self,
        alias: str,
        registry: DeploymentRegistry,
        policy: MonitoringPolicy,
        *,
        created_at: str,
    ) -> HealthAssessment:
        if alias not in policy.aliases:
            raise OperationalError("deployment alias is outside the monitoring policy")
        deployment = registry.deployment(alias)
        assessment_time = _timestamp(created_at, "assessment timestamp")
        observations = [
            item
            for item in self._observations.values()
            if item.alias == alias
            and item.candidate_id == deployment.candidate_id
            and item.deployment_sequence == deployment.sequence
            and 0
            <= (assessment_time - _timestamp(item.observed_at, "observation timestamp")).total_seconds()
            <= policy.max_observation_age_seconds
        ]
        observations.sort(key=lambda item: (_timestamp(item.observed_at, "observation timestamp"), item.id))
        observations = observations[-policy.window_size :]
        metric_results: list[MetricAssessment] = []
        for threshold in policy.thresholds:
            missing = sum(threshold.name not in item.metrics for item in observations)
            breaches = missing + sum(
                threshold.breached(item.metrics[threshold.name])
                for item in observations
                if threshold.name in item.metrics
            )
            fraction = breaches / len(observations) if observations else 0.0
            metric_results.append(
                MetricAssessment(
                    threshold.name,
                    threshold.direction,
                    threshold.bound,
                    len(observations),
                    breaches,
                    missing,
                    fraction,
                )
            )
        if len(observations) < policy.minimum_samples:
            state = HealthState.INSUFFICIENT
        elif any(item.breach_fraction > policy.max_breach_fraction for item in metric_results):
            state = HealthState.UNHEALTHY
        else:
            state = HealthState.HEALTHY
        assessment = HealthAssessment(
            alias=alias,
            candidate_id=deployment.candidate_id or "",
            deployment_sequence=deployment.sequence,
            policy_id=policy.id,
            policy_version=policy.version,
            policy_sha256=policy.semantic_id,
            authorization_id=policy.authorized_by,
            policy=policy,
            state=state,
            action=policy.action if state is HealthState.UNHEALTHY else EnforcementAction.NONE,
            observation_ids=tuple(item.id for item in observations),
            observation_sha256s=tuple(item.semantic_id for item in observations),
            metrics=tuple(metric_results),
            created_at=created_at,
        )
        existing = self._assessments.get(assessment.id)
        if existing is not None:
            return existing
        self._assessments[assessment.id] = assessment
        self._write()
        return assessment

    def assessment(self, identifier: str) -> HealthAssessment:
        try:
            return self._assessments[identifier]
        except KeyError as exc:
            raise KeyError(identifier) from exc

    def _recover_enforcement(
        self,
        assessment: HealthAssessment,
        registry: DeploymentRegistry,
        *,
        actor: str,
        enforced_at: str,
    ) -> EnforcementReceipt | None:
        rollback = registry.receipt_for_assessment(assessment.id)
        quarantine = registry.quarantine_record(assessment.candidate_id)
        if rollback is None and (quarantine is None or quarantine.incident_id != assessment.id):
            return None
        if rollback is not None and (
            rollback.alias != assessment.alias
            or rollback.previous_candidate_id != assessment.candidate_id
            or rollback.policy_id != assessment.policy_id
            or rollback.authorization_id != assessment.authorization_id
        ):
            raise OperationalError("registry contains a conflicting enforcement receipt")
        if quarantine is None or (
            quarantine.incident_id != assessment.id
            or quarantine.policy_id != assessment.policy_id
            or quarantine.authorization_id != assessment.authorization_id
        ):
            raise OperationalError("registry contains incomplete or conflicting quarantine state")
        if assessment.action is EnforcementAction.ROLLBACK and rollback is None:
            return None
        return EnforcementReceipt(
            assessment.id,
            assessment.semantic_id,
            assessment.alias,
            assessment.candidate_id,
            assessment.action,
            assessment.id,
            assessment.policy_id,
            assessment.authorization_id,
            rollback.actor if rollback else quarantine.actor,
            quarantine.quarantined_at,
            rollback.sequence if rollback else None,
            rollback.candidate_id if rollback else None,
        )

    def enforce(
        self,
        assessment_id: str,
        registry: DeploymentRegistry,
        *,
        actor: str,
        enforced_at: str,
    ) -> EnforcementReceipt:
        _nonempty(actor, "enforcement actor")
        _timestamp(enforced_at, "enforcement timestamp")
        assessment = self.assessment(assessment_id)
        existing = self._enforcements.get(assessment_id)
        if existing is not None:
            return existing
        if assessment.state is not HealthState.UNHEALTHY or assessment.action is EnforcementAction.NONE:
            raise OperationalError("only a persisted unhealthy assessment may be enforced")
        recovered = self._recover_enforcement(
            assessment,
            registry,
            actor=actor,
            enforced_at=enforced_at,
        )
        if recovered is not None:
            self._enforcements[assessment_id] = recovered
            self._write()
            return recovered
        deployment = registry.deployment(assessment.alias, allow_quarantined=True)
        if (
            deployment.candidate_id != assessment.candidate_id
            or deployment.sequence != assessment.deployment_sequence
        ):
            raise OperationalError("health assessment targets a stale or different deployment")
        reason = f"deployment failed operational health assessment {assessment.id}"
        registry.quarantine(
            assessment.candidate_id,
            deployment_sequence=assessment.deployment_sequence,
            actor=actor,
            reason=reason,
            incident_id=assessment.id,
            policy_id=assessment.policy_id,
            authorization_id=assessment.authorization_id,
            quarantined_at=enforced_at,
        )
        rollback = None
        if assessment.action is EnforcementAction.ROLLBACK:
            registry.rollback_target(assessment.alias)
            rollback = registry.rollback(
                assessment.alias,
                actor=actor,
                reason=reason,
                assessment_id=assessment.id,
                policy_id=assessment.policy_id,
                authorization_id=assessment.authorization_id,
            )
        receipt = EnforcementReceipt(
            assessment.id,
            assessment.semantic_id,
            assessment.alias,
            assessment.candidate_id,
            assessment.action,
            assessment.id,
            assessment.policy_id,
            assessment.authorization_id,
            actor,
            enforced_at,
            rollback.sequence if rollback else None,
            rollback.candidate_id if rollback else None,
        )
        self._enforcements[assessment.id] = receipt
        self._write()
        return receipt

    @property
    def observations(self) -> tuple[HealthObservation, ...]:
        return tuple(self._observations[key] for key in sorted(self._observations))

    @property
    def assessments(self) -> tuple[HealthAssessment, ...]:
        return tuple(self._assessments[key] for key in sorted(self._assessments))

    @property
    def enforcements(self) -> tuple[EnforcementReceipt, ...]:
        return tuple(self._enforcements[key] for key in sorted(self._enforcements))


__all__ = [
    "MONITORING_SCHEMA_VERSION",
    "DeploymentMonitor",
    "EnforcementAction",
    "EnforcementReceipt",
    "HealthAssessment",
    "HealthObservation",
    "HealthState",
    "MetricAssessment",
    "MetricThreshold",
    "MonitoringPolicy",
    "ThresholdDirection",
]
