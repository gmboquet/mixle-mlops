from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mixle_mlops.control import (
    DeploymentMonitor,
    DeploymentRegistry,
    EnforcementAction,
    EvidenceKind,
    EvidenceReceipt,
    HealthObservation,
    HealthState,
    LocalArtifactStore,
    MetricThreshold,
    ModelCandidate,
    MonitoringPolicy,
    OwnerScope,
    PromotionPolicy,
    ThresholdDirection,
)
from mixle_mlops.control.contracts import OperationalError

NOW = "2026-07-15T12:00:00Z"


def candidate(artifacts, owner, number: int):
    artifact = artifacts.put(
        owner,
        f"model-v{number}".encode(),
        media_type="application/octet-stream",
        semantic_type="model",
    )
    value = ModelCandidate(
        f"candidate-{number}",
        "science-model",
        str(number),
        artifact,
        f"factory-{number}",
        (f"harness-{number}",),
    )
    evidence = (
        EvidenceReceipt(
            f"factory-{number}",
            EvidenceKind.FACTORY,
            "factory",
            artifact.sha256,
            True,
            ("build",),
        ),
        EvidenceReceipt(
            f"harness-{number}",
            EvidenceKind.HARNESS,
            "harness",
            artifact.sha256,
            True,
            ("quality", "calibration", "security"),
        ),
    )
    return value, evidence


def deployment(tmp_path, *, versions: int = 2):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    registry = DeploymentRegistry(tmp_path / "registry")
    owner = OwnerScope("org", "models")
    promotion = PromotionPolicy(("factory",), ("harness",), ("quality", "calibration", "security"))
    values = []
    for number in range(1, versions + 1):
        model, evidence = candidate(artifacts, owner, number)
        registry.register(model)
        receipt = registry.promote(model.id, "production", evidence, promotion, actor="release-manager")
        values.append((model, evidence, receipt))
    return registry, promotion, values


def policy(*, action: EnforcementAction = EnforcementAction.ROLLBACK) -> MonitoringPolicy:
    return MonitoringPolicy(
        "production-health",
        "1",
        (
            MetricThreshold("error_rate", ThresholdDirection.MAXIMUM, 0.1),
            MetricThreshold("latency_seconds", ThresholdDirection.MAXIMUM, 1.0),
        ),
        "authorization:ops-2026-07",
        window_size=3,
        minimum_samples=2,
        action=action,
    )


def observation(identifier: str, receipt, *, metrics=None, observed_at=NOW) -> HealthObservation:
    return HealthObservation(
        identifier,
        "production",
        receipt.candidate_id,
        receipt.sequence,
        observed_at,
        "gateway:production",
        metrics or {"latency_seconds": 0.2, "error_rate": 0.0},
    )


def test_healthy_windows_are_deployment_bound_immutable_and_durable(tmp_path) -> None:
    registry, _promotion, values = deployment(tmp_path)
    receipt = values[-1][2]
    monitor = DeploymentMonitor(tmp_path / "monitor")
    first = observation("sample-1", receipt, observed_at="2026-07-15T11:59:00Z")
    second = observation("sample-2", receipt)
    future = observation("future", receipt, observed_at="2026-07-15T12:01:00Z")
    assert monitor.observe(first, registry) == first
    assert monitor.observe(first, registry) == first
    monitor.observe(second, registry)
    monitor.observe(future, registry)

    assessment = monitor.assess("production", registry, policy(), created_at=NOW)
    assert assessment.state is HealthState.HEALTHY
    assert assessment.action is EnforcementAction.NONE
    assert assessment.observation_ids == ("sample-1", "sample-2")
    assert all(item.breaches == 0 and item.missing == 0 for item in assessment.metrics)

    restored = DeploymentMonitor(tmp_path / "monitor")
    assert restored.assessment(assessment.id) == assessment
    assert restored.observations == (future, first, second)
    with pytest.raises(TypeError, match="immutable"):
        first.metrics["latency_seconds"] = 9.0
    with pytest.raises(OperationalError, match="collision"):
        restored.observe(replace(first, source="different"), registry)
    with pytest.raises(OperationalError, match="stale"):
        restored.observe(replace(first, id="stale", deployment_sequence=receipt.sequence - 1), registry)
    with pytest.raises(OperationalError, match="finite"):
        replace(first, id="nan", metrics={"latency_seconds": float("nan")})

    ledger = tmp_path / "monitor" / "monitoring.json"
    tampered = json.loads(ledger.read_text())
    tampered["observations"]["sample-1"]["metrics"]["latency_seconds"] = 99.0
    ledger.write_text(json.dumps(tampered))
    with pytest.raises(OperationalError, match="observation identity"):
        DeploymentMonitor(tmp_path / "monitor")


def test_missing_or_breached_metrics_quarantine_and_roll_back_exact_candidate(tmp_path) -> None:
    registry, promotion, values = deployment(tmp_path)
    first, second = values[0][0], values[1][0]
    receipt = values[-1][2]
    monitor = DeploymentMonitor(tmp_path / "monitor")
    monitor.observe(
        observation("bad-1", receipt, metrics={"latency_seconds": 2.0}),
        registry,
    )
    monitor.observe(observation("bad-2", receipt), registry)
    assessment = monitor.assess("production", registry, policy(), created_at=NOW)
    assert assessment.state is HealthState.UNHEALTHY
    assert {item.name: (item.breaches, item.missing) for item in assessment.metrics} == {
        "error_rate": (1, 1),
        "latency_seconds": (1, 0),
    }

    enforced = monitor.enforce(assessment.id, registry, actor="monitor", enforced_at=NOW)
    assert enforced.rollback_candidate_id == first.id
    assert registry.resolve("production") == first
    assert registry.quarantine_record(second.id).incident_id == assessment.id
    assert monitor.enforce(assessment.id, registry, actor="retry", enforced_at=NOW) == enforced
    with pytest.raises(OperationalError, match="quarantined"):
        registry.promote(second.id, "production", values[1][1], promotion, actor="release-manager")

    assert DeploymentRegistry(tmp_path / "registry").resolve("production") == first
    restored = DeploymentMonitor(tmp_path / "monitor")
    assert restored.enforcements == (enforced,)


def test_insufficient_and_stale_assessments_fail_closed(tmp_path) -> None:
    registry, _promotion, values = deployment(tmp_path)
    receipt = values[-1][2]
    monitor = DeploymentMonitor(tmp_path / "monitor")
    monitor.observe(observation("only", receipt, metrics={"latency_seconds": 4.0}), registry)
    insufficient = monitor.assess("production", registry, policy(), created_at=NOW)
    assert insufficient.state is HealthState.INSUFFICIENT
    with pytest.raises(OperationalError, match="only a persisted unhealthy"):
        monitor.enforce(insufficient.id, registry, actor="monitor", enforced_at=NOW)
    with pytest.raises(KeyError):
        monitor.enforce("unknown", registry, actor="monitor", enforced_at=NOW)

    monitor.observe(observation("second", receipt, metrics={"latency_seconds": 4.0}), registry)
    unhealthy = monitor.assess("production", registry, policy(), created_at=NOW)
    registry.rollback("production", actor="operator", reason="manual intervention")
    with pytest.raises(OperationalError, match="stale"):
        monitor.enforce(unhealthy.id, registry, actor="monitor", enforced_at=NOW)


def test_quarantine_only_policy_disables_an_unhealthy_deployment_without_history(tmp_path) -> None:
    registry, _promotion, values = deployment(tmp_path, versions=1)
    receipt = values[-1][2]
    monitor = DeploymentMonitor(tmp_path / "monitor")
    for number in range(2):
        monitor.observe(
            observation(f"failure-{number}", receipt, metrics={"latency_seconds": 3.0, "error_rate": 1.0}),
            registry,
        )
    assessment = monitor.assess(
        "production",
        registry,
        policy(action=EnforcementAction.QUARANTINE),
        created_at=NOW,
    )
    enforced = monitor.enforce(assessment.id, registry, actor="monitor", enforced_at=NOW)
    assert enforced.rollback_sequence is None
    assert (
        registry.quarantine_record(receipt.candidate_id).authorization_id
        == policy(action=EnforcementAction.QUARANTINE).authorized_by
    )
    with pytest.raises(OperationalError, match="quarantined"):
        registry.resolve("production")

    fallback_registry, _promotion, fallback_values = deployment(tmp_path / "rollback-without-history", versions=1)
    fallback_receipt = fallback_values[-1][2]
    fallback_monitor = DeploymentMonitor(tmp_path / "rollback-without-history" / "monitor")
    for number in range(2):
        fallback_monitor.observe(
            observation(
                f"rollback-failure-{number}",
                fallback_receipt,
                metrics={"latency_seconds": 3.0, "error_rate": 1.0},
            ),
            fallback_registry,
        )
    fallback_assessment = fallback_monitor.assess("production", fallback_registry, policy(), created_at=NOW)
    with pytest.raises(OperationalError, match="no rollback candidate"):
        fallback_monitor.enforce(fallback_assessment.id, fallback_registry, actor="monitor", enforced_at=NOW)
    with pytest.raises(OperationalError, match="quarantined"):
        fallback_registry.resolve("production")


def test_interrupted_registry_enforcement_is_recovered_without_a_second_rollback(tmp_path) -> None:
    registry, _promotion, values = deployment(tmp_path)
    receipt = values[-1][2]
    monitor = DeploymentMonitor(tmp_path / "monitor")
    for number in range(2):
        monitor.observe(
            observation(f"failure-{number}", receipt, metrics={"latency_seconds": 3.0, "error_rate": 1.0}),
            registry,
        )
    assessment = monitor.assess("production", registry, policy(), created_at=NOW)
    reason = f"deployment failed operational health assessment {assessment.id}"
    registry.quarantine(
        assessment.candidate_id,
        deployment_sequence=assessment.deployment_sequence,
        actor="original-monitor",
        reason=reason,
        incident_id=assessment.id,
        policy_id=assessment.policy_id,
        authorization_id=assessment.authorization_id,
        quarantined_at="2026-07-15T12:01:00Z",
    )
    rollback = registry.rollback(
        assessment.alias,
        actor="original-monitor",
        reason=reason,
        assessment_id=assessment.id,
        policy_id=assessment.policy_id,
        authorization_id=assessment.authorization_id,
    )

    recovered = DeploymentMonitor(tmp_path / "monitor").enforce(
        assessment.id,
        DeploymentRegistry(tmp_path / "registry"),
        actor="retry-monitor",
        enforced_at="2026-07-15T12:02:00Z",
    )
    assert recovered.rollback_sequence == rollback.sequence
    assert recovered.actor == "original-monitor"
    assert recovered.enforced_at == "2026-07-15T12:01:00Z"


def test_monitoring_contracts_reject_ambiguous_or_unbounded_policy() -> None:
    threshold = MetricThreshold("latency", ThresholdDirection.MAXIMUM, 1.0)
    with pytest.raises(OperationalError, match="unique"):
        MonitoringPolicy("p", "1", (threshold, threshold), "auth")
    with pytest.raises(OperationalError, match="window"):
        MonitoringPolicy("p", "1", (threshold,), "auth", window_size=1, minimum_samples=2)
    with pytest.raises(OperationalError, match="age"):
        MonitoringPolicy("p", "1", (threshold,), "auth", max_observation_age_seconds=float("inf"))
    with pytest.raises(OperationalError, match="authorization"):
        MonitoringPolicy("p", "1", (threshold,), "")
    with pytest.raises(OperationalError, match="timezone"):
        HealthObservation("o", "production", "c", 0, "2026-07-15T12:00:00", "source", {"latency": 1.0})
