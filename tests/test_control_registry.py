from __future__ import annotations

from dataclasses import replace

import pytest

from mixle_mlops.control import (
    DeploymentRegistry,
    EvidenceKind,
    EvidenceReceipt,
    LocalArtifactStore,
    ModelCandidate,
    OwnerScope,
    PromotionPolicy,
)
from mixle_mlops.control.contracts import OperationalError


def setup(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = OwnerScope("org", "models")
    artifact = artifacts.put(owner, b"model-v1", media_type="application/octet-stream", semantic_type="model")
    candidate = ModelCandidate("candidate-1", "science-model", "1", artifact, "factory-1", ("harness-1",))
    policy = PromotionPolicy(("factory",), ("harness",), ("quality", "calibration", "security"))
    evidence = (
        EvidenceReceipt("factory-1", EvidenceKind.FACTORY, "factory", artifact.sha256, True, ("build",)),
        EvidenceReceipt(
            "harness-1",
            EvidenceKind.HARNESS,
            "harness",
            artifact.sha256,
            True,
            ("quality", "calibration", "security"),
        ),
    )
    registry = DeploymentRegistry(tmp_path / "registry")
    registry.register(candidate)
    return registry, candidate, policy, evidence, artifacts, owner


def test_promotion_requires_exact_trusted_factory_and_harness_evidence(tmp_path) -> None:
    registry, candidate, policy, evidence, _artifacts, _owner = setup(tmp_path)
    receipt = registry.promote(candidate.id, "stage", evidence, policy, actor="release-manager")
    assert receipt.candidate_id == candidate.id
    assert registry.resolve("stage") == candidate
    restored = DeploymentRegistry(tmp_path / "registry")
    assert restored.resolve("stage") == candidate


def test_forged_subject_failed_evidence_and_missing_suite_are_rejected(tmp_path) -> None:
    registry, candidate, policy, evidence, _artifacts, _owner = setup(tmp_path)
    forged = (evidence[0], replace(evidence[1], subject_sha256="f" * 64))
    with pytest.raises(OperationalError, match="different artifact"):
        registry.promote(candidate.id, "stage", forged, policy, actor="x")
    failed = (evidence[0], replace(evidence[1], passed=False))
    with pytest.raises(OperationalError, match="failed"):
        registry.promote(candidate.id, "stage", failed, policy, actor="x")
    missing = (evidence[0], replace(evidence[1], suites=("quality",)))
    with pytest.raises(OperationalError, match="suites"):
        registry.promote(candidate.id, "stage", missing, policy, actor="x")


def test_untrusted_issuer_and_unknown_alias_are_rejected(tmp_path) -> None:
    registry, candidate, policy, evidence, _artifacts, _owner = setup(tmp_path)
    with pytest.raises(OperationalError, match="trusted"):
        registry.promote(
            candidate.id, "stage", (replace(evidence[0], issuer="unknown"), evidence[1]), policy, actor="x"
        )
    with pytest.raises(OperationalError, match="alias"):
        registry.promote(candidate.id, "latest", evidence, policy, actor="x")


def test_second_promotion_and_forced_failure_roll_back_exact_candidate(tmp_path) -> None:
    registry, first, policy, first_evidence, artifacts, owner = setup(tmp_path)
    registry.promote(first.id, "production", first_evidence, policy, actor="release-manager")
    artifact = artifacts.put(owner, b"model-v2", media_type="application/octet-stream", semantic_type="model")
    second = ModelCandidate("candidate-2", "science-model", "2", artifact, "factory-2", ("harness-2",))
    evidence = (
        EvidenceReceipt("factory-2", EvidenceKind.FACTORY, "factory", artifact.sha256, True, ("build",)),
        EvidenceReceipt(
            "harness-2",
            EvidenceKind.HARNESS,
            "harness",
            artifact.sha256,
            True,
            ("quality", "calibration", "security"),
        ),
    )
    registry.register(second)
    registry.promote(second.id, "production", evidence, policy, actor="release-manager")
    rollback = registry.mark_unhealthy("production", actor="monitor", incident_id="incident-1")
    assert rollback.action == "rollback"
    assert registry.resolve("production") == first


def test_candidate_identity_is_immutable_and_rollback_needs_history(tmp_path) -> None:
    registry, candidate, _policy, _evidence, _artifacts, _owner = setup(tmp_path)
    with pytest.raises(OperationalError, match="collision"):
        registry.register(replace(candidate, version="different"))
    with pytest.raises(OperationalError, match="rollback"):
        registry.rollback("production", actor="x", reason="none")
