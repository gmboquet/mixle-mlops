from __future__ import annotations

import json
from collections.abc import Callable

from mixle_mlops.control import (
    DeploymentRegistry,
    EvidenceKind,
    EvidenceReceipt,
    LocalArtifactStore,
    ModelCandidate,
    OwnerScope,
    PromotionPolicy,
)
from mixle_mlops.control.integrity import IntegrityFinding, check_registry_integrity


def _owner() -> OwnerScope:
    return OwnerScope(organization_id="org", project_id="models")


def _policy() -> PromotionPolicy:
    return PromotionPolicy(
        trusted_factory_issuers=("factory",),
        trusted_harness_issuers=("harness",),
        required_suites=("quality", "calibration", "security"),
    )


def _candidate(
    artifacts: LocalArtifactStore,
    owner: OwnerScope,
    *,
    candidate_id: str,
    payload: bytes,
) -> tuple[ModelCandidate, tuple[EvidenceReceipt, ...]]:
    artifact = artifacts.put(owner, payload, media_type="application/octet-stream", semantic_type="model")
    candidate = ModelCandidate(
        id=candidate_id,
        model_id="science-model",
        version=candidate_id,
        artifact=artifact,
        factory_receipt_id=f"{candidate_id}-factory",
        harness_receipt_ids=(f"{candidate_id}-harness",),
    )
    evidence = (
        EvidenceReceipt(
            id=f"{candidate_id}-factory",
            kind=EvidenceKind.FACTORY,
            issuer="factory",
            subject_sha256=artifact.sha256,
            passed=True,
            suites=("build",),
        ),
        EvidenceReceipt(
            id=f"{candidate_id}-harness",
            kind=EvidenceKind.HARNESS,
            issuer="harness",
            subject_sha256=artifact.sha256,
            passed=True,
            suites=("quality", "calibration", "security"),
        ),
    )
    return candidate, evidence


def _rewrite(registry: DeploymentRegistry, mutate: Callable[[dict], None]) -> None:
    """Hand-edit the durable JSON file the way an operator or a partial write might, then reload."""
    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    mutate(payload)
    registry.path.write_text(json.dumps(payload), encoding="utf-8")


def test_empty_registry_is_clean(tmp_path) -> None:
    registry = DeploymentRegistry(tmp_path / "registry")

    report = check_registry_integrity(registry)

    assert report.clean
    assert report.checked_candidates == 0
    assert report.checked_receipts == 0
    assert report.checked_aliases == 0
    assert report.checked_artifacts is None


def test_clean_promotion_history_reports_no_issues(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")

    first, first_evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(first)
    registry.promote(first.id, "stage", first_evidence, policy, actor="release-manager")

    second, second_evidence = _candidate(artifacts, owner, candidate_id="candidate-2", payload=b"model-v2")
    registry.register(second)
    registry.promote(second.id, "production", second_evidence, policy, actor="release-manager")

    report = check_registry_integrity(registry, artifacts=artifacts)

    assert report.clean
    assert report.checked_candidates == 2
    assert report.checked_receipts == 2
    assert report.checked_aliases == 2
    assert report.checked_artifacts == 2


def test_promotion_then_rollback_history_replays_cleanly(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")

    first, first_evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(first)
    registry.promote(first.id, "production", first_evidence, policy, actor="release-manager")

    second, second_evidence = _candidate(artifacts, owner, candidate_id="candidate-2", payload=b"model-v2")
    registry.register(second)
    registry.promote(second.id, "production", second_evidence, policy, actor="release-manager")

    registry.mark_unhealthy("production", actor="monitor", incident_id="incident-1")

    report = check_registry_integrity(registry, artifacts=artifacts)

    assert report.clean
    assert report.checked_receipts == 3
    assert registry.resolve("production") == first


def test_dangling_alias_is_detected(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(candidate)
    registry.promote(candidate.id, "stage", evidence, policy, actor="release-manager")

    def mutate(payload: dict) -> None:
        payload["aliases"]["stage"] = "candidate-does-not-exist"

    _rewrite(registry, mutate)
    reloaded = DeploymentRegistry(tmp_path / "registry")

    report = check_registry_integrity(reloaded)

    dangling = report.issues_of(IntegrityFinding.DANGLING_ALIAS)
    assert len(dangling) == 1
    assert dangling[0].subject == "stage"


def test_dangling_previous_is_detected_independently_of_dangling_alias(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    first, first_evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(first)
    registry.promote(first.id, "production", first_evidence, policy, actor="release-manager")
    second, second_evidence = _candidate(artifacts, owner, candidate_id="candidate-2", payload=b"model-v2")
    registry.register(second)
    registry.promote(second.id, "production", second_evidence, policy, actor="release-manager")

    def mutate(payload: dict) -> None:
        payload["previous"]["production"] = "candidate-does-not-exist"

    _rewrite(registry, mutate)
    reloaded = DeploymentRegistry(tmp_path / "registry")

    report = check_registry_integrity(reloaded)

    dangling_previous = report.issues_of(IntegrityFinding.DANGLING_PREVIOUS)
    assert len(dangling_previous) == 1
    assert dangling_previous[0].subject == "production"
    # the live alias itself still points at a real candidate -- this is the only structural issue
    assert not report.issues_of(IntegrityFinding.DANGLING_ALIAS)


def test_projection_drift_is_detected_without_a_dangling_pointer(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    first, first_evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(first)
    registry.promote(first.id, "stage", first_evidence, policy, actor="release-manager")
    # a second, real candidate that was never promoted anywhere
    second, _second_evidence = _candidate(artifacts, owner, candidate_id="candidate-2", payload=b"model-v2")
    registry.register(second)

    def mutate(payload: dict) -> None:
        payload["aliases"]["stage"] = "candidate-2"

    _rewrite(registry, mutate)
    reloaded = DeploymentRegistry(tmp_path / "registry")

    report = check_registry_integrity(reloaded)

    assert not report.issues_of(IntegrityFinding.DANGLING_ALIAS)
    drift = report.issues_of(IntegrityFinding.PROJECTION_DRIFT)
    assert len(drift) == 1
    assert drift[0].subject == "aliases[stage]"
    assert "candidate-1" in drift[0].detail
    assert "candidate-2" in drift[0].detail


def test_receipt_sequence_gap_and_duplicate_are_detected(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    first, first_evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(first)
    registry.promote(first.id, "production", first_evidence, policy, actor="release-manager")
    second, second_evidence = _candidate(artifacts, owner, candidate_id="candidate-2", payload=b"model-v2")
    registry.register(second)
    registry.promote(second.id, "production", second_evidence, policy, actor="release-manager")
    # two receipts now exist on disk with sequence 0 and 1

    def mutate(payload: dict) -> None:
        payload["receipts"][1]["sequence"] = 0

    _rewrite(registry, mutate)
    reloaded = DeploymentRegistry(tmp_path / "registry")

    report = check_registry_integrity(reloaded)

    duplicates = report.issues_of(IntegrityFinding.RECEIPT_SEQUENCE_DUPLICATE)
    gaps = report.issues_of(IntegrityFinding.RECEIPT_SEQUENCE_GAP)
    assert [issue.subject for issue in duplicates] == ["0"]
    assert [issue.subject for issue in gaps] == ["1"]


def test_receipt_naming_unregistered_candidate_is_detected(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(candidate)
    registry.promote(candidate.id, "stage", evidence, policy, actor="release-manager")

    def mutate(payload: dict) -> None:
        payload["receipts"][0]["candidate_id"] = "candidate-ghost"
        # keep the alias in step with the corrupted receipt so replay still agrees with live state --
        # isolates the unknown-candidate check from projection drift, which is a separate concern.
        payload["aliases"]["stage"] = "candidate-ghost"

    _rewrite(registry, mutate)
    reloaded = DeploymentRegistry(tmp_path / "registry")

    report = check_registry_integrity(reloaded)

    unknown = report.issues_of(IntegrityFinding.RECEIPT_UNKNOWN_CANDIDATE)
    assert len(unknown) == 1
    assert unknown[0].subject == "receipt[0].candidate_id"
    assert not report.issues_of(IntegrityFinding.PROJECTION_DRIFT)
    # the alias now agrees with the corrupted receipt log, so it is *also* a dangling alias --
    # both findings are correct, independent observations about the same corruption.
    assert report.issues_of(IntegrityFinding.DANGLING_ALIAS)


def test_unknown_receipt_action_is_reported(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(candidate)
    registry.promote(candidate.id, "stage", evidence, policy, actor="release-manager")

    def mutate(payload: dict) -> None:
        payload["receipts"][0]["action"] = "demote"

    _rewrite(registry, mutate)
    reloaded = DeploymentRegistry(tmp_path / "registry")

    report = check_registry_integrity(reloaded)

    unknown_action = report.issues_of(IntegrityFinding.RECEIPT_UNKNOWN_ACTION)
    assert len(unknown_action) == 1
    assert unknown_action[0].subject == "receipt[0]"


def test_missing_artifact_is_detected_when_a_store_is_checked(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(candidate)
    registry.promote(candidate.id, "stage", evidence, policy, actor="release-manager")

    empty_artifacts = LocalArtifactStore(tmp_path / "empty-artifacts")

    report = check_registry_integrity(registry, artifacts=empty_artifacts)

    missing = report.issues_of(IntegrityFinding.ARTIFACT_MISSING)
    assert len(missing) == 1
    assert missing[0].subject == "candidate-1"
    assert report.checked_artifacts == 1


def test_artifact_digest_mismatch_is_detected(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(candidate)
    registry.promote(candidate.id, "stage", evidence, policy, actor="release-manager")

    [stored_path] = list(artifacts.root.rglob(candidate.artifact.sha256))
    stored_path.write_bytes(b"a completely different payload of a different length")

    report = check_registry_integrity(registry, artifacts=artifacts)

    mismatched = report.issues_of(IntegrityFinding.ARTIFACT_DIGEST_MISMATCH)
    assert len(mismatched) == 1
    assert mismatched[0].subject == "candidate-1"


def test_artifact_verification_is_skipped_when_no_store_is_given(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(candidate)
    registry.promote(candidate.id, "stage", evidence, policy, actor="release-manager")

    report = check_registry_integrity(registry)

    assert report.checked_artifacts is None
    assert report.clean


def test_registry_read_only_accessors_are_copies(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    owner = _owner()
    policy = _policy()
    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, evidence = _candidate(artifacts, owner, candidate_id="candidate-1", payload=b"model-v1")
    registry.register(candidate)
    registry.promote(candidate.id, "stage", evidence, policy, actor="release-manager")

    aliases = registry.aliases
    aliases["stage"] = "tampered"

    assert registry.resolve("stage") == candidate
    assert registry.aliases["stage"] == candidate.id
