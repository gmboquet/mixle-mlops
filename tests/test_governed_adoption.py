import pytest

from mixle_mlops.control import (
    ArchitectureEpochPin,
    ArtifactRef,
    EvaluationAttestation,
    EvidenceKind,
    EvidenceReceipt,
    GovernedAdoptionPolicy,
    GovernedDeploymentRegistry,
    ModelCandidate,
    OperationalError,
    OwnerScope,
    PromotionPolicy,
)


def _candidate(epoch="epoch-0"):
    artifact = ArtifactRef(
        OwnerScope("org", "project"),
        "a" * 64,
        10,
        "application/octet-stream",
        "blob://a",
        "model",
    )
    return ModelCandidate(
        "candidate-1",
        "capability.solver",
        "0.1.0",
        artifact,
        "factory-1",
        ("harness-1",),
        {"architecture_epoch_id": epoch},
    )


def _evidence():
    return (
        EvidenceReceipt("factory-1", EvidenceKind.FACTORY, "factory", "a" * 64, True, ("build",)),
        EvidenceReceipt("harness-1", EvidenceKind.HARNESS, "harness", "a" * 64, True, ("heldout",)),
    )


def _evaluation(**overrides):
    values = {
        "id": "evaluation-1",
        "candidate_id": "candidate-1",
        "artifact_digest": "a" * 64,
        "recommendation": "accept",
        "evaluator_project": "PRJ-HARNESS",
        "builder_project": "PRJ-AIFACTORY",
        "signature": "signed",
    }
    values.update(overrides)
    return EvaluationAttestation(**values)


def _authorization(**overrides):
    values = {
        "decision_id": "auth-1",
        "capability": {"capability_id": "capability.solver", "version": "0.1.0", "digest": "a" * 64},
        "outcome": "granted",
        "issued_by": "release-authority",
        "scopes": ["deploy:production"],
        "decided_at": "2026-01-01T00:00:00Z",
        "expires_at": "2026-02-01T00:00:00Z",
        "revoked_at": None,
        "revoked_by": None,
    }
    values.update(overrides)
    return values


def _epoch(identifier="epoch-0"):
    return ArchitectureEpochPin(identifier, 0, "b" * 64, "architecture-authority", "2026-01-01T00:00:00Z")


def _policy():
    return GovernedAdoptionPolicy(
        PromotionPolicy(("factory",), ("harness",), ("heldout",), aliases=("production",)),
        "deploy:production",
        ("PRJ-HARNESS",),
        ("release-authority",),
        ("architecture-authority",),
    )


def _registry(tmp_path, candidate=None):
    registry = GovernedDeploymentRegistry(tmp_path)
    registry.deployments.register(candidate or _candidate())
    return registry


def test_adoption_requires_acceptance_authorization_and_epoch_pin(tmp_path):
    registry = _registry(tmp_path)
    receipt = registry.adopt(
        "candidate-1",
        "production",
        _evidence(),
        _evaluation(),
        _authorization(),
        _epoch(),
        _policy(),
        actor="operator",
        verify_evaluation=lambda evaluation: evaluation.signature == "signed",
        adopted_at="2026-01-02T00:00:00Z",
    )
    assert receipt.authorization_id == "auth-1"
    assert receipt.architecture_epoch_id == "epoch-0"
    assert registry.deployments.resolve("production").id == "candidate-1"
    assert GovernedDeploymentRegistry(tmp_path).receipts == (receipt,)


@pytest.mark.parametrize(
    ("evaluation", "authorization", "epoch", "verifier", "message"),
    [
        (_evaluation(recommendation="reject"), _authorization(), _epoch(), lambda _: True, "did not recommend"),
        (_evaluation(), _authorization(expires_at="2026-01-01T12:00:00Z"), _epoch(), lambda _: True, "expired"),
        (
            _evaluation(),
            _authorization(revoked_at="2026-01-01T12:00:00Z", revoked_by="authority"),
            _epoch(),
            lambda _: True,
            "revoked",
        ),
        (_evaluation(), _authorization(), _epoch(), lambda _: False, "signature"),
        (_evaluation(builder_project="PRJ-HARNESS"), _authorization(), _epoch(), lambda _: True, "must be distinct"),
    ],
)
def test_missing_or_invalid_gate_fails_before_deployment(
    tmp_path, evaluation, authorization, epoch, verifier, message
):
    registry = _registry(tmp_path)
    with pytest.raises(OperationalError, match=message):
        registry.adopt(
            "candidate-1",
            "production",
            _evidence(),
            evaluation,
            authorization,
            epoch,
            _policy(),
            actor="operator",
            verify_evaluation=verifier,
            adopted_at="2026-01-02T00:00:00Z",
        )
    with pytest.raises(KeyError):
        registry.deployments.resolve("production")


def test_candidate_cannot_move_to_a_different_architecture_epoch(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(OperationalError, match="not pinned"):
        registry.adopt(
            "candidate-1",
            "production",
            _evidence(),
            _evaluation(),
            _authorization(),
            _epoch("epoch-1"),
            _policy(),
            actor="operator",
            verify_evaluation=lambda _: True,
            adopted_at="2026-01-02T00:00:00Z",
        )
