from __future__ import annotations

from dataclasses import replace

import pytest

from mixle_mlops.control import (
    CapabilityRef,
    DurableLocalRunner,
    InvocationSpec,
    JobSpec,
    JobState,
    LocalArtifactStore,
    OwnerScope,
    ResourceLimits,
    RetryPolicy,
)
from mixle_mlops.control.contracts import OperationalError


def owner(project: str = "project-a") -> OwnerScope:
    return OwnerScope("org-1", project)


def job(store: LocalArtifactStore, *, identifier="job-1", key="key-1", parameter=1) -> JobSpec:
    input_artifact = store.put(owner(), b"input", media_type="application/json", semantic_type="inquiry-action")
    return JobSpec(
        id=identifier,
        owner=owner(),
        invocation=InvocationSpec(
            CapabilityRef("mixle.inquiry.action", "1", "input/v1", "result/v1"),
            (input_artifact,),
            {"parameter": parameter},
            knowledge_snapshot_id="knowledge-1",
            seed=7,
            policy_id="policy-1",
        ),
        resources=ResourceLimits(60, 1_000_000, 60, 1_000_000, event_count=100),
        retry=RetryPolicy(2, ("worker_lost",)),
        idempotency_key=key,
    )


def test_idempotency_is_owner_scoped_and_semantic(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    first = runner.submit(job(artifacts))
    duplicate = runner.submit(job(artifacts, identifier="different-operational-id"))
    assert duplicate.spec.id == first.spec.id
    with pytest.raises(OperationalError, match="different semantic"):
        runner.submit(job(artifacts, identifier="other", parameter=2))


def test_owner_isolation_blocks_job_and_artifact_enumeration(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    record = runner.submit(job(artifacts))
    with pytest.raises(PermissionError):
        runner.get(record.spec.id, owner("project-b"))
    with pytest.raises(PermissionError):
        artifacts.get(owner("project-b"), record.spec.invocation.inputs[0])


def test_claim_start_progress_checkpoint_and_complete_are_durable(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    root = tmp_path / "runner"
    runner = DurableLocalRunner(root, artifact_store=artifacts)
    spec = job(artifacts)
    runner.submit(spec)
    record, lease = runner.claim("worker-1")
    assert record.state is JobState.LEASED and record.attempt == 1
    runner.start(spec.id, lease.token)
    runner.progress(spec.id, lease.token, fraction=0.5)
    checkpoint = runner.checkpoint(spec.id, lease.token, b"checkpoint")
    result = runner.complete(
        spec.id, lease.token, b"result", media_type="application/json", semantic_type="data-result"
    )
    restored = DurableLocalRunner(root, artifact_store=artifacts).get(spec.id, owner())
    assert restored.state is JobState.SUCCEEDED
    assert restored.checkpoints[0].sha256 == checkpoint.sha256
    assert restored.results[0].sha256 == result.sha256
    succeeded = restored.events[-1]
    assert succeeded["payload"]["epistemic_disposition"] == "not_evaluated"


def test_duplicate_delivery_and_invalid_worker_transition_fail(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job(artifacts)
    runner.submit(spec)
    _record, lease = runner.claim("worker-1")
    assert runner.claim("worker-2") is None
    with pytest.raises(OperationalError, match="lease"):
        runner.start(spec.id, "wrong-token")
    runner.start(spec.id, lease.token)
    with pytest.raises(OperationalError, match="lease"):
        runner.start(spec.id, lease.token)


def test_lease_expiry_requeues_and_preserves_semantic_identity(tmp_path) -> None:
    now = [100.0]

    def clock():
        return now[0]

    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts, clock=clock)
    spec = job(artifacts)
    runner.submit(spec)
    _record, first_lease = runner.claim("dead-worker", lease_seconds=5)
    runner.start(spec.id, first_lease.token)
    runner.checkpoint(spec.id, first_lease.token, b"resume-state")
    now[0] = 106
    assert runner.recover_expired() == (spec.id,)
    recovered, second_lease = runner.claim("replacement", lease_seconds=5)
    assert recovered.attempt == 2
    assert recovered.spec.semantic_id == spec.semantic_id
    assert second_lease.token != first_lease.token


def test_retry_policy_and_terminal_lease_expiry(tmp_path) -> None:
    now = [0.0]
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts, clock=lambda: now[0])
    spec = job(artifacts)
    runner.submit(spec)
    _record, lease = runner.claim("w", lease_seconds=1)
    runner.start(spec.id, lease.token)
    assert runner.fail(spec.id, lease.token, code="worker_lost", detail="crash").state is JobState.QUEUED
    _record, lease = runner.claim("w2", lease_seconds=1)
    now[0] = 2
    runner.recover_expired()
    assert runner.get(spec.id, owner()).state is JobState.FAILED


def test_cancellation_race_cannot_publish_a_result(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job(artifacts)
    runner.submit(spec)
    _record, lease = runner.claim("worker")
    runner.start(spec.id, lease.token)
    runner.request_cancel(spec.id, owner(), reason="no longer needed")
    with pytest.raises(OperationalError, match="cannot be completed"):
        runner.complete(spec.id, lease.token, b"late", media_type="text/plain", semantic_type="result")
    assert runner.acknowledge_cancel(spec.id, lease.token).state is JobState.CANCELLED


def test_checkpoint_tampering_is_detected(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job(artifacts)
    runner.submit(spec)
    _record, lease = runner.claim("worker")
    runner.start(spec.id, lease.token)
    checkpoint = runner.checkpoint(spec.id, lease.token, b"good")
    path = tmp_path / "artifacts" / "org-1" / "project-a" / checkpoint.sha256[:2] / checkpoint.sha256
    path.write_bytes(b"tampered")
    with pytest.raises(OperationalError, match="verification"):
        artifacts.get(owner(), checkpoint)


def test_job_inputs_cannot_cross_owner_scope(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    spec = job(artifacts)
    with pytest.raises(OperationalError, match="job owner"):
        replace(spec, owner=owner("project-b"))
