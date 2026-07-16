"""The worker harness: the first code that actually drives DurableLocalRunner jobs to
completion, instead of every caller hand-driving claim/start/heartbeat/complete/fail."""

from __future__ import annotations

import hashlib

import pytest

from mixle_mlops.control import (
    CapabilityRef,
    CooperativeCancellation,
    DurableLocalRunner,
    HandlerFailure,
    HandlerResult,
    InvocationSpec,
    JobSpec,
    JobState,
    LocalArtifactStore,
    OwnerScope,
    ResourceLimits,
    RetryPolicy,
    WorkerContext,
    WorkOutcome,
    drain,
    run_forever,
    run_once,
)
from mixle_mlops.control.contracts import OperationalError


def owner(project: str = "project-a") -> OwnerScope:
    return OwnerScope(organization_id="org-1", project_id=project)


def job_spec(
    store: LocalArtifactStore,
    *,
    identifier: str = "job-1",
    key: str | None = "key-1",
    max_attempts: int = 1,
    retryable_codes: tuple[str, ...] = (),
    output_bytes: int = 1_000_000,
) -> JobSpec:
    input_artifact = store.put(owner(), b"input", media_type="application/json", semantic_type="inquiry-action")
    return JobSpec(
        id=identifier,
        owner=owner(),
        invocation=InvocationSpec(
            capability=CapabilityRef(
                id="mixle.inquiry.action", version="1", input_schema="input/v1", output_schema="result/v1"
            ),
            inputs=(input_artifact,),
            parameters={"parameter": 1},
        ),
        resources=ResourceLimits(
            timeout_seconds=60, memory_bytes=1_000_000, cpu_seconds=60, output_bytes=output_bytes, event_count=100
        ),
        retry=RetryPolicy(max_attempts=max_attempts, retryable_codes=retryable_codes),
        idempotency_key=key,
    )


def succeed(_record, _context: WorkerContext) -> HandlerResult:
    return HandlerResult(data=b"result", media_type="application/json", semantic_type="data-result")


class ManualClock:
    """A controllable clock so lease-expiry races are deterministic, not timing-dependent."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def advance(self, delta: float) -> None:
        self.value += delta

    def __call__(self) -> float:
        return self.value


def test_no_queued_job_reports_no_job(tmp_path) -> None:
    runner = DurableLocalRunner(tmp_path / "runner")
    report = run_once(runner, "worker-1", succeed)
    assert report.outcome is WorkOutcome.NO_JOB
    assert report.job_id is None


def test_run_once_claims_starts_and_durably_completes_a_job(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    root = tmp_path / "runner"
    runner = DurableLocalRunner(root, artifact_store=artifacts)
    spec = job_spec(artifacts)
    runner.submit(spec)

    report = run_once(runner, "worker-1", succeed)

    assert report.outcome is WorkOutcome.SUCCEEDED
    assert report.job_id == spec.id
    assert report.attempt == 1
    restored = DurableLocalRunner(root, artifact_store=artifacts).get(spec.id, owner())
    assert restored.state is JobState.SUCCEEDED
    assert restored.results[0].sha256 == hashlib.sha256(b"result").hexdigest()
    assert restored.results[0].media_type == "application/json"


def test_handler_can_report_progress_and_checkpoint_before_completing(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job_spec(artifacts)
    runner.submit(spec)

    def handler(record, context: WorkerContext) -> HandlerResult:
        assert record.spec.id == spec.id
        context.progress(fraction=0.5)
        context.checkpoint(b"partial", media_type="application/octet-stream")
        return HandlerResult(data=b"final", media_type="application/json", semantic_type="data-result")

    report = run_once(runner, "worker-1", handler)

    assert report.outcome is WorkOutcome.SUCCEEDED
    restored = runner.get(spec.id, owner())
    assert restored.checkpoints[0].sha256 == hashlib.sha256(b"partial").hexdigest()
    assert any(event["kind"] == "progress" and event["payload"]["fraction"] == 0.5 for event in restored.events)


def test_retryable_failure_requeues_and_a_later_attempt_succeeds(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job_spec(artifacts, max_attempts=2, retryable_codes=("upstream_timeout",))
    runner.submit(spec)
    attempts: list[int] = []

    def flaky(record, _context: WorkerContext) -> HandlerResult:
        attempts.append(record.attempt)
        if len(attempts) == 1:
            raise HandlerFailure(code="upstream_timeout", detail="first attempt timed out")
        return HandlerResult(data=b"ok", media_type="application/json", semantic_type="data-result")

    reports = drain(runner, "worker-1", flaky)

    assert [report.outcome for report in reports] == [WorkOutcome.RETRY_QUEUED, WorkOutcome.SUCCEEDED]
    assert attempts == [1, 2]
    assert runner.get(spec.id, owner()).state is JobState.SUCCEEDED


def test_non_retryable_failure_is_terminal_and_records_the_code(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job_spec(artifacts, max_attempts=3, retryable_codes=())  # nothing is retryable
    runner.submit(spec)

    def always_fails(_record, _context: WorkerContext) -> HandlerResult:
        raise HandlerFailure(code="bad_input", detail="the input was malformed")

    report = run_once(runner, "worker-1", always_fails)

    assert report.outcome is WorkOutcome.FAILED
    restored = runner.get(spec.id, owner())
    assert restored.state is JobState.FAILED
    assert restored.error == {"code": "bad_input", "detail": "the input was malformed"}


def test_unclassified_exception_fails_closed_without_crashing_the_loop(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job_spec(artifacts)
    runner.submit(spec)

    def buggy(_record, _context: WorkerContext) -> HandlerResult:
        raise ValueError("boom")

    report = run_once(runner, "worker-1", buggy)

    assert report.outcome is WorkOutcome.FAILED
    assert report.detail == "ValueError: boom"
    assert runner.get(spec.id, owner()).error["code"] == "handler_exception"


def test_oversized_result_fails_the_job_instead_of_raising(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job_spec(artifacts, output_bytes=2)  # succeed() returns 6 bytes
    runner.submit(spec)

    report = run_once(runner, "worker-1", succeed)

    assert report.outcome is WorkOutcome.FAILED
    restored = runner.get(spec.id, owner())
    assert restored.state is JobState.FAILED
    assert restored.error["code"] == "result_rejected"


def test_cooperative_cancellation_is_acknowledged_through_the_lease(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job_spec(artifacts)
    runner.submit(spec)

    def cancels_when_asked(record, context: WorkerContext) -> HandlerResult:
        runner.request_cancel(record.spec.id, record.spec.owner, reason="operator stop")
        if not context.cancel_requested:
            raise AssertionError("cancellation was requested but not observed through the context")
        raise CooperativeCancellation

    report = run_once(runner, "worker-1", cancels_when_asked)

    assert report.outcome is WorkOutcome.CANCELLED
    assert runner.get(spec.id, owner()).state is JobState.CANCELLED


def test_drain_runs_every_queued_job_to_completion(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    for index in range(3):
        runner.submit(job_spec(artifacts, identifier=f"job-{index}", key=f"key-{index}"))

    reports = drain(runner, "worker-1", succeed)

    assert len(reports) == 3
    assert all(report.outcome is WorkOutcome.SUCCEEDED for report in reports)
    assert {report.job_id for report in reports} == {"job-0", "job-1", "job-2"}


def test_drain_max_jobs_bounds_the_batch(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    for index in range(3):
        runner.submit(job_spec(artifacts, identifier=f"job-{index}", key=f"key-{index}"))

    first_batch = drain(runner, "worker-1", succeed, max_jobs=2)
    assert len(first_batch) == 2

    remainder = drain(runner, "worker-1", succeed)
    assert len(remainder) == 1


def test_drain_rejects_a_negative_max_jobs(tmp_path) -> None:
    runner = DurableLocalRunner(tmp_path / "runner")
    with pytest.raises(OperationalError, match="max_jobs"):
        drain(runner, "worker-1", succeed, max_jobs=-1)


def test_drain_recovers_expired_leases_before_claiming_new_work(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    clock = ManualClock()
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts, clock=clock)
    spec = job_spec(artifacts, max_attempts=2)
    runner.submit(spec)
    runner.claim("dead-worker", lease_seconds=10.0)  # simulates a worker that crashed mid-job
    clock.advance(20.0)  # the crashed worker's lease is now expired

    reports = drain(runner, "worker-2", succeed)

    assert len(reports) == 1
    assert reports[0].outcome is WorkOutcome.SUCCEEDED
    assert reports[0].attempt == 2  # attempt 1 belonged to the crashed worker's claim


def test_lease_lost_mid_handler_is_reported_not_raised_and_stays_recoverable(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    clock = ManualClock()
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts, clock=clock)
    spec = job_spec(artifacts, max_attempts=2)
    runner.submit(spec)

    def outlives_its_lease(_record, _context: WorkerContext) -> HandlerResult:
        clock.advance(120.0)  # far past the lease granted below
        return HandlerResult(data=b"late", media_type="application/json", semantic_type="data-result")

    report = run_once(runner, "worker-1", outlives_its_lease, lease_seconds=30.0)

    assert report.outcome is WorkOutcome.UNRESOLVABLE
    # Nothing is stuck: the runner's own safety net still recovers it on the next sweep.
    recovered = runner.recover_expired()
    assert recovered == (spec.id,)
    assert runner.get(spec.id, owner()).state is JobState.QUEUED


def test_run_forever_polls_until_stopped_and_sleeps_only_when_idle(tmp_path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    runner = DurableLocalRunner(tmp_path / "runner", artifact_store=artifacts)
    spec = job_spec(artifacts)
    runner.submit(spec)
    sleep_calls: list[float] = []
    checks = {"count": 0}

    def should_stop() -> bool:
        checks["count"] += 1
        return checks["count"] > 3

    run_forever(runner, "worker-1", succeed, poll_interval=0.01, should_stop=should_stop, sleep=sleep_calls.append)

    assert runner.get(spec.id, owner()).state is JobState.SUCCEEDED
    assert sleep_calls == [0.01, 0.01]
