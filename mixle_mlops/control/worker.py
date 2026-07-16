"""The worker loop that actually drives ``DurableLocalRunner`` jobs to a terminal state.

``DurableLocalRunner`` persists job state and lease bookkeeping, but by its own contract
("domain execution is injected by a worker, not interpreted here") it never calls anything
itself. Nothing before this module closed that loop: every existing caller -- including the
runner's own tests -- drove ``claim``/``start``/``heartbeat``/``checkpoint``/``complete``/
``fail`` by hand, one call at a time. ``docs/operations.md`` already describes what a worker
does ("Workers heartbeat, checkpoint, and acknowledge cancellation using their lease token");
this module is the first actual implementation of that worker, not just its description.

It claims one job, invokes an injected domain-neutral handler, and resolves the job through
the runner's existing state machine -- translating handler exceptions and lease races into a
typed :class:`WorkReport` instead of raising out of the loop. It stays as domain-neutral as
the runner it drives: a handler is any callable that turns a :class:`~.runner.JobRecord` into
a :class:`HandlerResult` (opaque bytes plus their media/semantic type); this module has no
opinion about what those bytes mean.

Scope: a single in-process, single-thread reference loop, matching the runner's own "durable
single-node reference" framing. It deliberately does NOT add:

- distributed or multi-worker coordination (leasing already makes that safe across processes
  sharing one runner root; scheduling across them is not addressed here);
- delayed/backoff retry scheduling -- a retried job is immediately re-queued and can be
  reclaimed on the very next ``claim()``, exactly as the runner already behaves on its own;
- automatic mid-handler heartbeating -- a handler that runs longer than its lease must call
  ``context.heartbeat()`` itself, exactly as ``docs/operations.md`` already specifies. A
  handler that does not will have its lease expire and be reclaimed by ``recover_expired()``,
  which is the existing, intended safety net, not a bug in this module.

Those remain later work, matching ``docs/operations.md``'s own "Distributed queues, ...
and SLO automation remain later work."
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .contracts import ArtifactRef, JobState, OperationalError
from .runner import DurableLocalRunner, JobRecord


class CooperativeCancellation(Exception):
    """A handler raises this once it observes ``WorkerContext.cancel_requested`` and stops."""


class HandlerFailure(Exception):
    """A handler raises this to fail its job with a specific, retry-policy-matchable code.

    ``code`` is matched against the job's ``RetryPolicy.retryable_codes`` by the runner itself
    (see ``DurableLocalRunner.fail``); this module does not duplicate that decision.
    """

    def __init__(self, *, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class HandlerResult:
    """The opaque output a handler hands back for the runner to store as the job's result."""

    data: bytes
    media_type: str
    semantic_type: str


class WorkOutcome(StrEnum):
    NO_JOB = "no_job"
    """``claim()`` found no eligible queued job; nothing else happened."""
    SUCCEEDED = "succeeded"
    """The handler returned a result and the runner recorded the job as terminally succeeded."""
    RETRY_QUEUED = "retry_queued"
    """The attempt failed with a retryable code and the job is queued again for another attempt."""
    FAILED = "failed"
    """The attempt failed and the runner recorded the job as terminally failed (no more retries)."""
    CANCELLED = "cancelled"
    """The handler observed ``cancel_requested`` and the runner recorded the cancellation."""
    UNRESOLVABLE = "unresolvable"
    """The runner rejected this attempt's resolution (lease expired/stolen mid-handler, or the
    job's state changed under a concurrent actor). The job's fate is owned by whoever holds it
    now, or by the next ``recover_expired()`` sweep -- this worker does not retry resolving it."""


@dataclass(frozen=True)
class WorkReport:
    """What happened to at most one job during one :func:`run_once` call."""

    outcome: WorkOutcome
    job_id: str | None = None
    attempt: int | None = None
    detail: str | None = None


class WorkerContext:
    """The handler's only channel back to the runner for the one job it was given."""

    def __init__(
        self,
        runner: DurableLocalRunner,
        record: JobRecord,
        *,
        worker_id: str,
        token: str,
        lease_seconds: float,
    ) -> None:
        self._runner = runner
        self._record = record
        self._token = token
        self._lease_seconds = lease_seconds
        self.worker_id = worker_id

    @property
    def job_id(self) -> str:
        return self._record.spec.id

    @property
    def cancel_requested(self) -> bool:
        """Re-reads current state so a concurrent ``request_cancel`` is observed mid-handler."""
        return self._runner.get(self.job_id, self._record.spec.owner).cancel_requested

    def heartbeat(self, *, lease_seconds: float | None = None) -> None:
        """Extend the lease. Raises ``OperationalError`` if the lease was already lost."""
        self._runner.heartbeat(
            self.job_id, self._token, lease_seconds=self._lease_seconds if lease_seconds is None else lease_seconds
        )

    def progress(self, **progress: Any) -> None:
        self._runner.progress(self.job_id, self._token, **progress)

    def checkpoint(self, data: bytes, *, media_type: str = "application/octet-stream") -> ArtifactRef:
        return self._runner.checkpoint(self.job_id, self._token, data, media_type=media_type)


WorkerHandler = Callable[[JobRecord, WorkerContext], HandlerResult]


def _resolve_failure(runner: DurableLocalRunner, job_id: str, token: str, *, code: str, detail: str) -> WorkOutcome:
    try:
        record = runner.fail(job_id, token, code=code, detail=detail)
    except OperationalError:
        # The lease was lost (expired, reclaimed, or the job already resolved) between the
        # handler returning and us recording that outcome. The job's terminal state is owned
        # by whoever holds it now (or by `recover_expired`); we must not raise out of the loop.
        return WorkOutcome.UNRESOLVABLE
    # `fail()` itself decides retry vs terminal from the job's own RetryPolicy; mirror that
    # decision in the report instead of collapsing both into one ambiguous "failed".
    return WorkOutcome.RETRY_QUEUED if record.state is JobState.QUEUED else WorkOutcome.FAILED


def run_once(
    runner: DurableLocalRunner,
    worker_id: str,
    handler: WorkerHandler,
    *,
    lease_seconds: float = 30.0,
) -> WorkReport:
    """Claim at most one queued job and drive it to resolution.

    Never raises for handler failures or lease races -- both come back as a typed
    :class:`WorkReport` so a caller can loop indefinitely without a top-level try/except.
    """
    claimed = runner.claim(worker_id, lease_seconds=lease_seconds)
    if claimed is None:
        return WorkReport(outcome=WorkOutcome.NO_JOB)
    record, lease = claimed
    job_id = record.spec.id
    attempt = record.attempt

    try:
        record = runner.start(job_id, lease.token)
    except OperationalError as exc:
        return WorkReport(outcome=WorkOutcome.UNRESOLVABLE, job_id=job_id, attempt=attempt, detail=str(exc))

    context = WorkerContext(runner, record, worker_id=worker_id, token=lease.token, lease_seconds=lease_seconds)
    try:
        result = handler(record, context)
    except CooperativeCancellation:
        try:
            runner.acknowledge_cancel(job_id, lease.token)
        except OperationalError as exc:
            return WorkReport(outcome=WorkOutcome.UNRESOLVABLE, job_id=job_id, attempt=attempt, detail=str(exc))
        return WorkReport(outcome=WorkOutcome.CANCELLED, job_id=job_id, attempt=attempt)
    except HandlerFailure as exc:
        outcome = _resolve_failure(runner, job_id, lease.token, code=exc.code, detail=exc.detail)
        return WorkReport(outcome=outcome, job_id=job_id, attempt=attempt, detail=exc.detail)
    except Exception as exc:  # noqa: BLE001 -- handler code is arbitrary injected domain work
        detail = f"{type(exc).__name__}: {exc}"
        outcome = _resolve_failure(runner, job_id, lease.token, code="handler_exception", detail=detail)
        return WorkReport(outcome=outcome, job_id=job_id, attempt=attempt, detail=detail)

    try:
        runner.complete(
            job_id, lease.token, result.data, media_type=result.media_type, semantic_type=result.semantic_type
        )
    except OperationalError as exc:
        outcome = _resolve_failure(runner, job_id, lease.token, code="result_rejected", detail=str(exc))
        return WorkReport(outcome=outcome, job_id=job_id, attempt=attempt, detail=str(exc))
    return WorkReport(outcome=WorkOutcome.SUCCEEDED, job_id=job_id, attempt=attempt)


def drain(
    runner: DurableLocalRunner,
    worker_id: str,
    handler: WorkerHandler,
    *,
    lease_seconds: float = 30.0,
    max_jobs: int | None = None,
    recover_first: bool = True,
) -> tuple[WorkReport, ...]:
    """Run jobs synchronously until the queue is empty (or ``max_jobs`` reports collected).

    Intended for batch/offline use and tests, not a long-lived process -- see
    :func:`run_forever` for that.
    """
    if max_jobs is not None and max_jobs < 0:
        raise OperationalError("max_jobs must not be negative")
    if recover_first:
        runner.recover_expired()
    reports: list[WorkReport] = []
    while max_jobs is None or len(reports) < max_jobs:
        report = run_once(runner, worker_id, handler, lease_seconds=lease_seconds)
        if report.outcome is WorkOutcome.NO_JOB:
            break
        reports.append(report)
    return tuple(reports)


def run_forever(
    runner: DurableLocalRunner,
    worker_id: str,
    handler: WorkerHandler,
    *,
    lease_seconds: float = 30.0,
    poll_interval: float = 1.0,
    should_stop: Callable[[], bool] = lambda: False,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll indefinitely, recovering expired leases every pass, until ``should_stop()`` is true.

    ``sleep`` and ``should_stop`` are injectable so this stays testable without a real clock or
    an actual unbounded loop -- pass a ``should_stop`` that flips true after N calls, and a
    ``sleep`` stub that just counts them.
    """
    while not should_stop():
        runner.recover_expired()
        report = run_once(runner, worker_id, handler, lease_seconds=lease_seconds)
        if report.outcome is WorkOutcome.NO_JOB:
            sleep(poll_interval)


__all__ = [
    "CooperativeCancellation",
    "HandlerFailure",
    "HandlerResult",
    "WorkOutcome",
    "WorkReport",
    "WorkerContext",
    "WorkerHandler",
    "drain",
    "run_forever",
    "run_once",
]
