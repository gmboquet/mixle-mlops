"""Durable single-node reference runner with leases, recovery, and cancellation."""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .artifacts import LocalArtifactStore
from .contracts import ArtifactRef, JobSpec, JobState, OperationalError, OwnerScope, canonical_json


@dataclass(frozen=True)
class Lease:
    worker_id: str
    token: str
    expires_at: float


@dataclass
class JobRecord:
    spec: JobSpec
    state: JobState = JobState.QUEUED
    attempt: int = 0
    lease: Lease | None = None
    cancel_requested: bool = False
    checkpoints: list[ArtifactRef] = field(default_factory=list)
    results: list[ArtifactRef] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "state": self.state.value,
            "attempt": self.attempt,
            "lease": None if self.lease is None else self.lease.__dict__,
            "cancel_requested": self.cancel_requested,
            "checkpoints": [item.__dict__ | {"owner": item.owner.__dict__} for item in self.checkpoints],
            "results": [item.__dict__ | {"owner": item.owner.__dict__} for item in self.results],
            "events": self.events,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JobRecord:
        def artifact(item: dict[str, Any]) -> ArtifactRef:
            return ArtifactRef(
                owner=OwnerScope(**item["owner"]), **{key: entry for key, entry in item.items() if key != "owner"}
            )

        return cls(
            spec=JobSpec.from_dict(value["spec"]),
            state=JobState(value["state"]),
            attempt=int(value["attempt"]),
            lease=Lease(**value["lease"]) if value.get("lease") else None,
            cancel_requested=bool(value.get("cancel_requested", False)),
            checkpoints=[artifact(item) for item in value.get("checkpoints", [])],
            results=[artifact(item) for item in value.get("results", [])],
            events=list(value.get("events", [])),
            error=value.get("error"),
        )


class DurableLocalRunner:
    """Reference operational runner; domain execution is injected by a worker, not interpreted here."""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        artifact_store: LocalArtifactStore | None = None,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "jobs.json"
        self.clock = clock
        self.artifacts = artifact_store or LocalArtifactStore(self.root / "artifacts")
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        if self.path.is_symlink():
            raise OperationalError("runner database cannot be a symlink")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        self._jobs = {identifier: JobRecord.from_dict(record) for identifier, record in value.get("jobs", {}).items()}
        self._idempotency = dict(value.get("idempotency", {}))

    def _write(self) -> None:
        rendered = (
            canonical_json(
                {
                    "schema_version": "1.0.0",
                    "jobs": {key: value.as_dict() for key, value in self._jobs.items()},
                    "idempotency": self._idempotency,
                }
            )
            + "\n"
        )
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    @staticmethod
    def _idempotency_key(spec: JobSpec) -> str | None:
        return None if spec.idempotency_key is None else f"{spec.owner.key}:{spec.idempotency_key}"

    def _event(self, record: JobRecord, kind: str, **payload: Any) -> None:
        if len(record.events) >= record.spec.resources.event_count:
            raise OperationalError("job event bound exhausted")
        record.events.append(
            {
                "sequence": len(record.events),
                "kind": kind,
                "at": self.clock(),
                "attempt": record.attempt,
                "payload": payload,
            }
        )

    def submit(self, spec: JobSpec) -> JobRecord:
        key = self._idempotency_key(spec)
        if key and key in self._idempotency:
            existing = self._jobs[self._idempotency[key]]
            if existing.spec.semantic_id != spec.semantic_id:
                raise OperationalError("idempotency key was reused for different semantic inputs")
            return existing
        if spec.id in self._jobs:
            existing = self._jobs[spec.id]
            if existing.spec.as_dict() != spec.as_dict():
                raise OperationalError("job id already exists with different operational inputs")
            return existing
        record = JobRecord(spec)
        self._event(record, "submitted", semantic_id=spec.semantic_id)
        self._jobs[spec.id] = record
        if key:
            self._idempotency[key] = spec.id
        self._write()
        return record

    def get(self, job_id: str, owner: OwnerScope) -> JobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.spec.owner != owner:
            raise PermissionError("job owner scope does not match caller")
        return record

    def claim(self, worker_id: str, *, lease_seconds: float = 30.0) -> tuple[JobRecord, Lease] | None:
        if lease_seconds <= 0:
            raise OperationalError("lease duration must be positive")
        eligible = sorted(
            (
                record
                for record in self._jobs.values()
                if record.state is JobState.QUEUED and not record.cancel_requested
            ),
            key=lambda record: (-record.spec.priority, record.spec.id),
        )
        if not eligible:
            return None
        record = eligible[0]
        lease = Lease(worker_id, secrets.token_hex(24), self.clock() + lease_seconds)
        record.state = JobState.LEASED
        record.attempt += 1
        record.lease = lease
        self._event(record, "leased", worker_id=worker_id, expires_at=lease.expires_at)
        self._write()
        return record, lease

    def _leased(self, job_id: str, token: str, *, states: tuple[JobState, ...]) -> JobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if (
            record.state not in states
            or record.lease is None
            or not secrets.compare_digest(record.lease.token, token)
        ):
            raise OperationalError("worker does not hold the required job lease")
        if record.lease.expires_at <= self.clock():
            raise OperationalError("job lease expired")
        return record

    def start(self, job_id: str, token: str) -> JobRecord:
        record = self._leased(job_id, token, states=(JobState.LEASED,))
        record.state = JobState.RUNNING
        self._event(record, "started")
        self._write()
        return record

    def heartbeat(self, job_id: str, token: str, *, lease_seconds: float = 30.0) -> Lease:
        record = self._leased(job_id, token, states=(JobState.LEASED, JobState.RUNNING))
        record.lease = Lease(record.lease.worker_id, token, self.clock() + lease_seconds)
        self._event(record, "heartbeat", expires_at=record.lease.expires_at)
        self._write()
        return record.lease

    def progress(self, job_id: str, token: str, **progress: Any) -> JobRecord:
        record = self._leased(job_id, token, states=(JobState.RUNNING,))
        self._event(record, "progress", **progress)
        self._write()
        return record

    def checkpoint(
        self, job_id: str, token: str, data: bytes, *, media_type: str = "application/octet-stream"
    ) -> ArtifactRef:
        record = self._leased(job_id, token, states=(JobState.RUNNING,))
        artifact = self.artifacts.put(record.spec.owner, data, media_type=media_type, semantic_type="checkpoint")
        record.checkpoints.append(artifact)
        self._event(record, "checkpointed", artifact_sha256=artifact.sha256)
        self._write()
        return artifact

    def complete(
        self,
        job_id: str,
        token: str,
        data: bytes,
        *,
        media_type: str,
        semantic_type: str,
    ) -> ArtifactRef:
        record = self._leased(job_id, token, states=(JobState.RUNNING,))
        if record.cancel_requested:
            raise OperationalError("cancelled work cannot be completed")
        if len(data) > record.spec.resources.output_bytes:
            raise OperationalError("result exceeds job output bound")
        artifact = self.artifacts.put(record.spec.owner, data, media_type=media_type, semantic_type=semantic_type)
        record.results.append(artifact)
        record.state = JobState.SUCCEEDED
        record.lease = None
        self._event(record, "succeeded", result_sha256=artifact.sha256, epistemic_disposition="not_evaluated")
        self._write()
        return artifact

    def fail(self, job_id: str, token: str, *, code: str, detail: str) -> JobRecord:
        record = self._leased(job_id, token, states=(JobState.RUNNING, JobState.LEASED))
        record.error = {"code": code, "detail": detail}
        retry = code in record.spec.retry.retryable_codes and record.attempt < record.spec.retry.max_attempts
        record.state = JobState.QUEUED if retry else JobState.FAILED
        record.lease = None
        self._event(record, "retry_queued" if retry else "failed", code=code)
        self._write()
        return record

    def request_cancel(self, job_id: str, owner: OwnerScope, *, reason: str) -> JobRecord:
        record = self.get(job_id, owner)
        if record.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            raise OperationalError("terminal job cannot be cancelled")
        record.cancel_requested = True
        self._event(record, "cancel_requested", reason=reason)
        if record.state is JobState.QUEUED:
            record.state = JobState.CANCELLED
        self._write()
        return record

    def acknowledge_cancel(self, job_id: str, token: str) -> JobRecord:
        record = self._leased(job_id, token, states=(JobState.LEASED, JobState.RUNNING))
        if not record.cancel_requested:
            raise OperationalError("job has no cancellation request")
        record.state = JobState.CANCELLED
        record.lease = None
        self._event(record, "cancelled")
        self._write()
        return record

    def recover_expired(self) -> tuple[str, ...]:
        recovered: list[str] = []
        for record in self._jobs.values():
            if record.state not in {JobState.LEASED, JobState.RUNNING} or record.lease is None:
                continue
            if record.lease.expires_at > self.clock():
                continue
            if record.cancel_requested:
                record.state = JobState.CANCELLED
                kind = "cancelled_after_lease_expiry"
            elif record.attempt < record.spec.retry.max_attempts:
                record.state = JobState.QUEUED
                kind = "lease_expired_requeued"
            else:
                record.state = JobState.FAILED
                record.error = {"code": "lease_expired", "detail": "worker lease expired"}
                kind = "lease_expired_failed"
            record.lease = None
            self._event(record, kind)
            recovered.append(record.spec.id)
        if recovered:
            self._write()
        return tuple(sorted(recovered))


__all__ = ["DurableLocalRunner", "JobRecord", "Lease"]
