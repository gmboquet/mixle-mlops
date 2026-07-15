"""Durable, resumable monitoring for knowledge-bundle runs between models."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def bundle_digest(bundle: Mapping[str, Any]) -> str:
    """Validate the minimum knowledge boundary and return its canonical digest."""

    missing = {"id", "project_id", "task", "target_kind", "revision", "items", "gaps"} - set(bundle)
    if missing:
        raise ValueError(f"knowledge bundle is missing required fields: {sorted(missing)}")
    if not isinstance(bundle["revision"], int) or bundle["revision"] < 1:
        raise ValueError("knowledge bundle revision must be a positive integer")
    item_ids = [item.get("id") for item in bundle["items"] if isinstance(item, Mapping)]
    if len(item_ids) != len(bundle["items"]) or None in item_ids or len(item_ids) != len(set(item_ids)):
        raise ValueError("knowledge bundle items require unique ids")
    if any(not item.get("kind") or not item.get("schema_uri") or not item.get("content_hash") for item in bundle["items"]):
        raise ValueError("knowledge bundle items require kind, schema_uri, and content_hash")
    return hashlib.sha256(_canonical(bundle)).hexdigest()


class ContextRunState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TRANSITIONS = {
    ContextRunState.PLANNED: {ContextRunState.RUNNING, ContextRunState.CANCELLED},
    ContextRunState.RUNNING: {ContextRunState.COMPLETED, ContextRunState.FAILED, ContextRunState.CANCELLED},
    ContextRunState.FAILED: {ContextRunState.RUNNING, ContextRunState.CANCELLED},
    ContextRunState.COMPLETED: set(),
    ContextRunState.CANCELLED: set(),
}


@dataclass(frozen=True)
class ContextEvent:
    sequence: int
    kind: str
    occurred_at: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextRun:
    id: str
    bundle_id: str
    bundle_revision: int
    bundle_sha256: str
    model_id: str
    model_version: str | None = None
    required_capability_ids: list[str] = field(default_factory=list)
    parent_run_id: str | None = None
    idempotency_key: str | None = None
    state: ContextRunState = ContextRunState.PLANNED
    attempt: int = 0
    events: list[ContextEvent] = field(default_factory=list)
    result_refs: list[str] = field(default_factory=list)
    continuation: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextRun:
        return cls(
            **{
                **dict(value),
                "state": ContextRunState(value["state"]),
                "events": [ContextEvent(**event) for event in value.get("events", [])],
            }
        )


class ContextRunStore:
    """Filesystem run ledger with atomic updates and explicit transitions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or ".." in run_id:
            raise ValueError("run id must be a non-empty path-safe identifier")
        return self.root / f"{run_id}.json"

    def _write(self, run: ContextRun) -> None:
        target = self._path(run.id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)

    def get(self, run_id: str) -> ContextRun:
        path = self._path(run_id)
        if not path.is_file():
            raise KeyError(run_id)
        return ContextRun.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def start(
        self,
        run_id: str,
        bundle: Mapping[str, Any],
        *,
        model_id: str,
        model_version: str | None = None,
        required_capability_ids: list[str] | None = None,
        parent_run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> ContextRun:
        digest = bundle_digest(bundle)
        path = self._path(run_id)
        if path.exists():
            existing = self.get(run_id)
            identity = (existing.bundle_sha256, existing.model_id, existing.idempotency_key)
            if identity != (digest, model_id, idempotency_key):
                raise ValueError("run id already exists with different bundle/model identity")
            return existing
        run = ContextRun(
            id=run_id,
            bundle_id=str(bundle["id"]),
            bundle_revision=int(bundle["revision"]),
            bundle_sha256=digest,
            model_id=model_id,
            model_version=model_version,
            required_capability_ids=list(required_capability_ids or bundle.get("required_capability_ids", [])),
            parent_run_id=parent_run_id,
            idempotency_key=idempotency_key,
        )
        self._write(run)
        return run

    def transition(
        self,
        run_id: str,
        state: ContextRunState,
        *,
        result_refs: list[str] | None = None,
        continuation: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> ContextRun:
        run = self.get(run_id)
        if state not in _TRANSITIONS[run.state]:
            raise ValueError(f"invalid context-run transition {run.state.value} -> {state.value}")
        run.state = state
        if state == ContextRunState.RUNNING:
            run.attempt += 1
            run.error = None
        if result_refs is not None:
            run.result_refs = list(result_refs)
        if continuation is not None:
            run.continuation = dict(continuation)
        if error is not None:
            run.error = dict(error)
        run.updated_at = _now()
        self._write(run)
        return run

    def event(self, run_id: str, kind: str, **data: Any) -> ContextRun:
        run = self.get(run_id)
        if run.state != ContextRunState.RUNNING:
            raise ValueError("events may only be appended while a context run is running")
        run.events.append(ContextEvent(len(run.events) + 1, kind, _now(), dict(data)))
        run.updated_at = _now()
        self._write(run)
        return run

    def checkpoint(self, run_id: str, continuation: Mapping[str, Any], *, state_refs: list[str] | None = None) -> ContextRun:
        run = self.event(run_id, "checkpoint", continuation=dict(continuation), state_refs=list(state_refs or []))
        run.continuation = dict(continuation)
        run.updated_at = _now()
        self._write(run)
        return run
