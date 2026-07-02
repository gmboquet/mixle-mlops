"""Persistence for user fine-tune jobs.

One :class:`FineTuneJob` row tracks a training request through its lifecycle (queued -> running -> succeeded /
failed, or planned for a GPU job), records who owns it, which served model it produced, and free-form
metrics/error text. Follows the ``datasets/models.py`` pattern (uuid pk, utc timestamps, JSON-encoded columns);
the table is created defensively at first use by the route.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Field, SQLModel

# lifecycle states a job moves through
QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
PLANNED = "planned"        # a GPU job whose (free) plan was produced; launching is the operator's keyed step
CANCELLED = "cancelled"
TERMINAL = {SUCCEEDED, FAILED, CANCELLED}


def _uuid() -> str:
    return "ft-" + uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FineTuneJob(SQLModel, table=True):
    """Metadata for one fine-tune request and the served model it produced."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str | None = Field(default=None, index=True)       # who requested it (nullable for anon)
    backend: str = Field(default="structured", index=True)      # "structured" | "llm" | "mixle"
    model: str = Field(index=True)                              # the served model id it registers as
    status: str = Field(default=QUEUED, index=True)
    base_model: str | None = None                              # for llm/mixle GPU backends
    artifact_path: str | None = None                          # where the trained artifact was written
    error: str | None = None
    metrics_json: str | None = None                           # JSON: {train_agreement, edges, labels, ...}
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None

    def metrics(self) -> dict[str, Any]:
        if not self.metrics_json:
            return {}
        try:
            obj = json.loads(self.metrics_json)
            return obj if isinstance(obj, dict) else {}
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def encode_metrics(metrics: dict[str, Any] | None) -> str | None:
        return None if metrics is None else json.dumps(metrics, default=str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "fine_tune",
            "backend": self.backend,
            "model": self.model,
            "status": self.status,
            "base_model": self.base_model,
            "artifact_path": self.artifact_path,
            "error": self.error,
            "metrics": self.metrics(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
