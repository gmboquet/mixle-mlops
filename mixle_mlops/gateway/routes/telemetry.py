"""Telemetry sink -- the control-plane endpoint local daemons push decision events to (L2 seed).

The pysparkplug workplan records every platform decision (fit, placement, route, escalation, context,
reason, pool_job, drift) as a typed, PII-free ``mixle.telemetry.Event`` in a local buffer. This route
is the shared sink: a local daemon POSTs its accumulated events here, and the learned-orchestration
workstream (J) reads them back as ``(features, choice, outcome)`` training rows -- the static policies
are the teachers, the learned routers/placers earn traffic only when receipted never-worse against
this history.

Events are scoped per user (the account that pushed them) and persisted as JSONL under
``{registry_root}/telemetry/{user}/events.jsonl`` -- the same on-disk format ``mixle.telemetry.Telemetry``
writes, so the local buffer and the shared sink are one lineage. Events carry decision FEATURES and
OUTCOMES only, never raw user content, so publishing them leaks nothing.

  * ``POST /v1/telemetry``               -- ``{"events": [{kind, features, choice, outcome, tags}, ...]}``.
  * ``GET  /v1/telemetry/stats``         -- event counts by kind.
  * ``GET  /v1/telemetry/training/{kind}`` -- the ``(features, choice, outcome)`` rows for a decision kind.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...accounts.models import User
from ...config import get_settings
from ..auth import require_user

router = APIRouter()

_LOCK = threading.Lock()


def _sink_path(user: User) -> Path:
    uid = str(getattr(user, "id", None) or getattr(user, "email", "shared"))
    return Path(get_settings().registry_root) / "telemetry" / uid / "events.jsonl"


def _recorder(user: User) -> Any:
    from mixle.telemetry import Telemetry

    return Telemetry(str(_sink_path(user)))


@router.post("/telemetry")
def push(body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    events = body.get("events")
    if not isinstance(events, list):
        raise HTTPException(status_code=422, detail='body must be {"events": [{kind, features, choice, outcome}, ...]}')
    with _LOCK:
        rec = _recorder(user)
        accepted, rejected = 0, 0
        for ev in events:
            if not isinstance(ev, dict) or "kind" not in ev:
                rejected += 1
                continue
            try:
                rec.record(
                    str(ev["kind"]),
                    features=dict(ev.get("features", {})),
                    choice=ev.get("choice"),
                    outcome=dict(ev.get("outcome", {})),
                    tags=dict(ev.get("tags", {})),
                    when=ev.get("ts"),
                )
                accepted += 1
            except ValueError:  # unknown event kind -> reject that one, keep the rest
                rejected += 1
        rec.flush()
    return {"accepted": accepted, "rejected": rejected, "total": len(_recorder(user))}


@router.get("/telemetry/stats")
def stats(user: User = Depends(require_user)) -> dict[str, Any]:
    rec = _recorder(user)
    kinds: dict[str, int] = {}
    for ev in rec.events():
        kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
    return {"n_events": len(rec), "kinds": kinds}


@router.get("/telemetry/training/{kind}")
def training(kind: str, user: User = Depends(require_user)) -> dict[str, Any]:
    """The (features, choice, outcome) rows for a decision kind -- what workstream J learns on."""
    rec = _recorder(user)
    rows = [{"features": f, "choice": c, "outcome": o} for f, c, o in rec.training_rows(kind)]
    return {"kind": kind, "n": len(rows), "rows": rows}
