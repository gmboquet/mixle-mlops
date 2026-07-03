"""Serve ``mixle.task.solve()`` artifacts — the deploy target for distilled task models.

A Solution artifact (calibrated student + conformal threshold + optional OOD gate + verification record)
is dropped under ``{registry_root}/tasks/{name}`` (e.g. ``Solution.save(...)`` straight to that path, or
rsync'd from wherever it was trained). The routes expose the honest serving contract:

  * ``GET  /tasks``                      — list deployed task names.
  * ``POST /tasks/{name}/decide``        — ``{"input": ...}`` -> ``{"label": ..., "escalate": bool}``.
                                           The caller owns the teacher: on ``escalate=true`` it runs the
                                           original code/API itself (the service never guesses).
  * ``POST /tasks/{name}/feedback``      — ``{"input": ..., "label": ...}`` -> harvested-pair count. The
                                           caller posts the teacher's answer for escalated inputs; pairs
                                           accumulate in ``harvested.jsonl`` for the next re-solve.
  * ``GET  /tasks/{name}/verification``  — the verification record baked into the artifact (held-out
                                           agreement, escalation rate, alpha, sizes), so "is this model
                                           trustworthy" is answerable from the endpoint alone.

All routes require an authenticated user when auth is enabled.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...accounts.models import User
from ...config import get_settings
from ..auth import require_user

router = APIRouter()

_CACHE: dict[str, tuple[float, Any]] = {}
_LOCK = threading.Lock()


def _tasks_root() -> Path:
    return Path(get_settings().registry_root) / "tasks"


def _manifest_mtime(path: Path) -> float:
    m = path / "manifest.json"
    return m.stat().st_mtime if m.exists() else 0.0


def _load(name: str) -> Any:
    """Load (and cache, keyed on manifest mtime) the calibrated task model for ``name``."""
    path = _tasks_root() / name
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"no deployed task named {name!r}")
    try:
        from mixle.task import CalibratedTaskModel
    except ImportError as exc:  # pragma: no cover - mixle is a hard dep of the platform
        raise HTTPException(status_code=503, detail=f"mixle is unavailable: {exc}") from exc
    stamp = _manifest_mtime(path)
    with _LOCK:
        hit = _CACHE.get(name)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    try:
        model = CalibratedTaskModel.load(str(path))
    except Exception as exc:  # noqa: BLE001 - a broken artifact must 500 with the reason, not a stack page
        raise HTTPException(status_code=500, detail=f"failed to load task artifact {name!r}: {exc}") from exc
    with _LOCK:
        _CACHE[name] = (stamp, model)
    return model


def _coerce_input(raw: Any) -> Any:
    """JSON dicts stay records; JSON lists become tuples (the record shape solve() trained on)."""
    return tuple(raw) if isinstance(raw, list) else raw


@router.get("/tasks")
def list_tasks(user: User = Depends(require_user)) -> dict[str, Any]:
    root = _tasks_root()
    names = sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists()) if root.is_dir() else []
    return {"tasks": names}


@router.post("/tasks/{name}/decide")
def decide(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "input" not in body:
        raise HTTPException(status_code=422, detail='body must be {"input": <text or record>}')
    model = _load(name)
    label = model.decide(_coerce_input(body["input"]))
    return {"label": label, "escalate": label is None}


@router.post("/tasks/{name}/feedback")
def feedback(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "input" not in body or "label" not in body:
        raise HTTPException(status_code=422, detail='body must be {"input": ..., "label": ...}')
    path = _tasks_root() / name
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"no deployed task named {name!r}")
    line = json.dumps({"input": body["input"], "label": body["label"]})
    harvested = path / "harvested.jsonl"
    with _LOCK:
        with open(harvested, "a") as f:
            f.write(line + "\n")
        count = sum(1 for _ in open(harvested))
    return {"harvested": count}


@router.get("/tasks/{name}/verification")
def verification(name: str, user: User = Depends(require_user)) -> dict[str, Any]:
    model = _load(name)
    meta = (getattr(model.task, "meta", None) or {}).get("solve", {})
    ver = meta.get("verification")
    if not ver:
        raise HTTPException(status_code=404, detail=f"task {name!r} carries no verification record")
    return {"kind": meta.get("kind"), "ood": meta.get("ood"), "verification": ver}
