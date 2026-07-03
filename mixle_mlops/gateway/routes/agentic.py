"""Serve the agentic artifacts — distilled function-callers and planners, decide-or-escalate over HTTP.

Artifacts produced by ``mixle.task.distill_tool_caller(...).save(...)`` and
``mixle.task.distill_planner(...).save(...)`` are dropped under
``{registry_root}/toolcallers/{name}`` and ``{registry_root}/planners/{name}``. The server has no
frontier, so it serves exactly the teacher-free half of the honesty contract (``try_local`` /
``try_plan``): a trustworthy call or plan, or ``escalate=true`` — in which case the CALLER runs its
frontier and posts the answer back as a trace for the next distillation round.

  * ``GET  /toolcallers`` · ``GET /planners``                     — list deployed artifacts.
  * ``POST /toolcallers/{name}/call``    — ``{"input": text}`` -> ``{"tool", "args", "escalate"}``.
  * ``POST /planners/{name}/plan``       — ``{"input": text}`` -> ``{"plan", "escalate"}``.
  * ``POST /{kind}/{name}/feedback``     — the frontier's call/plan for an escalated input -> trace count.
  * ``GET  /{kind}/{name}/verification`` — the verification baked into the manifest.
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

_CACHE: dict[tuple[str, str], tuple[float, Any]] = {}
_LOCK = threading.Lock()

_MANIFEST = {"toolcallers": "toolcaller.json", "planners": "planner.json", "genplanners": "genplanner.json"}


def _root(kind: str) -> Path:
    return Path(get_settings().registry_root) / kind


def _never_teacher(*_a: Any, **_k: Any) -> Any:  # loaded artifacts serve local-only; escalation is the caller's job
    raise RuntimeError("the serving gateway has no frontier teacher")


def _load(kind: str, name: str) -> Any:
    path = _root(kind) / name
    manifest = path / _MANIFEST[kind]
    if not manifest.exists():
        raise HTTPException(status_code=404, detail=f"no deployed {kind[:-1]} named {name!r}")
    stamp = manifest.stat().st_mtime
    with _LOCK:
        hit = _CACHE.get((kind, name))
        if hit is not None and hit[0] == stamp:
            return hit[1]
    try:
        from mixle.task import GenerativePlanner, Planner, ToolCaller

        loader = {"toolcallers": ToolCaller, "planners": Planner, "genplanners": GenerativePlanner}[kind]
        obj = loader.load(str(path), _never_teacher)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken artifact must 500 with the reason
        raise HTTPException(status_code=500, detail=f"failed to load {kind[:-1]} {name!r}: {exc}") from exc
    with _LOCK:
        _CACHE[(kind, name)] = (stamp, obj)
    return obj


def _list(kind: str) -> dict[str, Any]:
    root = _root(kind)
    names = sorted(p.name for p in root.iterdir() if (p / _MANIFEST[kind]).exists()) if root.is_dir() else []
    return {kind: names}


def _feedback(kind: str, name: str, body: dict[str, Any], answer_key: str) -> dict[str, Any]:
    if "input" not in body or answer_key not in body:
        raise HTTPException(status_code=422, detail=f'body must be {{"input": ..., "{answer_key}": ...}}')
    path = _root(kind) / name
    if not (path / _MANIFEST[kind]).exists():
        raise HTTPException(status_code=404, detail=f"no deployed {kind[:-1]} named {name!r}")
    harvested = path / "harvested.jsonl"
    line = json.dumps({"input": body["input"], answer_key: body[answer_key]})
    with _LOCK:
        with open(harvested, "a") as f:
            f.write(line + "\n")
        count = sum(1 for _ in open(harvested))
    return {"harvested": count}


def _verification(kind: str, name: str) -> dict[str, Any]:
    path = _root(kind) / name / _MANIFEST[kind]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no deployed {kind[:-1]} named {name!r}")
    m = json.loads(path.read_text())
    keep = ("selection_agreement", "plan_agreement", "tools", "max_steps", "conf_floor", "constrained", "kind")
    return {k: m[k] for k in keep if k in m}


@router.get("/toolcallers")
def list_toolcallers(user: User = Depends(require_user)) -> dict[str, Any]:
    return _list("toolcallers")


@router.post("/toolcallers/{name}/call")
def call(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "input" not in body:
        raise HTTPException(status_code=422, detail='body must be {"input": <request text>}')
    local = _load("toolcallers", name).try_local(str(body["input"]))
    if local is None:
        return {"tool": None, "args": {}, "escalate": True}
    return {**local, "escalate": False}


@router.post("/toolcallers/{name}/feedback")
def toolcaller_feedback(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    return _feedback("toolcallers", name, body, "call")


@router.get("/toolcallers/{name}/verification")
def toolcaller_verification(name: str, user: User = Depends(require_user)) -> dict[str, Any]:
    return _verification("toolcallers", name)


@router.get("/planners")
def list_planners(user: User = Depends(require_user)) -> dict[str, Any]:
    return _list("planners")


@router.post("/planners/{name}/plan")
def plan(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "input" not in body:
        raise HTTPException(status_code=422, detail='body must be {"input": <request text>}')
    local = _load("planners", name).try_plan(str(body["input"]))
    if local is None:
        return {"plan": None, "escalate": True}
    return {"plan": local["plan"], "escalate": False}


@router.get("/genplanners")
def list_genplanners(user: User = Depends(require_user)) -> dict[str, Any]:
    return _list("genplanners")


@router.post("/genplanners/{name}/plan")
def genplan(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "input" not in body:
        raise HTTPException(status_code=422, detail='body must be {"input": <request text>}')
    local = _load("genplanners", name).try_plan(str(body["input"]))
    if local is None:
        return {"plan": None, "escalate": True}
    return {"plan": local, "escalate": False}


@router.post("/genplanners/{name}/feedback")
def genplanner_feedback(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    return _feedback("genplanners", name, body, "plan")


@router.get("/genplanners/{name}/verification")
def genplanner_verification(name: str, user: User = Depends(require_user)) -> dict[str, Any]:
    return _verification("genplanners", name)


@router.post("/planners/{name}/feedback")
def planner_feedback(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    return _feedback("planners", name, body, "plan")


@router.get("/planners/{name}/verification")
def planner_verification(name: str, user: User = Depends(require_user)) -> dict[str, Any]:
    return _verification("planners", name)
