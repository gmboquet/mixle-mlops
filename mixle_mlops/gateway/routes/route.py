"""Serve calibrated N-tier model routing over deployed task artifacts (mixle.task.Router semantics).

A route is a named stack of deployed tasks (see ``/v1/tasks``), cheapest-first. A request walks the
tiers' calibrated models; the first tier whose conformal decision is a confident singleton answers.
If every tier escalates, the response says so and the CALLER runs the frontier — and should post the
frontier's answer back to the LAST tier's feedback endpoint so the stack keeps learning.

  * ``PUT  /routes/{name}``            — ``{"tiers": ["tiny", "small"], "costs": [0.0001, 0.001, 0.03]}``
                                         (tier task names cheapest-first; costs has one extra final
                                         entry: the frontier's per-request cost, for the report).
  * ``POST /routes/{name}/decide``     — ``{"input": ...}`` -> ``{"label", "tier", "escalate"}``.
  * ``GET  /routes/{name}/report``     — per-tier traffic + realized cost vs frontier-only (receipts).
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
from ._paths import validate_path_segment
from .tasks import _coerce_input, _load

router = APIRouter()

_COUNTS: dict[str, dict[str, int]] = {}
_LOCK = threading.Lock()


def _routes_root() -> Path:
    return Path(get_settings().registry_root) / "routes"


def _spec_path(name: str) -> Path:
    validate_path_segment(name)
    return _routes_root() / f"{name}.json"


def _read_spec(name: str) -> dict[str, Any]:
    p = _spec_path(name)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no route named {name!r}")
    return json.loads(p.read_text())


@router.put("/routes/{name}")
def put_route(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    tiers = body.get("tiers")
    costs = body.get("costs")
    if not isinstance(tiers, list) or not tiers:
        raise HTTPException(status_code=422, detail='body must include "tiers": [task names cheapest-first]')
    if not isinstance(costs, list) or len(costs) != len(tiers) + 1:
        raise HTTPException(status_code=422, detail='"costs" needs one entry per tier plus the frontier cost')
    for t in tiers:
        _load(str(t))  # every tier must be a deployed, loadable task artifact — fail loudly now
    root = _routes_root()
    root.mkdir(parents=True, exist_ok=True)
    _spec_path(name).write_text(json.dumps({"tiers": [str(t) for t in tiers], "costs": [float(c) for c in costs]}))
    with _LOCK:
        _COUNTS.pop(name, None)
    return {"route": name, "tiers": tiers}


@router.post("/routes/{name}/decide")
def decide(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "input" not in body:
        raise HTTPException(status_code=422, detail='body must be {"input": <text or record>}')
    spec = _read_spec(name)
    x = _coerce_input(body["input"])
    with _LOCK:
        counts = _COUNTS.setdefault(name, {t: 0 for t in [*spec["tiers"], "frontier"]})
    for tier in spec["tiers"]:
        label = _load(tier).decide(x)
        if label is not None:
            with _LOCK:
                counts[tier] = counts.get(tier, 0) + 1
            return {"label": label, "tier": tier, "escalate": False}
    with _LOCK:
        counts["frontier"] = counts.get("frontier", 0) + 1
    return {"label": None, "tier": "frontier", "escalate": True}


@router.get("/routes/{name}/report")
def report(name: str, user: User = Depends(require_user)) -> dict[str, Any]:
    spec = _read_spec(name)
    with _LOCK:
        counts = dict(_COUNTS.get(name, {}))
    names = [*spec["tiers"], "frontier"]
    answered = [int(counts.get(t, 0)) for t in names]
    n = sum(answered)
    costs = spec["costs"]
    realized = float(sum(a * c for a, c in zip(answered, costs)))
    frontier_only = float(n * costs[-1])
    return {
        "route": name,
        "requests": n,
        "tiers": [
            {"tier": t, "answered": a, "share": (a / n) if n else 0.0, "cost_per_request": c}
            for t, a, c in zip(names, answered, costs)
        ],
        "realized_cost": realized,
        "frontier_only_cost": frontier_only,
        "savings": frontier_only - realized,
    }
