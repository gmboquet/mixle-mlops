"""Serve every solve-shape artifact — classification, regression, multi-label, structured — one route.

Artifacts saved by ``Solution.save`` / ``RegressionSolution.save`` / ``MultiLabelSolution.save`` /
``StructuredSolution.save`` drop under ``{registry_root}/solutions/{name}``; the artifact KIND is
sniffed from what's on disk, so the caller never declares it. The serving contract is the same
teacher-free half everywhere: the local decision when the calibrated gate allows it, ``escalate=true``
when it doesn't — the caller runs the original code and posts the answer back as harvest.

  * ``GET  /solutions``                    — ``{"solutions": {name: kind}}``.
  * ``POST /solutions/{name}/decide``      — ``{"input": ...}`` ->
        classification: ``{"kind", "label", "escalate"}``      regression: ``{"kind", "value", "escalate"}``
        multilabel:     ``{"kind", "labels", "escalate"}``      structured: ``{"kind", "output", "escalate"}``
  * ``POST /solutions/{name}/feedback``    — ``{"input": ..., "answer": ...}`` -> harvested count.
  * ``GET  /solutions/{name}/verification`` — the shape's trust surface straight from the manifest(s):
        was it verified, at what alpha, how tight/agreeing on holdout — answerable from the endpoint alone.
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
from ._paths import safe_join
from .tasks import _coerce_input

router = APIRouter()

_CACHE: dict[str, tuple[float, str, Any]] = {}
_LOCK = threading.Lock()


def _root() -> Path:
    return Path(get_settings().registry_root) / "solutions"


def _never_teacher(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError("the serving gateway has no teacher; escalation is the caller's job")


def _sniff(path: Path) -> tuple[str, Path]:
    """The artifact kind and its freshness stamp file, from what's actually on disk."""
    if (path / "structured.json").exists():
        return "structured", path / "structured.json"
    manifest = path / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text()).get("meta", {})
        if "regress" in meta:
            return "regression", manifest
        if "multilabel" in meta:
            return "multilabel", manifest
        return "classification", manifest
    raise HTTPException(status_code=404, detail=f"no deployed solution at {path.name!r}")


def _load(name: str) -> tuple[str, Any]:
    path = safe_join(_root(), name)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"no deployed solution named {name!r}")
    kind, stamp_file = _sniff(path)
    stamp = stamp_file.stat().st_mtime
    with _LOCK:
        hit = _CACHE.get(name)
        if hit is not None and hit[0] == stamp:
            return hit[1], hit[2]
    try:
        from mixle.task import CalibratedTaskModel, MultiLabelSolution, RegressionSolution, StructuredSolution

        if kind == "structured":
            obj: Any = StructuredSolution.load(str(path), _never_teacher)
        elif kind == "regression":
            obj = RegressionSolution.load(str(path), _never_teacher)
        elif kind == "multilabel":
            obj = MultiLabelSolution.load(str(path), _never_teacher)
        else:
            obj = CalibratedTaskModel.load(str(path))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken artifact must 500 with the reason
        raise HTTPException(status_code=500, detail=f"failed to load solution {name!r}: {exc}") from exc
    with _LOCK:
        _CACHE[name] = (stamp, kind, obj)
    return kind, obj


@router.get("/solutions")
def list_solutions(user: User = Depends(require_user)) -> dict[str, Any]:
    root = _root()
    out: dict[str, str] = {}
    if root.is_dir():
        for p in sorted(root.iterdir()):
            try:
                out[p.name] = _sniff(p)[0]
            except HTTPException:
                continue
    return {"solutions": out}


@router.post("/solutions/{name}/decide")
def decide(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "input" not in body:
        raise HTTPException(status_code=422, detail='body must be {"input": <text or record>}')
    kind, obj = _load(name)
    x = _coerce_input(body["input"])
    if kind == "classification":
        label = obj.decide(x)
        return {"kind": kind, "label": label, "escalate": label is None}
    if kind == "regression":
        if obj.answers_locally:
            return {"kind": kind, "value": float(obj._predict([x])[0]), "qhat": float(obj.qhat), "escalate": False}
        return {"kind": kind, "value": None, "escalate": True}
    if kind == "multilabel":
        labels = obj.try_local(x)
        return {"kind": kind, "labels": labels, "escalate": labels is None}
    out = obj.try_local(x)  # structured
    return {"kind": kind, "output": out, "escalate": out is None}


def _verification_of(path: Path) -> dict[str, Any]:
    """The shape's trust surface read straight from the manifest(s) — no model load needed."""
    kind, _stamp = _sniff(path)
    if kind == "structured":
        schema = json.loads((path / "structured.json").read_text())
        fields: dict[str, Any] = {}
        for key in schema.get("cat", []):
            meta = json.loads((path / "cat" / key / "manifest.json").read_text()).get("meta", {})
            fields[key] = {"kind": "categorical", **(meta.get("solve", {}).get("verification") or {})}
        for key in schema.get("num", []):
            m = json.loads((path / "num" / key / "manifest.json").read_text()).get("meta", {}).get("regress", {})
            fields[key] = {
                "kind": "numeric",
                **{k: m.get(k) for k in ("qhat", "tol", "alpha", "holdout_mae")},
                "answers_locally": bool(m.get("qhat", float("inf")) <= m.get("tol", 0.0)),
            }
        return {"kind": kind, "fields": fields}
    meta = json.loads((path / "manifest.json").read_text()).get("meta", {})
    if kind == "regression":
        m = meta.get("regress", {})
        return {
            "kind": kind,
            **{k: m.get(k) for k in ("qhat", "tol", "alpha", "holdout_mae")},
            "answers_locally": bool(m.get("qhat", float("inf")) <= m.get("tol", 0.0)),
        }
    if kind == "multilabel":
        m = meta.get("multilabel", {})
        return {"kind": kind, **{k: m.get(k) for k in ("labels", "alpha", "holdout_set_agreement")}}
    solve_meta = meta.get("solve", {})
    ver = solve_meta.get("verification")
    if not ver:
        raise HTTPException(status_code=404, detail="artifact carries no verification record")
    return {"kind": kind, "ood": solve_meta.get("ood"), "verification": ver}


@router.get("/solutions/{name}/verification")
def verification(name: str, user: User = Depends(require_user)) -> dict[str, Any]:
    path = safe_join(_root(), name)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"no deployed solution named {name!r}")
    return _verification_of(path)


@router.post("/solutions/{name}/feedback")
def feedback(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "input" not in body or "answer" not in body:
        raise HTTPException(status_code=422, detail='body must be {"input": ..., "answer": ...}')
    path = safe_join(_root(), name)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"no deployed solution named {name!r}")
    harvested = path / "harvested.jsonl"
    with _LOCK:
        with open(harvested, "a") as f:
            f.write(json.dumps({"input": body["input"], "answer": body["answer"]}) + "\n")
        count = sum(1 for _ in open(harvested))
    return {"harvested": count}
