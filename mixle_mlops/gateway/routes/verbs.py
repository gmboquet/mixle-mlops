"""Serving twins for the creation verbs (I1): /v1/create, /v1/uq, /v1/simulate, /v1/synthesize, /v1/skills.

Each pysparkplug verb gets an HTTP twin over a per-user model store, so a thin client can run the whole
loop remotely: CREATE a certified artifact from records, quantify its UQ, SIMULATE synthetic data from
it (with do-interventions when the artifact is a learned Bayesian network), SYNTHESIZE a verified
dataset against a declarative constraint, and register/find the artifact as a SKILL. Every stored model
carries its certificate summary and parameter fingerprint (the same round-trip discipline as /v1/pool).

Callables cannot cross HTTP, so ``/synthesize`` accepts a *declarative* verifier -- ``{"index": i,
"min":, "max":, "in": [...]}`` conditions -- which the server compiles to a predicate; the response
reports acceptance_rate and echoes the constraint so the client can re-check rows independently (the
verifier still travels with the data, in declarative form).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...accounts.models import User
from ...config import get_settings
from ..auth import require_user

router = APIRouter()


def _user_dir(user: User) -> Path:
    uid = str(getattr(user, "email", None) or getattr(user, "id", "anon")).replace("/", "_")
    d = Path(get_settings().registry_root) / "verbs" / uid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store_model(user: User, model: Any, meta: dict[str, Any]) -> str:
    from mixle.inference import param_fingerprint

    model_id = uuid.uuid4().hex[:12]
    payload = {"model_json": model.to_json(), "fingerprint": param_fingerprint(model), "meta": meta}
    (_user_dir(user) / f"{model_id}.json").write_text(json.dumps(payload))
    return model_id


def _load_model(user: User, model_id: str) -> tuple[Any, dict[str, Any]]:
    from mixle.utils.serialization import from_json

    p = _user_dir(user) / f"{model_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no model {model_id!r}")
    payload = json.loads(p.read_text())
    return from_json(payload["model_json"]), payload


def _rows(data: Any) -> list[Any]:
    if not isinstance(data, list) or not data:
        raise HTTPException(status_code=422, detail='body must include "data": [record, ...]')
    return [tuple(r) if isinstance(r, list) else r for r in data]


@router.post("/create")
def create_route(body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """CREATE: records -> a certified, stored model artifact (guarantee + calibration + fingerprint)."""
    from mixle.inference import create, data_fingerprint

    rows = _rows(body.get("data"))
    calibrate = body.get("calibrate")
    art = create(
        rows,
        calibrate=float(calibrate) if calibrate is not None else None,
        quantify_uq=bool(body.get("quantify_uq", False)),
        seed=int(body.get("seed", 0)),
    )
    meta = {
        "guarantee": art.certificate.guarantee.name,
        "why": art.why(),
        "is_calibrated": art.is_calibrated(),
        "strategy": art.strategy,
        "n": len(rows),
        "data_fingerprint": data_fingerprint(rows),  # the lineage edge back to the exact training data
    }
    model_id = _store_model(user, art.model, meta)
    return {"model_id": model_id, **meta}


@router.post("/uq")
def uq_route(body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """UQ: a stored model (+ its data) -> a credible interval on its predictive mean, or an honest
    'not quantifiable' when the model cannot be Laplace-flattened (never a fabricated interval)."""
    import numpy as np

    from mixle.inference import uq

    model, _payload = _load_model(user, str(body.get("model_id", "")))
    rows = _rows(body.get("data"))
    level = float(body.get("level", 0.9))

    def predictive_mean(m: Any) -> float:  # a generic readout: works for anything sampleable
        return float(np.mean([float(x) for x in m.sampler(seed=0).sample(200)]))

    try:
        result = uq(model, rows)
        lo, hi = result.credible_interval(predictive_mean, alpha=1.0 - level, n=200)
        return {
            "kind": result.kind,
            "method": result.method,
            "readout": "predictive_mean",
            "interval": [float(lo), float(hi)],
            "level": level,
        }
    except NotImplementedError as exc:
        return {"kind": None, "method": None, "note": f"not quantifiable: {exc}"}


@router.post("/simulate")
def simulate_route(body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """SIMULATE: draw synthetic records from a stored model; do-interventions for a learned BN."""
    from mixle.inference import simulate

    model, _payload = _load_model(user, str(body.get("model_id", "")))
    sim = simulate(model)
    interventions = body.get("interventions")
    iv = {int(k): v for k, v in interventions.items()} if isinstance(interventions, dict) else None
    try:
        rows = sim.run(int(body.get("n", 100)), interventions=iv, seed=int(body.get("seed", 0)))
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"n": len(rows), "rows": [list(r) if isinstance(r, tuple) else r for r in rows]}


def _compile_verifier(constraint: dict[str, Any] | None):
    """A declarative row constraint -> a predicate. None = accept everything (reported as such)."""
    if not constraint:
        return None
    idx = constraint.get("index")

    def field_of(x: Any) -> Any:
        return x if idx is None else x[int(idx)]

    def verify(x: Any) -> bool:
        v = field_of(x)
        if "in" in constraint:
            return v in set(constraint["in"])
        ok = True
        if "min" in constraint:
            ok = ok and float(v) >= float(constraint["min"])
        if "max" in constraint:
            ok = ok and float(v) <= float(constraint["max"])
        return ok

    return verify


@router.post("/synthesize")
def synthesize_route(body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """SYNTHESIZE: a verified dataset from a stored model or posted records, vs a declarative constraint."""
    from mixle.inference import synthesize

    if body.get("model_id"):
        source, _ = _load_model(user, str(body["model_id"]))
    else:
        source = _rows(body.get("data"))
    verify = _compile_verifier(body.get("constraint"))
    ds = synthesize(
        source,
        verify=verify,
        n=int(body.get("n", 50)),
        max_tries=int(body["max_tries"]) if body.get("max_tries") else None,
        seed=int(body.get("seed", 0)),
    )
    return {
        "n": len(ds),
        "acceptance_rate": ds.acceptance_rate,
        "n_rejected": ds.n_rejected,
        "constraint": body.get("constraint"),  # the verifier travels with the data, declaratively
        "rows": [list(r) if isinstance(r, tuple) else r for r in ds.inputs],
    }


@router.post("/skills")
def register_skill(body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """SKILL: register a stored model as a named, described, findable capability."""
    name = str(body.get("name", "")).strip()
    if not name or not body.get("model_id"):
        raise HTTPException(status_code=422, detail='body must include "name" and "model_id"')
    _model, payload = _load_model(user, str(body["model_id"]))  # existence + fingerprint check
    skills_path = _user_dir(user) / "skills.json"
    skills = json.loads(skills_path.read_text()) if skills_path.exists() else {}
    skills[name] = {
        "name": name,
        "description": str(body.get("description", "")),
        "tags": list(body.get("tags", [])),
        "model_id": body["model_id"],
        "guarantee": payload["meta"].get("guarantee"),  # the skill inherits the artifact's certificate
        "fingerprint": payload["fingerprint"],
    }
    skills_path.write_text(json.dumps(skills))
    return skills[name]


@router.get("/skills")
def find_skills(query: str | None = None, user: User = Depends(require_user)) -> dict[str, Any]:
    """Find skills by lexical overlap of the query with name/description/tags (all skills when no query)."""
    import re

    skills_path = _user_dir(user) / "skills.json"
    skills = list(json.loads(skills_path.read_text()).values()) if skills_path.exists() else []
    if not query:
        return {"skills": skills}
    q = set(re.findall(r"[a-z0-9]+", query.lower()))

    def score(s: dict[str, Any]) -> float:
        hay = set(re.findall(r"[a-z0-9]+", f"{s['name']} {s['description']} {' '.join(s['tags'])}".lower()))
        return len(q & hay) / len(q) if q else 0.0

    ranked = sorted(((score(s), s) for s in skills), key=lambda t: -t[0])
    return {"skills": [s for sc, s in ranked if sc > 0]}


@router.get("/lineage/{model_id}")
def lineage_route(model_id: str, user: User = Depends(require_user)) -> dict[str, Any]:
    """The lineage graph for a stored artifact (I3): data -> model -> the skills that expose it.

    One query answers "where did this model come from, and what depends on it": the exact training-data
    fingerprint recorded at create-time, the parameter fingerprint (bit-identity), the certificate
    summary, and every registered skill that references the model. Deleting or replacing the model can
    be judged against its dependents instead of guessed."""
    _model, payload = _load_model(user, model_id)
    skills_path = _user_dir(user) / "skills.json"
    skills = list(json.loads(skills_path.read_text()).values()) if skills_path.exists() else []
    dependents = [s["name"] for s in skills if s.get("model_id") == model_id]
    return {
        "model_id": model_id,
        "fingerprint": payload["fingerprint"],
        "data_fingerprint": payload["meta"].get("data_fingerprint"),
        "guarantee": payload["meta"].get("guarantee"),
        "n_training_rows": payload["meta"].get("n"),
        "skills": dependents,
        "edges": (
            [{"from": "data:" + str(payload["meta"].get("data_fingerprint"))[:12], "to": model_id, "kind": "fit"}]
            + [{"from": model_id, "to": f"skill:{s}", "kind": "exposes"} for s in dependents]
        ),
    }
