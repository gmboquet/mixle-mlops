"""The pool plane over HTTP -- job gateway (H2), bit-exact round-trip (H3), quotas + spend (H4).

The library rails exist in ``mixle.pool`` (budget-reject, billable-confirm); this route puts them behind
the gateway so a laptop can submit work the 99/1 topology says may offload, and get the artifact back
with proof it round-tripped intact:

  * ``POST /v1/pool/jobs``      -- submit a job manifest. Rails run BEFORE any work: over-budget ->
        rejected; a priced job (``est_cost > 0``) requires ``confirm: true`` (spend is never implicit);
        a job that would push the user past their quota -> rejected. v1 executes ``spec.op == "fit"``
        server-side (mixle ``optimize`` over the posted records, automatic families).
  * ``GET  /v1/pool/jobs``      -- this user's queue (most recent first).
  * ``GET  /v1/pool/jobs/{id}`` -- one job's status/cost/reason.
  * ``GET  /v1/pool/jobs/{id}/artifact`` -- the ROUND-TRIP primitive: the fitted artifact serialized
        (``to_json``) plus its canonical parameter fingerprint, so the client can reload and verify
        bit-exactness (``mixle.inference.param_fingerprint`` of the reload equals the served one) with
        the job id + reason as provenance.
  * ``GET  /v1/pool/spend``     -- the spend ledger: total spent, quota, remaining (H4).

Per-user state persists under ``{registry_root}/pool/{user}/`` (jobs.json + artifacts/). Every outcome
appends realized cost to the ledger only when the job actually ran -- rejected jobs cost nothing.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...accounts.models import User
from ...config import get_settings
from ..auth import require_user

router = APIRouter()

DEFAULT_QUOTA = 100.0  # dollars; a deliberate soft default -- deployments override per user


def _user_dir(user: User) -> Path:
    uid = str(getattr(user, "email", None) or getattr(user, "id", "anon")).replace("/", "_")
    d = Path(get_settings().registry_root) / "pool" / uid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_jobs(user: User) -> list[dict[str, Any]]:
    p = _user_dir(user) / "jobs.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _save_jobs(user: User, jobs: list[dict[str, Any]]) -> None:
    (_user_dir(user) / "jobs.json").write_text(json.dumps(jobs))


def _spent(jobs: list[dict[str, Any]]) -> float:
    return float(sum(j.get("cost", 0.0) for j in jobs if j.get("status") == "done"))


def _quota(user: User) -> float:
    return float(getattr(user, "pool_quota", None) or DEFAULT_QUOTA)


def _run_fit(spec: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    """Execute a v1 'fit' job server-side: optimize over the posted records, automatic families."""
    import numpy as np

    from mixle.inference import certify, optimize, param_fingerprint

    data = spec.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError('spec must include "data": [record, ...]')
    rows = [tuple(r) if isinstance(r, list) else r for r in data]
    seed = int(spec.get("seed", 0))
    model = optimize(rows, out=None, max_its=int(spec.get("max_its", 25)), rng=np.random.RandomState(seed))
    fp = param_fingerprint(model)
    artifact_path.write_text(json.dumps({"model_json": model.to_json(), "fingerprint": fp}))
    cert = certify(model)
    return {"fingerprint": fp, "guarantee": cert.guarantee.name, "n": len(rows)}


@router.post("/pool/jobs")
def submit_job(body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    spec = body.get("spec") or {}
    est_cost = float(body.get("est_cost", 0.0))
    budget = float(body.get("budget", float("inf")))
    confirm = bool(body.get("confirm", False))
    jobs = _load_jobs(user)

    job: dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "kind": str(body.get("kind", "verb")),
        "reason": str(body.get("reason", "")),
        "est_cost": est_cost,
        "budget": None if budget == float("inf") else budget,
        "status": "queued",
        "cost": 0.0,
        "submitted_at": time.time(),
    }

    # -- the rails, before any work (the mixle.pool discipline, server-side) -----------------------
    spent = _spent(jobs)
    quota = _quota(user)
    if est_cost > budget:
        job.update(status="rejected", reason_out=f"estimated cost {est_cost} exceeds budget {budget}")
    elif est_cost > 0 and not confirm:
        job.update(status="rejected", reason_out="priced job requires confirm=true (spend is never implicit)")
    elif spent + est_cost > quota:
        job.update(
            status="rejected",
            reason_out=f"quota: spent {spent:.2f} + est {est_cost:.2f} exceeds quota {quota:.2f}",
        )
    else:
        op = str(spec.get("op", ""))
        try:
            if op == "fit":
                artifact_path = _user_dir(user) / "artifacts"
                artifact_path.mkdir(exist_ok=True)
                summary = _run_fit(spec, artifact_path / f"{job['id']}.json")
                job.update(status="done", cost=est_cost, summary=summary)
            else:
                job.update(status="rejected", reason_out=f'unknown spec.op {op!r}; v1 supports "fit"')
        except Exception as exc:  # noqa: BLE001 - a failed job is a recorded outcome, not a 500
            job.update(status="error", reason_out=str(exc))

    jobs.append(job)
    _save_jobs(user, jobs)
    return job


@router.get("/pool/jobs")
def list_jobs(user: User = Depends(require_user)) -> dict[str, Any]:
    jobs = _load_jobs(user)
    return {"jobs": sorted(jobs, key=lambda j: -j["submitted_at"])}


@router.get("/pool/jobs/{job_id}")
def get_job(job_id: str, user: User = Depends(require_user)) -> dict[str, Any]:
    for j in _load_jobs(user):
        if j["id"] == job_id:
            return j
    raise HTTPException(status_code=404, detail=f"no job {job_id!r}")


@router.get("/pool/jobs/{job_id}/artifact")
def get_artifact(job_id: str, user: User = Depends(require_user)) -> dict[str, Any]:
    """The round-trip primitive (H3): the serialized artifact + its fingerprint + provenance."""
    job = get_job(job_id, user)
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"job {job_id!r} is {job.get('status')}, no artifact")
    p = _user_dir(user) / "artifacts" / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="artifact file missing")
    payload = json.loads(p.read_text())
    payload["provenance"] = {"job_id": job_id, "kind": job["kind"], "reason": job["reason"]}
    return payload


@router.get("/pool/spend")
def spend(user: User = Depends(require_user)) -> dict[str, Any]:
    jobs = _load_jobs(user)
    spent = _spent(jobs)
    quota = _quota(user)
    return {
        "spent": round(spent, 4),
        "quota": quota,
        "remaining": round(max(0.0, quota - spent), 4),
        "n_done": sum(1 for j in jobs if j["status"] == "done"),
        "n_rejected": sum(1 for j in jobs if j["status"] == "rejected"),
    }
