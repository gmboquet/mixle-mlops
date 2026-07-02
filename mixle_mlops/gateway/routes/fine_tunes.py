"""Fine-tuning routes — turn a labeled dataset into a hosted model through the gateway.

``POST /v1/fine_tunes`` ({backend, model?, records, label_field|labels, n_components?, min_gain?}) trains a model
and (for the ``structured`` backend) registers it live so it appears in ``/v1/models`` and answers ``/v1/chat`` +
``/v1/mixle/{predict,score}`` immediately. ``GET /v1/fine_tunes`` lists the caller's jobs; ``GET
/v1/fine_tunes/{id}`` fetches one; ``POST /v1/fine_tunes/{id}/cancel`` cancels a non-terminal job. All require an
authenticated user.

The ``structured`` backend runs locally (no GPU, no torch) via ``mixle.task.distill_structured``. The ``llm`` and
``mixle`` backends return the offline vast.ai training *plan* (status ``planned``); actually renting the GPU is the
operator's keyed ``mixle_mlops.compute.launch`` step.

Wiring (integrator): ``from .routes import fine_tunes`` then
``app.include_router(fine_tunes.router, prefix="/v1", tags=["fine_tunes"])`` in ``gateway/app.py``.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Session

from ...accounts.models import User
from ...config import get_settings
from ...storage.db import get_engine, get_session
from ...training import models as ftmodels
from ...training import service
from ...training.models import FineTuneJob
from ..auth import require_user

router = APIRouter(prefix="/fine_tunes", tags=["fine_tunes"])

_table_ready = False


def _ensure_table() -> None:
    global _table_ready
    if _table_ready:
        return
    SQLModel.metadata.create_all(get_engine(), tables=[FineTuneJob.__table__])
    _table_ready = True


class FineTuneBody(BaseModel):
    backend: str = "structured"                          # "structured" | "llm" | "mixle"
    model: str | None = None                             # served model id (default: the job id)
    records: list[Any] = Field(default_factory=list)     # feature records (dicts or tuples) for the structured backend
    label_field: str | None = None                       # key in each record holding the target label
    labels: list[str] | None = None                      # OR an explicit parallel label list
    n_components: int = 1                                 # >1 -> a latent-regime mixture-of-trees student
    min_gain: float = 1.0                                 # description-length gain an edge must clear to be kept
    # llm / mixle GPU backends (returns a plan; launching is the operator's keyed step)
    base_model: str | None = None
    dataset: str | None = None
    script: str | None = None
    repo: str | None = None
    workdir: str | None = None
    gpu: str = "RTX_4090"
    epochs: float = 1.0
    qlora: bool = False


@router.post("")
def create_fine_tune(
    body: FineTuneBody,
    request: Request,
    background: BackgroundTasks,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    _ensure_table()
    model_name = body.model or ""
    job = FineTuneJob(user_id=user.id, backend=body.backend, model=model_name or "pending",
                      base_model=body.base_model)
    if not model_name:
        job.model = f"ft:{job.id}"                        # a stable served id derived from the job id
    session.add(job)
    session.commit()
    session.refresh(job)

    if body.backend == "structured":
        if not body.records:
            _fail(session, job, "structured backend requires records=[...]")
            raise HTTPException(status_code=400, detail="structured backend requires records=[...]")
        registry = request.app.state.registry
        background.add_task(
            service.run_structured_finetune,
            job.id, engine=get_engine(), registry=registry,
            records=body.records, label_field=body.label_field, labels=body.labels,
            n_components=body.n_components, min_gain=body.min_gain,
            artifact_root=_artifact_root(),
        )
        return job.to_dict()

    if body.backend in ("llm", "mixle"):
        try:
            plan = service.plan_gpu_job(
                name=job.model, backend=body.backend, base_model=body.base_model,
                dataset=body.dataset, script=body.script, repo=body.repo, workdir=body.workdir,
                gpu=body.gpu, epochs=body.epochs, qlora=body.qlora,
            )
        except Exception as e:  # invalid job spec -> record + 400
            _fail(session, job, f"{type(e).__name__}: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e
        service._finish(session, job, ftmodels.PLANNED,
                        metrics_json=FineTuneJob.encode_metrics({"backend": body.backend, "plan": plan}))
        return job.to_dict()

    _fail(session, job, f"unknown backend {body.backend!r}")
    raise HTTPException(status_code=400, detail=f"unknown backend {body.backend!r}")


@router.get("")
def list_fine_tunes(
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    _ensure_table()
    jobs = service.user_jobs(session, user.id)
    return {"object": "list", "data": [j.to_dict() for j in jobs]}


@router.get("/{job_id}")
def get_fine_tune(
    job_id: str,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    _ensure_table()
    job = _owned_job(session, job_id, user)
    return job.to_dict()


@router.post("/{job_id}/cancel")
def cancel_fine_tune(
    job_id: str,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    _ensure_table()
    job = _owned_job(session, job_id, user)
    if job.status in ftmodels.TERMINAL:
        raise HTTPException(status_code=409, detail=f"job already {job.status}")
    service._finish(session, job, ftmodels.CANCELLED)
    return job.to_dict()


def _artifact_root() -> str:
    import os

    return os.path.join(get_settings().data_dir, "finetunes")


def _owned_job(session: Session, job_id: str, user: User) -> FineTuneJob:
    job = session.get(FineTuneJob, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="fine-tune job not found")
    return job


def _fail(session: Session, job: FineTuneJob, message: str) -> None:
    service._finish(session, job, ftmodels.FAILED, error=message)
