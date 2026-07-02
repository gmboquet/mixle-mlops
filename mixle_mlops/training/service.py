"""Train a fine-tune job and serve the result.

``run_structured_finetune`` is the local, no-GPU path: distil a tiny structured probabilistic classifier from a
labeled dataset (``mixle.task.distill_structured_from_labels``), persist it as a json artifact, and register a
``TaskCascadeAdapter`` into the live registry so it is instantly hosted at ``/v1/models`` + ``/v1/chat`` +
``/v1/mixle/{predict,score}``. It updates the :class:`FineTuneJob` row through its lifecycle and never raises into
the caller -- a failure is recorded on the row. ``plan_gpu_job`` produces the offline vast.ai plan for the
``llm``/``mixle`` backends.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from . import models as m
from .models import FineTuneJob


def _split_labels(
    records: list[Any], label_field: str | None, labels: list[str] | None
) -> tuple[list[Any], list[str]]:
    """Return (feature_records, label_list). Either ``labels`` is given in parallel, or ``label_field`` names the
    target key to strip out of each dict record (the rest are features)."""
    if labels is not None:
        if len(labels) != len(records):
            raise ValueError("labels length must match records length")
        return list(records), [str(y) for y in labels]
    if label_field is not None:
        feats, ys = [], []
        for r in records:
            if not isinstance(r, dict) or label_field not in r:
                raise ValueError(f"label_field {label_field!r} missing from a record")
            row = {k: v for k, v in r.items() if k != label_field}
            feats.append(row)
            ys.append(str(r[label_field]))
        return feats, ys
    raise ValueError("provide either labels=[...] or label_field=...")


def train_structured(
    records: list[Any],
    *,
    label_field: str | None = None,
    labels: list[str] | None = None,
    n_components: int = 1,
    min_gain: float = 1.0,
    task: str = "",
) -> tuple[Any, dict[str, Any]]:
    """Distil a structured classifier from labeled records; return (TaskModel, metrics). Pure mixle-core, no torch."""
    from mixle.task import distill_structured_from_labels

    feats, ys = _split_labels(records, label_field, labels)
    if len({*ys}) < 2:
        raise ValueError("need at least two distinct labels to train a classifier")
    student = distill_structured_from_labels(
        feats, ys, n_components=n_components, min_gain=min_gain, task=task or "fine-tuned structured classifier"
    )
    metrics = {
        "backend": "structured",
        "n_examples": len(feats),
        "labels": student.meta.get("labels", sorted({*ys})),
        "edges": student.meta.get("edges", []),
        "train_agreement": student.meta.get("train_agreement"),
        "n_components": n_components,
    }
    return student, metrics


def _serve(registry: Any, name: str, student: Any) -> None:
    """Register the trained student as a hosted model (idempotent overwrite of the same id)."""
    from ..models.task_cascade import TaskCascadeAdapter

    registry.register(TaskCascadeAdapter(name, student))


def _finish(session: Session, job: FineTuneJob, status: str, **fields: Any) -> None:
    job.status = status
    for k, v in fields.items():
        setattr(job, k, v)
    if status in m.TERMINAL or status == m.PLANNED:
        job.finished_at = datetime.now(timezone.utc)
    session.add(job)
    session.commit()


def run_structured_finetune(
    job_id: str,
    *,
    engine: Any,
    registry: Any,
    records: list[Any],
    label_field: str | None = None,
    labels: list[str] | None = None,
    n_components: int = 1,
    min_gain: float = 1.0,
    artifact_root: str = "./trained",
) -> None:
    """Background entry point: train the structured student, persist it, register it, and update the job row.

    Opens its own DB session (the request's is already closed). Records any failure on the row rather than raising,
    so a bad request can never crash the worker."""
    with Session(engine) as session:
        job = session.get(FineTuneJob, job_id)
        if job is None or job.status == m.CANCELLED:
            return
        _finish(session, job, m.RUNNING)
        try:
            student, metrics = train_structured(
                records, label_field=label_field, labels=labels,
                n_components=n_components, min_gain=min_gain, task=f"fine-tune {job.model}",
            )
            path = os.path.join(artifact_root, job_id)
            os.makedirs(artifact_root, exist_ok=True)
            student.save(path)
            _serve(registry, job.model, student)
            _finish(session, job, m.SUCCEEDED, artifact_path=path,
                    metrics_json=FineTuneJob.encode_metrics(metrics))
        except Exception as e:  # noqa: BLE001 - a training failure is data, recorded on the row
            _finish(session, job, m.FAILED, error=f"{type(e).__name__}: {e}")


def plan_gpu_job(
    *,
    name: str,
    backend: str,
    base_model: str | None = None,
    dataset: str | None = None,
    script: str | None = None,
    repo: str | None = None,
    workdir: str | None = None,
    gpu: str = "RTX_4090",
    epochs: float = 1.0,
    qlora: bool = False,
) -> dict[str, Any]:
    """Produce the offline vast.ai plan for a GPU fine-tune (no spend). Launching is the operator's keyed step
    (``mixle_mlops.compute.launch`` with ``MIXLE_VAST_API_KEY``), surfaced here so the REST client sees exactly
    what would run."""
    from ..compute import TrainingJob
    from ..compute.launcher import plan

    job = TrainingJob(
        name=name, backend=backend, mode=("onstart" if repo else "ssh"),
        base_model=base_model, dataset=dataset, script=script, repo=repo, workdir=workdir,
        gpu=gpu, epochs=epochs, qlora=qlora,
    )
    return plan(job)


def user_jobs(session: Session, user_id: str | None) -> list[FineTuneJob]:
    stmt = select(FineTuneJob).where(FineTuneJob.user_id == user_id).order_by(FineTuneJob.created_at.desc())
    return list(session.exec(stmt))
