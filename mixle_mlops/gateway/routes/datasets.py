"""Dataset-generation routes — mixle's home turf: a generative library makes labeled data with verifiable
labels.

``POST /v1/datasets/generate`` ({source, model, n, schema?, prompt?, format, seed?, columns?}) pulls the
model from the registry, samples/drives it, exports the bytes into the blob store, records a
:class:`~mixle_mlops.datasets.models.DatasetArtifact`, and returns the artifact ref. ``GET /v1/datasets/{id}``
fetches a previously-generated artifact's metadata. Both require an authenticated user.

Wiring (integrator):
  * ``from .routes import datasets`` then ``app.include_router(datasets.router, prefix="/v1", tags=["datasets"])``
    in ``gateway/app.py``.
  * optional: ``from ..datasets import models as _datasets  # noqa: F401`` inside
    ``storage/db.init_db`` so the table is created up-front (the route also creates it defensively).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Session, select
from starlette.concurrency import run_in_threadpool

from ...accounts.models import User
from ...datasets.code_tasks import (
    DEFAULT_FIELD_POOL,
    LLMTeacher,
    ReferenceTeacher,
    build_tasks,
    harvest,
)
from ...datasets.export import export_dataset
from ...datasets.generate import DatasetSpec, GeneratedDataset, generate_dataset
from ...datasets.models import DatasetArtifact
from ...multimodal.store import get_blob_store
from ...storage.db import get_engine, get_session
from ..auth import require_user

router = APIRouter(prefix="/datasets", tags=["datasets"])

_table_ready = False


def _ensure_table() -> None:
    """Idempotently create the DatasetArtifact table (so this works without editing init_db)."""
    global _table_ready
    if _table_ready:
        return
    SQLModel.metadata.create_all(get_engine(), tables=[DatasetArtifact.__table__])
    _table_ready = True


class GenerateBody(BaseModel):
    source: str = "mixle"  # "mixle" | "llm"
    model: str
    n: int = Field(default=100, ge=1, le=100_000)
    seed: int = 0
    schema_: dict[str, str] | None = Field(default=None, alias="schema")
    prompt: str | None = None
    format: str = "jsonl"  # "jsonl" | "csv" | "parquet"
    columns: list[str] | None = None

    model_config = {"populate_by_name": True}


def _persist(
    session: Session, user: User, dataset: GeneratedDataset, artifact_ref: dict[str, Any], fmt: str
) -> DatasetArtifact:
    row = DatasetArtifact(
        user_id=getattr(user, "id", None),
        source=dataset.source,
        model=dataset.model,
        fmt=fmt,
        n_rows=dataset.n_rows,
        seed=dataset.seed,
        prompt=dataset.prompt,
        blob_id=artifact_ref.get("id"),
        blob_url=artifact_ref.get("url"),
        schema_def=DatasetArtifact.encode_schema(dataset.schema),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


@router.post("/generate")
async def generate(
    body: GenerateBody,
    request: Request,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    """Generate a labeled dataset, export it to the blob store, and return the artifact ref."""
    _ensure_table()
    registry = request.app.state.registry
    spec = DatasetSpec(
        source=body.source,
        model=body.model,
        n=body.n,
        seed=body.seed,
        schema=body.schema_,
        prompt=body.prompt,
        fmt=body.format,
        columns=body.columns,
    )
    try:
        dataset = await generate_dataset(spec, registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        artifact_ref = export_dataset(dataset, body.format, store=get_blob_store())
    except ValueError as exc:  # unknown format
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:  # missing optional dep (parquet)
        raise HTTPException(status_code=501, detail=str(exc))

    row = _persist(session, user, dataset, artifact_ref, body.format)
    result = row.to_dict()
    result["artifact"] = artifact_ref
    return result


class CodeTasksBody(BaseModel):
    """Harvest an execution-verified ``(page -> parser code)`` SFT dataset."""

    n_tasks: int = Field(default=200, ge=1, le=5000)
    teacher: str = "reference"  # "reference" (rule-based) | a served model id (an LLM teacher)
    pool: dict[str, str] | None = None  # field -> "str"|"int"|"float"; default DEFAULT_FIELD_POOL
    fields_per_task: int = Field(default=3, ge=1, le=12)
    n_rows: int = Field(default=4, ge=1, le=100)
    templates: list[str] | None = None  # subset of ("table","divs","list")
    noise: float = Field(default=0.5, ge=0.0, le=1.0)
    attempts: int = Field(default=2, ge=1, le=8)
    seed: int = 0


@router.post("/code_tasks")
async def code_tasks(
    body: CodeTasksBody,
    request: Request,
    authorization: str | None = Header(default=None),
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    """Manufacture pages from known records, have a teacher write parser code, keep only what EXECUTES true.

    ``teacher="reference"`` uses the built-in rule-based teacher (no LLM, always verifies -- proves the
    plumbing and mints data for free). Any other value is a served model id: the platform harvests its own
    training data by having a model it serves write the code, with the execution verifier -- not the
    model's word -- deciding what enters the dataset. The JSONL lands in the blob store, ready to hand to
    ``POST /v1/fine_tunes``.
    """
    _ensure_table()
    try:
        tasks = build_tasks(
            body.n_tasks,
            pool=body.pool,
            fields_per_task=body.fields_per_task,
            n_rows=body.n_rows,
            templates=body.templates,
            noise=body.noise,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if body.teacher == "reference":
        teacher: Any = ReferenceTeacher()
    else:
        token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
        base_url = str(request.base_url).rstrip("/") + "/v1"
        teacher = LLMTeacher(body.teacher, base_url=base_url, api_key=token)

    # harvest runs sync httpx + subprocesses; keep it off the event loop
    ds = await run_in_threadpool(harvest, teacher, tasks, attempts=body.attempts)

    ref = get_blob_store().put(ds.jsonl_bytes(), filename="code_tasks.jsonl", content_type="application/x-ndjson")
    row = DatasetArtifact(
        user_id=getattr(user, "id", None),
        source="code_tasks",
        model=(None if body.teacher == "reference" else body.teacher),
        fmt="jsonl",
        n_rows=len(ds.jsonl_rows()),
        seed=body.seed,
        blob_id=ref.id,
        blob_url=ref.url,
        schema_def=DatasetArtifact.encode_schema(body.pool or DEFAULT_FIELD_POOL),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    result = row.to_dict()
    result["artifact"] = ref.to_dict()
    result["harvest"] = {
        "attempted": ds.attempted,
        "verified": len(ds.trajectories),
        "pairs": len(ds.jsonl_rows()),
        "yield_rate": ds.yield_rate,
        "repairs": len(ds.repairs),
    }
    return result


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    """Metadata for a previously-generated dataset artifact."""
    _ensure_table()
    row = session.get(DatasetArtifact, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id!r} not found")
    return row.to_dict()


@router.get("")
async def list_datasets(
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    """List the authenticated user's generated datasets (most recent first)."""
    _ensure_table()
    stmt = (
        select(DatasetArtifact)
        .where(DatasetArtifact.user_id == getattr(user, "id", None))
        .order_by(DatasetArtifact.created_at.desc())
    )
    rows = session.exec(stmt).all()
    return {"object": "list", "data": [r.to_dict() for r in rows]}
