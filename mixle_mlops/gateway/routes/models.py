"""Model catalog (OpenAI-compatible /v1/models) — lists hosted mixle + LLM + composite models and their
capabilities, so a client can discover which support the mixle distribution/decision routes."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...accounts.models import User
from ...config import get_settings
from ..auth import current_user, require_user

router = APIRouter()


@router.get("/models")
async def list_models(request: Request, user: User | None = Depends(current_user)):
    return {"object": "list", "data": [m.model_dump() for m in request.app.state.registry.list()]}


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request, user: User | None = Depends(current_user)):
    registry = request.app.state.registry
    if not registry.has(model_id):
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
    return registry.get(model_id).info().model_dump()


class LoadModelBody(BaseModel):
    name: str  # the artifact's directory name under registry_root (== the fine-tune job's served model id)


@router.post("/models/load")
async def load_model(
    body: LoadModelBody, request: Request, user: User = Depends(require_user),
) -> dict[str, str]:
    """Load an already-trained artifact from ``registry_root`` into the LIVE registry, so it becomes
    servable at ``/v1/models/{name}``, ``/v1/chat/completions``, and ``/v1/mixle/*`` without a gateway
    restart. Reads the ``metadata.json`` a completed fine-tune (or ``mixle_mlops.compute.launch``) job
    already writes -- doesn't train or spend anything itself, so it's gated on being an admin (like
    substrate's promotion approval) rather than requiring a fresh grant per model.

    Currently supports the ``llm`` backend only (a base model + an optional LoRA/QLoRA adapter, loaded
    via ``load_local_engine``); the ``mixle``/``structured`` backends are registered through their own
    existing paths (``run_structured_finetune`` registers synchronously; a ``mixle``-backend artifact is
    a mixle model file, loaded the way any other mixle model is)."""
    if not bool(getattr(user, "is_admin", False)):
        raise HTTPException(status_code=403, detail="only an admin may load a model into the live registry")

    root = get_settings().registry_root / body.name
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"no registered artifact named {body.name!r} under {root.parent}")
    try:
        meta = json.loads(meta_path.read_text())
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"unreadable metadata for {body.name!r}: {e}") from e

    backend = meta.get("backend")
    if backend != "llm":
        raise HTTPException(
            status_code=400,
            detail=f"POST /v1/models/load only serves the 'llm' backend today (got {backend!r})",
        )
    base_model = meta.get("base_model")
    if not base_model:
        raise HTTPException(status_code=500, detail=f"{body.name!r}'s metadata has no base_model recorded")
    adapter_dir = meta.get("artifact")  # run_local/launch always resolve this to a local adapter directory

    try:
        from ...models.local_engine import load_local_engine

        served = load_local_engine(body.name, [base_model], adapter_path=adapter_dir)
    except Exception as e:  # missing local extras, bad/missing adapter dir, etc. -> a clear 400, not a 500 crash
        raise HTTPException(status_code=400, detail=f"could not load {body.name!r}: {type(e).__name__}: {e}") from e

    request.app.state.registry.register(served)
    return {"model": body.name, "base_model": base_model, "adapter": adapter_dir or ""}
