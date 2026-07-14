"""Structured-knowledge routes (M1c, IC-13): query the federated knowledge-bundle store, fetch a bundle by
id, and hand a bundle off to a target model for a verified `KnowledgeDelta` write-back.

  * ``POST /v1/knowledge/query``          — federate the caller's document/conversation retriever (M1b) into
    one ranked, deduped `KnowledgeBundle` and persist it.
  * ``GET  /v1/knowledge/bundles/{id}``   — fetch a bundle (optionally a specific revision), access-filtered.
  * ``POST /v1/knowledge/handoff``        — render + hand ``bundle_id`` to ``target_model`` (M1c's `handoff`),
    requiring a structured delta back; applies it (M2a) and returns the resulting bundle + receipt.

All routes require an authenticated caller (``Depends(require_user)``); reads/writes are gated through a
`CallerScope` built from the caller's own identity/admin flag (mirrors ``gateway/routes/rag.py``'s
per-user scoping). Wiring (integrator): ``app.include_router(knowledge.router, prefix="/v1",
tags=["knowledge"])`` in ``gateway/app.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel

from ...accounts.models import User
from ...rag.augment import default_knowledge_store
from ...rag.index import retrieve
from ...rag.vectorstore import get_vector_store
from ..auth import require_user

router = APIRouter()


def _caller_scope(user: User) -> Any:
    from mixle_knowledge.kb.store import CallerScope

    return CallerScope(identity=user.id, is_admin=user.is_admin)


class QueryRequest(BaseModel):
    text: str | None = None
    k: int = 8
    modalities: list[str] | None = None
    project_id: str = "default"
    task: str = "chat_rag"
    target_kind: str = "model"
    target_id: str | None = None


class HandoffRequest(BaseModel):
    bundle_id: str
    target_model: str
    question: str


@router.post("/knowledge/query")
async def knowledge_query(body: QueryRequest = Body(...), user: User = Depends(require_user)):
    """Federate the caller's document/conversation retriever into one ranked `KnowledgeBundle`."""
    from mixle_knowledge.kb.adapters import DocumentRAGAdapter, KnowledgeQuery
    from mixle_knowledge.kb.federated import FederatedKB

    def _retrieve_fn(uid: str, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        return retrieve(uid, text, store=get_vector_store(), **kwargs)

    federated = FederatedKB([DocumentRAGAdapter(_retrieve_fn)], default_knowledge_store())
    query = KnowledgeQuery(text=body.text, user_id=user.id, k=body.k, modalities=body.modalities)
    bundle = federated.query(
        query,
        project_id=body.project_id,
        task=body.task,
        target_kind=body.target_kind,
        target_id=body.target_id,
        caller_scope=_caller_scope(user),
    )
    return bundle.model_dump(mode="json")


@router.get("/knowledge/bundles/{bundle_id}")
async def get_bundle(bundle_id: str, revision: int | None = None, user: User = Depends(require_user)):
    """Fetch a bundle (the latest revision, or a specific one), access-filtered for the caller."""
    store = default_knowledge_store()
    try:
        bundle = store.get_bundle(bundle_id, revision=revision, caller_scope=_caller_scope(user))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return bundle.model_dump(mode="json")


@router.post("/knowledge/handoff")
async def knowledge_handoff(request: Request, body: HandoffRequest = Body(...), user: User = Depends(require_user)):
    """Hand ``bundle_id`` off to ``target_model``, requiring a structured `KnowledgeDelta` reply, and apply it."""
    from ...knowledge.handoff import HandoffError, handoff

    registry = request.app.state.registry
    try:
        result = await handoff(
            body.bundle_id,
            target_model=body.target_model,
            question=body.question,
            store=default_knowledge_store(),
            registry=registry,
            caller_scope=_caller_scope(user),
        )
    except HandoffError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:  # StaleDeltaError (a ValueError subclass) + apply_delta's own hash/conflict rejections
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "source_bundle_id": result.source_bundle_id,
        "delta": result.delta.model_dump(mode="json"),
        "output_bundle": result.output_bundle.model_dump(mode="json"),
        "receipt": result.receipt.to_json(),
        "rendered": {
            "resources": result.rendered.resources,
            "preserved_item_ids": result.rendered.preserved_item_ids,
            "omitted_item_ids": result.rendered.omitted_item_ids,
            "bytes_used": result.rendered.bytes_used,
            "tokens_used": result.rendered.tokens_used,
        },
    }
