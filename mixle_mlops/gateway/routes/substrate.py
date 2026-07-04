"""Serve the knowledge substrate + all-data RAG over HTTP -- deploy the reasoner's retrieval half.

The pysparkplug frontier-ecosystem workplan (v0.6.2) built ``mixle.substrate``: a typed, provenanced
store with cross-kind retrieval, multi-hop lineage chaining, budgeted+compressed context assembly, and
an answer-or-abstain loop. This route deploys the retrieval half of that loop, following the same
honesty contract as ``/v1/tasks`` (decide-or-escalate): the server RETRIEVES the evidence, ASSEMBLES a
budgeted context, and DECIDES whether the evidence is strong enough -- but it does NOT generate the
answer. The caller owns the model (a local student or an LLM): on a confident context it runs its own
answerer over the returned passages, and on ``abstain=true`` it withholds rather than guessing.
Nothing is returned without provenance -- every context carries its citations.

  * ``GET  /v1/substrate/{name}``              -- stats: item count + kinds present.
  * ``POST /v1/substrate/{name}/documents``    -- ``{"docs": [...], "source": ...}`` ingest text.
  * ``POST /v1/substrate/{name}/items``        -- ``{"kind", "text", "payload", "links", ...}`` ingest one.
  * ``POST /v1/substrate/{name}/retrieve``     -- ``{"query", "k", "diversify", "hops"}`` -> cited items.
  * ``POST /v1/substrate/{name}/context``      -- ``{"query", "budget", "hops", "compress", "min_confidence"}``
        -> ``{"abstain", "confidence", "context", "citations"}`` -- the RAG serving contract.

A shard persists under ``{registry_root}/substrate/{name}`` (the same on-disk layout ``Substrate.save``
writes), so it survives restarts and can be rsync'd between the local daemon and the pool.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...accounts.models import User
from ...config import get_settings
from ..auth import require_user

router = APIRouter()

_CACHE: dict[str, tuple[float, Any]] = {}
_LOCK = threading.Lock()


def _root() -> Path:
    return Path(get_settings().registry_root) / "substrate"


def _shard(name: str, *, create: bool = False) -> Any:
    """Load (and cache, keyed on the items file mtime) the named substrate shard."""
    from mixle.substrate import Substrate

    path = _root() / name
    items = path / "items.jsonl"
    if not items.exists() and not create:
        raise HTTPException(status_code=404, detail=f"no substrate shard named {name!r}")
    stamp = items.stat().st_mtime if items.exists() else 0.0
    with _LOCK:
        hit = _CACHE.get(name)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    sub = Substrate(str(path)) if items.exists() else Substrate(str(path))
    with _LOCK:
        _CACHE[name] = (stamp, sub)
    return sub


def _persist(name: str, sub: Any) -> None:
    sub.save(str(_root() / name))
    with _LOCK:
        _CACHE.pop(name, None)  # invalidate: next read reloads from the new mtime


@router.get("/substrate/{name}")
def stats(name: str, user: User = Depends(require_user)) -> dict[str, Any]:
    sub = _shard(name)
    kinds: dict[str, int] = {}
    for it in sub.all():
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    return {"name": name, "n_items": len(sub), "kinds": kinds}


@router.post("/substrate/{name}/documents")
def ingest_docs(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    from mixle.substrate import ingest_documents

    docs = body.get("docs")
    if not isinstance(docs, list) or not docs:
        raise HTTPException(status_code=422, detail='body must be {"docs": [<text or {text,tags,payload}>, ...]}')
    sub = _shard(name, create=True)
    ids = ingest_documents(sub, docs, source=str(body.get("source", "api")), scope=str(body.get("scope", "local")))
    _persist(name, sub)
    return {"ingested": len(ids), "ids": ids}


@router.post("/substrate/{name}/items")
def add_item(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    from mixle.substrate import SubstrateItem

    if "kind" not in body:
        raise HTTPException(status_code=422, detail='body must include "kind" (text/record/image/...)')
    sub = _shard(name, create=True)
    try:
        item = SubstrateItem(
            kind=str(body["kind"]),
            text=str(body.get("text", "")),
            payload=dict(body.get("payload", {})),
            provenance=dict(body.get("provenance", {})),
            tags=list(body.get("tags", [])),
            links=list(body.get("links", [])),
            scope=str(body.get("scope", "local")),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    iid = sub.put(item)
    _persist(name, sub)
    return {"id": iid}


@router.post("/substrate/{name}/retrieve")
def retrieve_route(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    if "query" not in body:
        raise HTTPException(status_code=422, detail='body must be {"query": ...}')
    sub = _shard(name)
    query = str(body["query"])
    k = int(body.get("k", 8))
    scope = body.get("scope")
    hops = int(body.get("hops", 1))
    if hops > 1:
        from mixle.substrate import multihop

        chain = multihop(sub, query, max_hops=hops, scope=scope)
        return {"query": query, "hops": hops, "items": chain.provenance()}
    from mixle.substrate import retrieve

    r = retrieve(sub, query, k=k, diversify=bool(body.get("diversify", True)), scope=scope)
    return {"query": query, "kinds": r.kinds(), "items": r.provenance()}


@router.post("/substrate/{name}/context")
def context_route(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """The RAG serving contract: retrieve + assemble + DECIDE. The caller runs its own answerer over the
    returned context, or respects ``abstain=true`` and withholds. The server never generates the answer."""
    if "query" not in body:
        raise HTTPException(status_code=422, detail='body must be {"query": ...}')
    from mixle.substrate import ContextBudget, answer_from_substrate

    sub = _shard(name)
    budget = body.get("budget", {})
    cb = ContextBudget(
        max_chars=int(budget.get("max_chars", 2000)),
        max_items=int(budget.get("max_items", 20)),
        shape=str(budget.get("shape", "passages")),
    )

    # answer_from_substrate with a NO-OP answerer: it does the retrieve/assemble/decide, and because the
    # answerer is only called when confident, we capture the context+decision and return it for the
    # caller's own model to answer (or the abstain signal to honor). The server owns no answerer.
    captured: dict[str, Any] = {}

    def passthrough(_question: str, context: str) -> str:
        captured["context"] = context
        return "__served__"

    ans = answer_from_substrate(
        sub,
        str(body["query"]),
        passthrough,
        budget=cb,
        hops=int(body.get("hops", 1)),
        min_confidence=float(body.get("min_confidence", 0.1)),
        compress=bool(body.get("compress", True)),
        scope=body.get("scope"),
    )
    return {
        "query": ans.question,
        "abstain": ans.abstained,
        "confidence": round(ans.confidence, 4),
        "context": None if ans.abstained else captured.get("context", ans.context.render()),
        "citations": ans.citations(),
        "note": ans.note,
    }
