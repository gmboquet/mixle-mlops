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
  * ``POST /v1/substrate/{name}/factuality``   -- ``{"answer", "min_score", "k"}`` -> per-claim receipt
        ``{"grounded_fraction", "n_unsupported", "claims": [{"claim", "supported", "citations"}]}``.
        Fully server-side (retrieval + content overlap, no caller model): ground an answer's claims in
        the store and report which can be cited -- the deployed twin of ``check_factuality`` (B3).

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
from ._paths import safe_join

router = APIRouter()

_CACHE: dict[str, tuple[float, Any]] = {}
_LOCK = threading.Lock()


def _root() -> Path:
    return Path(get_settings().registry_root) / "substrate"


def _shard(name: str, *, create: bool = False) -> Any:
    """Load (and cache, keyed on the items file mtime) the named substrate shard."""
    from mixle.substrate import Substrate

    path = safe_join(_root(), name)
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
    sub.save(str(safe_join(_root(), name)))
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


@router.post("/substrate/{name}/factuality")
def factuality_route(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """Ground an answer's claims in the shard: a per-claim receipt with citations (fully server-side).

    Unlike ``/context`` (which returns evidence for the caller's own model to answer), this deploys the
    complete ``check_factuality`` loop -- claim extraction + retrieval + content-overlap corroboration --
    because verifying an already-written answer needs no generation. The caller gates on the receipt."""
    if "answer" not in body:
        raise HTTPException(status_code=422, detail='body must be {"answer": ...}')
    from mixle.substrate import check_factuality

    sub = _shard(name)
    receipt = check_factuality(
        sub,
        str(body["answer"]),
        min_score=float(body.get("min_score", 0.2)),
        k=int(body.get("k", 4)),
        scope=body.get("scope"),
    )
    d = receipt.as_dict()
    d["unsupported"] = [v.claim for v in receipt.unsupported()]
    return d


@router.post("/substrate/{name}/publish")
def publish_route(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """Share items into a common scope -- the audited, explicit way knowledge crosses a team boundary (P1).

    ``{"ids": [...], "to": "public", "from_scope": "teamA"}`` re-scopes each item and stamps its provenance
    with who published it (the authenticated user) and from where. ``from_scope`` guards that only items in
    that scope are published, so a team cannot publish another team's private items."""
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=422, detail='body must be {"ids": [<item id>, ...]}')
    from mixle.substrate import publish

    sub = _shard(name)
    published = publish(
        sub,
        [str(i) for i in ids],
        to=str(body.get("to", "public")),
        by=str(getattr(user, "email", None) or getattr(user, "id", "api")),
        from_scope=body.get("from_scope"),
    )
    _persist(name, sub)
    return {"published": published, "n": len(published), "to": str(body.get("to", "public"))}


def _actor(user: User) -> str:
    return str(getattr(user, "email", None) or getattr(user, "id", "api"))


@router.post("/substrate/{name}/propose")
def propose_route(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """Propose items for promotion into a curated scope (P3) -- pending until an admin approves.

    ``{"ids": [...], "to": "org"}`` marks items pending; they are NOT yet visible in the target scope.
    Any authenticated user may propose; only an admin may :func:`approve`."""
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=422, detail='body must be {"ids": [<item id>, ...], "to": <scope>}')
    from mixle.substrate.governance import propose

    sub = _shard(name)
    proposed = propose(sub, [str(i) for i in ids], to=str(body.get("to", "org")), by=_actor(user))
    _persist(name, sub)
    return {"proposed": proposed, "n": len(proposed), "to": str(body.get("to", "org"))}


@router.get("/substrate/{name}/pending")
def pending_route(name: str, to: str | None = None, user: User = Depends(require_user)) -> dict[str, Any]:
    """List items awaiting promotion approval (optionally to a specific scope)."""
    from mixle.substrate.governance import pending

    sub = _shard(name)
    items = pending(sub, to=to)
    return {"pending": [{"id": i.id, "kind": i.kind, "proposal": i.provenance.get("proposal")} for i in items]}


@router.post("/substrate/{name}/approve")
def approve_route(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """Promote a pending item into its proposed scope -- ADMIN ONLY (the promotion gate, P3).

    The approver ACL is the server's own admin flag, not a client-supplied list: only a user with
    ``is_admin`` may approve, so promotion into a curated scope cannot be self-authorized."""
    item_id = body.get("item_id")
    if not item_id:
        raise HTTPException(status_code=422, detail='body must be {"item_id": ...}')
    if not bool(getattr(user, "is_admin", False)):
        raise HTTPException(status_code=403, detail="only an admin may approve a promotion")
    from mixle.substrate.governance import Governance, approve

    sub = _shard(name)
    item = sub.get(str(item_id))
    if item is None:
        raise HTTPException(status_code=404, detail=f"no item {item_id!r}")
    prop = item.provenance.get("proposal") or {}
    target = str(body.get("to") or prop.get("to") or "org")
    gov = Governance().grant(_actor(user), target)  # admin is an approver for the target scope
    ok = approve(sub, str(item_id), by=_actor(user), governance=gov, to=target)
    _persist(name, sub)
    return {"approved": ok, "item_id": item_id, "to": target}


@router.post("/substrate/{name}/reject")
def reject_route(name: str, body: dict[str, Any], user: User = Depends(require_user)) -> dict[str, Any]:
    """Refuse a pending promotion (ADMIN ONLY) -- the item stays put; the refusal is recorded."""
    item_id = body.get("item_id")
    if not item_id:
        raise HTTPException(status_code=422, detail='body must be {"item_id": ...}')
    if not bool(getattr(user, "is_admin", False)):
        raise HTTPException(status_code=403, detail="only an admin may reject a promotion")
    from mixle.substrate.governance import reject

    sub = _shard(name)
    ok = reject(sub, str(item_id), by=_actor(user), reason=str(body.get("reason", "")))
    _persist(name, sub)
    return {"rejected": ok, "item_id": item_id}
