"""RAG / document routes: upload a document (parse → chunk → index), list documents, and search the retriever.

  * ``POST /v1/documents``       — multipart upload → :class:`BlobStore` → parse into located chunks (page/row +
    artifact ref/selector) → embed → index into the user's vector store (``document`` namespace). Returns the
    document record.
  * ``GET  /v1/documents``       — list the caller's ingested documents.
  * ``POST /v1/rag/search``      — retrieve ranked snippets for a query across conversation memory + documents.

All routes require an authenticated user (``Depends(require_user)``); the vector store and documents are scoped to
``user.id``. Wiring (integrator): ``app.include_router(rag.router, prefix="/v1", tags=["rag"])`` in
``gateway/app.py``.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from ...accounts.models import User
from ...documents.parse import DocumentParseError, parse_and_chunk_located
from ...multimodal.store import get_blob_store
from ...rag.index import index_document_chunks, retrieve
from ...rag.models import Document
from ...rag.vectorstore import _ensure_tables, get_vector_store
from ...storage.db import get_engine
from ..auth import require_user

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    namespace: str | None = None        # "conversation" | "document" | None (both)
    min_score: float | None = None


@router.post("/documents")
async def upload_document(
    file: UploadFile,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
    user: User = Depends(require_user),
):
    """Upload a document, store its bytes, parse + chunk + index its text for retrieval.

    Parsing goes through :func:`parse_and_chunk_located` so each chunk carries page/row/col location plus a
    content-addressed ``artifact_ref``/``selector`` back to canonical structure (a PDF page, an XLSX row range,
    an LAS curve set) -- that per-chunk location is threaded into :func:`index_document_chunks` via ``extra_meta``
    one chunk at a time (its own ``extra_meta`` is a single dict shared by a whole batch, so a batch of one is
    how per-chunk metadata gets in without touching that signature).
    """
    data = await file.read()
    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    try:
        located = parse_and_chunk_located(
            data, filename=filename, content_type=content_type, tokens=chunk_tokens, overlap=overlap_tokens
        )
    except DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not located:
        raise HTTPException(status_code=400, detail="no extractable text in document")

    record = get_blob_store().put(data, filename=filename, content_type=content_type)
    _ensure_tables()
    doc = Document(
        user_id=user.id,
        filename=filename,
        content_type=content_type,
        blob_id=record.id,
        n_chunks=len(located),
        n_chars=sum(len(c.text) for c in located),
    )
    with Session(get_engine()) as session:
        session.add(doc)
        session.commit()
        session.refresh(doc)
        doc_dict = doc.to_dict()

    vector_store = get_vector_store()
    for i, chunk in enumerate(located):
        extra_meta = {k: v for k, v in asdict(chunk).items() if k != "text"}
        extra_meta["location_index"] = i        # global order -- each call below is its own batch of one
        index_document_chunks(
            user.id, doc_dict["id"], [chunk.text], filename=filename,
            extra_meta=extra_meta, store=vector_store, replace=(i == 0),
        )
    return doc_dict


@router.get("/documents")
async def list_documents(user: User = Depends(require_user)):
    """List the caller's ingested documents (most recent first)."""
    _ensure_tables()
    with Session(get_engine()) as session:
        rows = list(
            session.exec(select(Document).where(Document.user_id == user.id))
        )
    rows.sort(key=lambda d: d.created_at or 0, reverse=True)
    return {"object": "list", "data": [d.to_dict() for d in rows]}


@router.post("/rag/search")
async def rag_search(body: SearchRequest = Body(...), user: User = Depends(require_user)):
    """Retrieve ranked context snippets for a query across conversation memory + uploaded documents."""
    hits = retrieve(
        user.id, body.query, k=body.k, namespace=body.namespace,
        min_score=body.min_score, store=get_vector_store(),
    )
    return {"object": "list", "query": body.query, "data": hits}
