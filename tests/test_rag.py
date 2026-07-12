"""RAG + embeddings + vector store + document ingestion.

Self-contained: builds the app via ``create_app()``, includes the rag router, signs up for an API key, and uses
the **local-fallback embedder** (deterministic hashing — no embeddings server needed) over a fresh ``data_dir``.

Covers:
  * upload a small text 'document', list it, search it, get a relevant hit;
  * index a synthetic conversation and retrieve it;
  * ``build_rag_messages`` prepends a retrieved-context system block;
  * the embedder fallback, chunking, and document text extraction directly.
"""
from __future__ import annotations

import mixle_mlops.multimodal.store as store_mod
import mixle_mlops.rag.embeddings as emb_mod
import mixle_mlops.rag.vectorstore as vs_mod
import mixle_mlops.storage.db as db
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.config import get_settings
from mixle_mlops.documents.parse import chunk_text, extract_text, parse_and_chunk
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import rag as rag_route
from mixle_mlops.rag.augment import build_rag_messages
from mixle_mlops.rag.embeddings import Embedder
from mixle_mlops.rag.index import index_conversation, retrieve
from mixle_mlops.rag.vectorstore import LocalVectorStore


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    """Fresh data_dir + caches reset, with the embedder forced to its local fallback (no server)."""
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    store_mod.reset_blob_store()
    vs_mod.reset_vector_store()
    # force the deterministic local embedder everywhere (no embeddings server in CI)
    emb_mod.reset_embedder()
    monkeypatch.setattr(emb_mod, "get_embedder", lambda: Embedder(allow_remote=False))
    import mixle_mlops.rag.index as index_mod
    monkeypatch.setattr(index_mod, "get_embedder", lambda: Embedder(allow_remote=False))
    yield tmp_path
    get_settings.cache_clear()
    db._engine = None
    store_mod.reset_blob_store()
    vs_mod.reset_vector_store()
    emb_mod.reset_embedder()


@pytest.fixture
def client(local_env):
    app = create_app()
    app.include_router(rag_route.router, prefix="/v1", tags=["rag"])
    with TestClient(app) as c:
        yield c


def _signup(client) -> tuple[dict, str]:
    body = client.post("/auth/signup", json={"email": "r@t.com", "password": "pw12345"}).json()
    return {"Authorization": f"Bearer {body['api_key']}"}, body["user"]["id"]


# --- unit: embedder fallback ---
def test_local_embedder_is_deterministic_and_normalised():
    e = Embedder(allow_remote=False)
    a = e.embed(["the cat sat on the mat", "completely different content here"])
    b = e.embed(["the cat sat on the mat", "completely different content here"])
    assert a.shape == (2, e.dim)
    assert np.allclose(a, b)                                   # deterministic across calls
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)        # L2-normalised rows
    # similar text scores higher than dissimilar text
    q = e.embed_one("a cat on a mat")
    assert float(a[0] @ q) > float(a[1] @ q)
    assert e.embed([]).shape[0] == 0


# --- unit: chunking + extraction ---
def test_chunking_overlap_and_extraction():
    text = "word " * 500
    chunks = chunk_text(text, chunk_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(len(c) <= 20 * 4 + 4 for c in chunks)
    # txt/md extraction
    assert "hello world" in extract_text(b"hello world", filename="a.txt")
    assert "# Title" in extract_text(b"# Title\nbody", filename="a.md")
    full, ch = parse_and_chunk(b"alpha beta gamma", filename="x.txt", chunk_tokens=4)
    assert full == "alpha beta gamma" and ch


# --- unit: vector store cosine ranking ---
def test_local_vector_store_query(local_env):
    vs = LocalVectorStore()
    e = Embedder(allow_remote=False)
    texts = ["python programming language", "baking sourdough bread", "machine learning models"]
    vecs = e.embed(texts)
    vs.add("user-1", [{"text": texts[i], "vector": vecs[i], "namespace": "document",
                       "source_id": "doc-1"} for i in range(3)])
    hits = vs.query("user-1", e.embed_one("how to code in python"), k=2)
    assert hits and hits[0].text == "python programming language"
    assert vs.count("user-1") == 3
    # filtering by another user returns nothing
    assert vs.query("user-2", e.embed_one("python"), k=2) == []
    # delete_source removes the chunks
    assert vs.delete_source("user-1", "doc-1") == 3
    assert vs.count("user-1") == 0


# --- unit: conversation indexing + retrieve ---
def test_index_conversation_and_retrieve(local_env):
    messages = [
        {"role": "user", "content": "My favorite color is teal and I drive a red truck."},
        {"role": "assistant", "content": "Noted: teal is your favorite color."},
        {"role": "user", "content": "Remind me what we ate yesterday."},
    ]
    ids = index_conversation("user-1", "conv-1", messages, store=LocalVectorStore())
    assert ids
    hits = retrieve("user-1", "what is my favorite color?", k=3, store=LocalVectorStore())
    assert hits
    assert any("teal" in h["text"] for h in hits)
    assert hits[0]["namespace"] == "conversation"


# --- unit: build_rag_messages prepends context ---
def test_build_rag_messages_prepends_context(local_env):
    index_conversation(
        "user-1", "conv-1",
        [{"role": "user", "content": "The launch code is BLUE-HERON-7."}],
        store=LocalVectorStore(),
    )
    msgs = [{"role": "user", "content": "what was the launch code again?"}]
    out = build_rag_messages("user-1", msgs, store=LocalVectorStore())
    assert len(out) == len(msgs) + 1
    assert out[0]["role"] == "system"
    assert "BLUE-HERON-7" in out[0]["content"]
    # original messages preserved after the prepended block
    assert out[1] == msgs[0]
    # no user_id → unchanged
    assert build_rag_messages(None, msgs, store=LocalVectorStore()) == msgs
    # irrelevant-but-empty query short-circuits to unchanged
    assert build_rag_messages("user-1", [{"role": "user", "content": "   "}],
                              store=LocalVectorStore()) == [{"role": "user", "content": "   "}]


# --- route: upload document, list, search ---
def test_document_upload_list_and_search(client):
    headers, _uid = _signup(client)

    # auth required
    assert client.post("/v1/documents",
                       files={"file": ("d.txt", b"hi", "text/plain")}).status_code == 401

    doc_text = (
        "The Apollo guidance computer used core rope memory. "
        "Photosynthesis converts sunlight into chemical energy in plants. "
        "The mitochondria is the powerhouse of the cell."
    ).encode()
    up = client.post("/v1/documents", headers=headers,
                     files={"file": ("bio.txt", doc_text, "text/plain")})
    assert up.status_code == 200, up.text
    doc = up.json()
    assert doc["filename"] == "bio.txt" and doc["n_chunks"] >= 1

    listing = client.get("/v1/documents", headers=headers).json()
    assert any(d["id"] == doc["id"] for d in listing["data"])

    search = client.post("/v1/rag/search", headers=headers,
                         json={"query": "what is the powerhouse of the cell?", "k": 3})
    assert search.status_code == 200, search.text
    hits = search.json()["data"]
    assert hits
    assert any("mitochondria" in h["text"].lower() for h in hits)
    assert hits[0]["namespace"] == "document"


def test_search_is_user_scoped(client):
    headers_a, _ = _signup(client)
    client.post("/v1/documents", headers=headers_a,
                files={"file": ("secret.txt", b"the treasure is buried under the oak tree", "text/plain")})

    other = client.post("/auth/signup", json={"email": "b@t.com", "password": "pw12345"}).json()
    headers_b = {"Authorization": f"Bearer {other['api_key']}"}
    res = client.post("/v1/rag/search", headers=headers_b, json={"query": "treasure", "k": 5}).json()
    assert res["data"] == []                                   # user B sees none of user A's documents


def test_unsupported_document_format_rejected(client):
    headers, _ = _signup(client)
    r = client.post("/v1/documents", headers=headers,
                    files={"file": ("x.bin", b"\x00\x01\x02", "application/octet-stream")})
    assert r.status_code == 400
