"""M1c DoD: ordinary structured RAG sends mixed resources, not one monolithic context string.

Self-contained (mirrors ``tests/test_rag.py``'s ``local_env`` fixture): a fresh ``data_dir``, the
deterministic local-fallback embedder (no embeddings server needed), and a dedicated on-disk
`StructuredKnowledgeStore` per test (passed explicitly -- no reliance on the process-wide default so
tests never touch a shared location).
"""

from __future__ import annotations

import mixle_mlops.multimodal.store as store_mod
import mixle_mlops.rag.embeddings as emb_mod
import mixle_mlops.rag.vectorstore as vs_mod
import mixle_mlops.storage.db as db
import pytest
from mixle_knowledge.kb.store import StructuredKnowledgeStore

from mixle_mlops.config import get_settings
from mixle_mlops.rag.augment import (
    CONTEXT_HEADER,
    build_rag_messages,
    build_structured_rag_messages,
)
from mixle_mlops.rag.embeddings import Embedder
from mixle_mlops.rag.index import index_conversation
from mixle_mlops.rag.vectorstore import LocalVectorStore


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    """Fresh data_dir + caches reset, with the embedder forced to its local fallback (no server)."""
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    store_mod.reset_blob_store()
    vs_mod.reset_vector_store()
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
def knowledge_store(tmp_path):
    return StructuredKnowledgeStore(tmp_path / "kb")


def _seed_conversation(vector_store):
    index_conversation(
        "user-1",
        "conv-1",
        [{"role": "user", "content": "The launch code is BLUE-HERON-7."}],
        store=vector_store,
    )


def test_structured_mode_sends_mixed_resources_not_one_monolithic_string(local_env, knowledge_store):
    vector_store = LocalVectorStore()
    _seed_conversation(vector_store)
    msgs = [{"role": "user", "content": "what was the launch code again?"}]

    result = build_structured_rag_messages(
        "user-1",
        msgs,
        store=vector_store,
        knowledge_store=knowledge_store,
        capabilities={"chat"},
    )

    assert result.bundle_id is not None
    assert result.bundle_revision == 1

    prepended = result.messages[: len(result.messages) - len(msgs)]
    # more than one message got prepended -- an index message PLUS at least one separate resource message,
    # never a single monolithic context blob the way the legacy path renders.
    assert len(prepended) >= 2
    assert result.messages[-len(msgs) :] == msgs  # original conversation untouched, still last

    # none of the mixed messages is the legacy flat-string block
    assert all(CONTEXT_HEADER not in m["content"] for m in prepended)

    # the retrieved evidence survives as its own separate, JSON-shaped, hash-addressed resource message --
    # never flattened into the plain-prose index/content messages the way the legacy path renders everything
    resource_messages = [m for m in prepended if m["content"].startswith("[resource:")]
    prose_messages = [m for m in prepended if not m["content"].startswith("[resource:")]
    assert resource_messages and prose_messages
    assert any("BLUE-HERON-7" in m["content"] for m in resource_messages)
    assert all('"content_hash"' in m["content"] for m in resource_messages)

    assert len(result.rendered.resources) >= 1
    resource = result.rendered.resources[0]
    assert resource["modality"] == "text"
    assert resource["content_hash"]
    assert resource["payload"]["text"]


def test_structured_resource_payload_matches_declared_content_hash(local_env, knowledge_store):
    from mixle_knowledge.kb.store import canonical_hash

    vector_store = LocalVectorStore()
    _seed_conversation(vector_store)
    msgs = [{"role": "user", "content": "what was the launch code again?"}]

    result = build_structured_rag_messages(
        "user-1",
        msgs,
        store=vector_store,
        knowledge_store=knowledge_store,
        capabilities={"chat"},
    )
    resource = result.rendered.resources[0]
    # the store independently verified this on write; re-deriving it from the resource descriptor's own
    # schema_uri/payload/artifact_ref (plus the real item's metadata, which the descriptor doesn't carry)
    # proves the descriptor still holds the exact canonical fields the hash covers, not a lossy summary.
    stored_item = knowledge_store.get_item(resource["item_id"])
    recomputed = canonical_hash(
        schema_uri=resource["schema_uri"],
        schema_version=resource["schema_version"],
        payload=resource["payload"],
        artifact_ref=resource["artifact_ref"],
        metadata=stored_item.metadata,
    )
    assert recomputed == stored_item.content_hash == resource["content_hash"]


def test_build_rag_messages_structured_wrapper_matches_richer_call(local_env, knowledge_store):
    """``build_rag_messages(..., knowledge_mode="structured")`` is a drop-in wrapper: same shape of output as
    calling :func:`build_structured_rag_messages` directly (each call federates its own fresh bundle, so the
    generated bundle id differs -- compare structure, not byte-for-byte message text)."""
    vector_store = LocalVectorStore()
    _seed_conversation(vector_store)
    msgs = [{"role": "user", "content": "what was the launch code again?"}]

    via_wrapper = build_rag_messages(
        "user-1",
        msgs,
        store=vector_store,
        knowledge_mode="structured",
        knowledge_store=knowledge_store,
    )
    richer = build_structured_rag_messages("user-1", msgs, store=vector_store, knowledge_store=knowledge_store)
    assert len(via_wrapper) == len(richer.messages)
    assert via_wrapper[-len(msgs) :] == msgs
    assert any("BLUE-HERON-7" in m["content"] for m in via_wrapper[: len(via_wrapper) - len(msgs)])


def test_legacy_mode_is_still_the_default_and_stays_one_monolithic_block(local_env, knowledge_store):
    """Regression guard: the default (``knowledge_mode`` unset) is completely unaffected by M1c."""
    vector_store = LocalVectorStore()
    _seed_conversation(vector_store)
    msgs = [{"role": "user", "content": "what was the launch code again?"}]

    out = build_rag_messages("user-1", msgs, store=vector_store)
    assert len(out) == len(msgs) + 1
    assert out[0]["role"] == "system"
    assert "BLUE-HERON-7" in out[0]["content"]
    assert CONTEXT_HEADER in out[0]["content"]


def test_structured_mode_is_defensive_on_no_user_or_empty_query(local_env, knowledge_store):
    vector_store = LocalVectorStore()
    msgs = [{"role": "user", "content": "hello"}]
    assert (
        build_structured_rag_messages(None, msgs, store=vector_store, knowledge_store=knowledge_store).messages
        == msgs
    )
    empty_query_msgs = [{"role": "user", "content": "   "}]
    result = build_structured_rag_messages(
        "user-1",
        empty_query_msgs,
        store=vector_store,
        knowledge_store=knowledge_store,
    )
    assert result.messages == empty_query_msgs
    assert result.bundle_id is None


def test_structured_mode_is_defensive_on_no_matching_evidence(local_env, knowledge_store):
    vector_store = LocalVectorStore()  # nothing indexed at all
    msgs = [{"role": "user", "content": "what was the launch code again?"}]
    result = build_structured_rag_messages(
        "user-1",
        msgs,
        store=vector_store,
        knowledge_store=knowledge_store,
    )
    assert result.messages == msgs
