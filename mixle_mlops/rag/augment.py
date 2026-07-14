"""The chat composition hook: prepend retrieved context to a chat.

``build_rag_messages(user_id, messages)`` takes the request's messages, retrieves snippets relevant to the latest
user turn (from the user's conversation memory *and* uploaded documents — one retriever), and returns a new
message list with that context prepended. The integrator calls this in the chat pipeline
(``gateway/routes/chat.py``) just before handing the messages to the adapter, gated on a per-request/user flag.

Two composition modes (M1c, IC-13):

  * ``knowledge_mode="legacy_text"`` (the default here) — the original behavior: one flat
    ``format_context_block`` system-message string. Kept as the default so every existing caller of
    ``build_rag_messages`` is unaffected; a full migration to structured-by-default is a later, separate step
    (the gateway opts in per-request via ``extra.knowledge_mode="structured"``, see ``gateway/routes/chat.py``).
  * ``knowledge_mode="structured"`` — federates the same document/conversation retriever through M1b's
    ``DocumentRAGAdapter``/``FederatedKB`` into a real IC-13 ``KnowledgeBundle``, then renders it with M1c's
    capability-aware ``render_bundle`` instead of flattening everything into one string: a compact evidence
    index plus one *separate* message per structured resource, never a single monolithic context blob.
    ``build_structured_rag_messages`` is the richer entry point (it also returns the source bundle id/revision
    so a caller can record provenance); ``build_rag_messages(..., knowledge_mode="structured")`` is the
    drop-in-compatible wrapper that just returns its message list.

Both modes are defensive: if retrieval/federation returns nothing (or errors — e.g. no store yet), the
original messages are returned unchanged so RAG can never break a chat.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .embeddings import Embedder
from .index import _message_text, retrieve
from .vectorstore import VectorStore

CONTEXT_HEADER = (
    "You have access to the following retrieved context from the user's past conversations and uploaded "
    "documents. Use it when relevant to answer; if it does not contain the answer, rely on your own knowledge "
    "and do not fabricate citations.\n\n"
)


def _latest_user_query(messages: Sequence[Any]) -> str:
    """The most recent user-authored text — what we retrieve against."""
    for m in reversed(list(messages)):
        role, text = _message_text(m)
        if role == "user" and text.strip():
            return text
    # fall back to the last message's text
    if messages:
        return _message_text(messages[-1])[1]
    return ""


def format_context_block(snippets: Sequence[dict[str, Any]]) -> str:
    """Render retrieved snippets into a single system-message string."""
    lines = [CONTEXT_HEADER]
    for i, s in enumerate(snippets, 1):
        meta = s.get("meta", {}) or {}
        src = meta.get("filename") or meta.get("conversation_id") or s.get("source_id") or s.get("namespace")
        tag = f"[{i}] ({s.get('namespace', 'context')}"
        if src:
            tag += f": {src}"
        tag += ")"
        lines.append(f"{tag}\n{s.get('text', '').strip()}")
    return "\n\n".join(lines)


def _make_message(role: str, content: str, messages: Sequence[Any], as_dict: bool) -> Any:
    """Build one prepended message matching ``messages``'s own style (a dict when ``as_dict`` and the inputs
    are themselves dicts, otherwise a ``ChatMessage``)."""
    inputs_are_dicts = bool(messages) and isinstance(messages[0], dict)
    if as_dict and inputs_are_dicts:
        return {"role": role, "content": content}
    from ..core.adapters import ChatMessage  # lazy: avoid import cycle at module load

    return ChatMessage(role=role, content=content)


def build_rag_messages(
    user_id: str | None,
    messages: Sequence[Any],
    *,
    k: int = 5,
    namespace: str | None = None,
    min_score: float | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    as_dict: bool = True,
    knowledge_mode: str = "legacy_text",
    **structured_kwargs: Any,
) -> list[Any]:
    """Return ``messages`` with retrieved context prepended (unchanged if nothing relevant).

    ``knowledge_mode="legacy_text"`` (the default) is the original single-string-block behavior. Passing
    ``knowledge_mode="structured"`` composes through :func:`build_structured_rag_messages` instead (M1c,
    IC-13) — ``structured_kwargs`` (``capabilities``, ``target_model``, ``project_id``, ``knowledge_store``,
    ``token_budget``, ``byte_budget``) are forwarded to it and ignored in legacy mode.

    ``messages`` items may be ``ChatMessage`` objects or ``{role, content}`` dicts; the prepended message(s)
    match the input style. Returns the original list's contents untouched on any failure.
    """
    if knowledge_mode == "structured":
        return build_structured_rag_messages(
            user_id, messages, k=k, embedder=embedder, store=store, as_dict=as_dict, **structured_kwargs
        ).messages

    out = list(messages)
    if not user_id:
        return out
    query = _latest_user_query(messages)
    if not query.strip():
        return out
    try:
        snippets = retrieve(
            user_id, query, k=k, namespace=namespace,
            min_score=min_score, embedder=embedder, store=store,
        )
    except Exception:
        return out
    if not snippets:
        return out

    block = format_context_block(snippets)
    return [_make_message("system", block, messages, as_dict), *out]


@dataclass
class StructuredRagResult:
    """The richer structured-RAG composition result: the message list to send, plus the source bundle's
    identity/revision (so a caller can record provenance, e.g. an ``X-Knowledge-Bundle-Id`` header) and the
    underlying `RenderedContext` (M1c) for anything wanting the raw resource descriptors directly."""

    messages: list[Any] = field(default_factory=list)
    bundle_id: str | None = None
    bundle_revision: int | None = None
    rendered: Any = None


_default_knowledge_store_cache: Any = None


def default_knowledge_store() -> Any:
    """Process-wide `StructuredKnowledgeStore`, rooted under this deployment's registry dir. Cached after
    first use, same convention as :func:`mixle_mlops.multimodal.store.get_blob_store`."""
    global _default_knowledge_store_cache
    if _default_knowledge_store_cache is None:
        from mixle_knowledge.kb.store import StructuredKnowledgeStore

        from ..config import get_settings

        _default_knowledge_store_cache = StructuredKnowledgeStore(Path(get_settings().registry_root) / "knowledge_store")
    return _default_knowledge_store_cache


def reset_default_knowledge_store() -> None:
    """Test hook: drop the cached default store so a fresh ``data_dir``/``registry_root`` is picked up."""
    global _default_knowledge_store_cache
    _default_knowledge_store_cache = None


def build_structured_rag_messages(
    user_id: str | None,
    messages: Sequence[Any],
    *,
    k: int = 5,
    capabilities: set[str] | None = None,
    target_model: str | None = None,
    project_id: str = "default",
    knowledge_store: Any = None,
    token_budget: int | None = None,
    byte_budget: int | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    as_dict: bool = True,
) -> StructuredRagResult:
    """Structured-RAG composition (M1c, IC-13): federate the document/conversation retriever into a real
    `KnowledgeBundle` (M1b's `DocumentRAGAdapter`/`FederatedKB`), render it for ``capabilities`` (M1c's
    `render_bundle`), and fold the rendering into ``messages`` as several *separate* messages -- one compact
    evidence-index message plus one message per structured resource -- rather than one monolithic context
    string. Defensive: any failure (or an empty query/bundle) returns ``messages`` unchanged.
    """
    out = list(messages)
    empty = StructuredRagResult(messages=out)
    if not user_id:
        return empty
    query = _latest_user_query(messages)
    if not query.strip():
        return empty

    try:
        from mixle_knowledge.kb.adapters import DocumentRAGAdapter, KnowledgeQuery
        from mixle_knowledge.kb.federated import FederatedKB

        from ..knowledge.render import render_bundle

        kb_store = knowledge_store or default_knowledge_store()

        def _retrieve_fn(uid: str, text: str, **kwargs: Any) -> list[dict[str, Any]]:
            return retrieve(uid, text, embedder=embedder, store=store, **kwargs)

        federated = FederatedKB([DocumentRAGAdapter(_retrieve_fn)], kb_store)
        bundle = federated.query(
            KnowledgeQuery(text=query, user_id=user_id, k=k),
            project_id=project_id, task="chat_rag", target_kind="model", target_id=target_model,
        )
        rendered = render_bundle(
            bundle, capabilities=capabilities or {"chat"}, token_budget=token_budget, byte_budget=byte_budget
        )
    except Exception:
        return empty

    if not bundle.items:
        return StructuredRagResult(messages=out, bundle_id=bundle.id, bundle_revision=bundle.revision, rendered=rendered)

    prefix: list[Any] = [_make_message(m["role"], m["content"], messages, as_dict) for m in rendered.messages]
    for resource in rendered.resources:
        content = f"[resource:{resource['item_id']}] " + json.dumps(resource, sort_keys=True, default=str)
        prefix.append(_make_message("system", content, messages, as_dict))

    return StructuredRagResult(
        messages=[*prefix, *out], bundle_id=bundle.id, bundle_revision=bundle.revision, rendered=rendered,
    )
