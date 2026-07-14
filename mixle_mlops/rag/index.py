"""Indexing + retrieval: chunk → embed → add into a user's vector store, and rank snippets for a query.

One retriever serves two sources that share the user's store via distinct namespaces:

  * ``conversation`` — :func:`index_conversation` turns a chat transcript into retrievable memory of past turns.
  * ``document``     — :func:`index_document_chunks` adds the chunks of an uploaded document.

:func:`retrieve` embeds the query once and cosine-ranks across both (or a filtered subset), returning snippet
dicts ready to drop into a context block. Two geoscience-aware extensions layer on top of the plain dense rank:

  * ``filters`` — metadata pre-filtering (bbox / formation / depth / element / date-range) against the chunk
    ``meta`` a parser (e.g. LAS curves, assay tables) attaches per chunk. Filtering happens *before* top-k so a
    filtered-out hit never displaces a matching one.
  * ``hybrid`` — blend the dense cosine score with an Okapi BM25 lexical score (:class:`.embeddings.Bm25Index`)
    over the candidate pool, so an exact rare-term match (a formation name, a sample id) isn't lost in a dense
    embedding's averaging.

Both extensions only touch which hits survive and their order/score; the hit dicts themselves (``id``, ``text``,
``namespace``, ``source_id``, and — importantly — every key under ``meta``, including a structured source's
``artifact_ref``/``selector``/hash/access metadata) pass through unmodified.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..documents.parse import chunk_text
from .embeddings import Bm25Index, Embedder, get_embedder
from .vectorstore import Hit, VectorStore, get_vector_store

NS_CONVERSATION = "conversation"
NS_DOCUMENT = "document"


def _stores(
    embedder: Embedder | None, store: VectorStore | None
) -> tuple[Embedder, VectorStore]:
    return embedder or get_embedder(), store or get_vector_store()


def _message_text(m: Any) -> tuple[str, str]:
    """Return ``(role, text)`` from a ChatMessage-like object or a plain ``{role, content}`` dict."""
    if isinstance(m, Mapping):
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, str):
            return role, content
        # list-of-parts: concatenate text parts
        parts = []
        for p in content if isinstance(content, list) else []:
            if isinstance(p, Mapping) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return role, " ".join(parts)
    role = str(getattr(m, "role", "user"))
    text_fn = getattr(m, "text", None)
    if callable(text_fn):
        return role, text_fn()
    return role, str(getattr(m, "content", ""))


def index_conversation(
    user_id: str,
    conversation_id: str,
    messages: Sequence[Any],
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
    roles: Iterable[str] = ("user", "assistant"),
    replace: bool = True,
) -> list[str]:
    """Index a conversation's messages as retrievable memory.

    Each ``role: text`` turn is chunked (long turns split), embedded, and stored under the ``conversation``
    namespace with ``source_id = conversation_id``. ``replace=True`` first drops any prior chunks for this
    conversation so re-indexing after new turns is idempotent.
    """
    emb, vs = _stores(embedder, store)
    roleset = set(roles)
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        role, text = _message_text(m)
        if role not in roleset:
            continue
        for j, ch in enumerate(chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)):
            labelled = f"{role}: {ch}"
            texts.append(labelled)
            metas.append({"source": "conversation", "conversation_id": conversation_id,
                          "role": role, "message_index": i, "chunk_index": j})
    if replace:
        vs.delete_source(user_id, conversation_id)
    if not texts:
        return []
    vectors = emb.embed(texts)
    items = [
        {"text": texts[i], "vector": vectors[i], "meta": metas[i],
         "namespace": NS_CONVERSATION, "source_id": conversation_id}
        for i in range(len(texts))
    ]
    return vs.add(user_id, items)


def index_document_chunks(
    user_id: str,
    document_id: str,
    chunks: Sequence[str],
    *,
    filename: str = "",
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    extra_meta: Mapping[str, Any] | None = None,
    replace: bool = True,
) -> list[str]:
    """Embed and add pre-chunked document text under the ``document`` namespace (``source_id = document_id``)."""
    emb, vs = _stores(embedder, store)
    chunks = [c for c in chunks if c and c.strip()]
    if replace:
        vs.delete_source(user_id, document_id)
    if not chunks:
        return []
    vectors = emb.embed(chunks)
    base = {"source": "document", "document_id": document_id, "filename": filename}
    if extra_meta:
        base.update(dict(extra_meta))
    items = [
        {"text": chunks[i], "vector": vectors[i],
         "meta": {**base, "chunk_index": i},
         "namespace": NS_DOCUMENT, "source_id": document_id}
        for i in range(len(chunks))
    ]
    return vs.add(user_id, items)


def _in_bbox(meta: Mapping[str, Any], bbox: Sequence[float]) -> bool:
    lon, lat = meta.get("lon"), meta.get("lat")
    if lon is None or lat is None:
        return False
    minx, miny, maxx, maxy = bbox
    return minx <= lon <= maxx and miny <= lat <= maxy


def _in_depth_range(meta: Mapping[str, Any], bounds: Sequence[Any]) -> bool:
    value = meta.get("depth")
    if value is None:
        return False
    lo, hi = bounds
    if isinstance(value, (list, tuple)) and len(value) == 2:
        v_lo, v_hi = value
        return v_lo <= hi and v_hi >= lo  # interval overlap (e.g. a LAS row's depth span)
    return lo <= value <= hi


def _coerce_date(value: Any) -> Any:
    """Best-effort coercion of an ISO date/datetime string (or ``date``/``datetime``) into a comparable value.

    Falls back to the raw value when it isn't ISO-parseable, so a plain lexicographically-sortable date string
    (``YYYY-MM-DD``) still compares correctly without a strict parse.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip())
        except ValueError:
            return value
    return value


def _in_date_range(meta: Mapping[str, Any], bounds: Sequence[Any]) -> bool:
    value = meta.get("date")
    if value is None:
        return False
    lo, hi = bounds
    v, lo_c, hi_c = _coerce_date(value), _coerce_date(lo), _coerce_date(hi)
    try:
        return lo_c <= v <= hi_c
    except TypeError:
        return False  # incomparable types (e.g. one side parsed to datetime, the other didn't) → no match


_FILTER_MATCHERS = {
    "bbox": _in_bbox,
    "formation": lambda meta, value: meta.get("formation") == value,
    "depth": _in_depth_range,
    "element": lambda meta, value: meta.get("element") == value,
    "date": _in_date_range,
}


def _matches_filters(meta: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, value in filters.items():
        matcher = _FILTER_MATCHERS.get(key)
        if matcher is None:
            raise ValueError(
                f"retrieve(): unknown filter key {key!r}; expected one of {sorted(_FILTER_MATCHERS)}"
            )
        if not matcher(meta, value):
            return False
    return True


def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Scale non-negative scores (BM25 is unbounded) into ``[0, 1]`` by dividing by the pool's max."""
    if scores.size == 0:
        return scores
    top = float(scores.max())
    return scores / top if top > 0 else np.zeros_like(scores)


def retrieve(
    user_id: str,
    query: str,
    k: int = 5,
    *,
    namespace: str | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    min_score: float | None = None,
    filters: Mapping[str, Any] | None = None,
    hybrid: bool = False,
) -> list[dict[str, Any]]:
    """Embed ``query`` and return the top-``k`` ranked snippets across the user's store.

    ``namespace`` restricts to ``conversation`` or ``document`` (default: both). ``min_score`` drops weak hits
    (applied last, after ranking).

    ``filters`` (all optional, ANDed together) pre-filter candidates against the chunk ``meta`` a structured
    parser attaches, *before* top-k is taken:

      * ``bbox=(minx, miny, maxx, maxy)`` — point-in-box on ``meta['lon']``/``meta['lat']``.
      * ``formation=<str>`` / ``element=<str>`` — exact match on ``meta['formation']`` / ``meta['element']``.
      * ``depth=(min, max)`` — range check on ``meta['depth']`` (or interval overlap if it's a ``(lo, hi)`` pair).
      * ``date=(start, end)`` — range check on ``meta['date']`` (ISO strings or ``date``/``datetime``).

    A hit missing the metadata a filter needs is dropped (conservative: no lon/lat means it can't be known to be
    in the bbox). Unknown filter keys raise ``ValueError`` rather than silently matching everything.

    ``hybrid=True`` re-ranks the (post-filter) candidate pool by ``0.5 * dense_score + 0.5 * bm25_normalised``,
    where the BM25 half comes from :class:`~.embeddings.Bm25Index` fit over just that pool's text — recovering
    exact lexical matches (a formation name, a sample id) that a dense embedding can under-rank.

    Returns plain dicts (``id, text, score, namespace, source_id, meta``) ready for a context block. Filtering
    and hybrid re-ranking only ever select/reorder/rescore hits — every other field, including anything a
    structured source wrote under ``meta`` (``artifact_ref``, ``selector``, content hash, access policy), passes
    through verbatim; no hit is ever rebuilt.
    """
    emb, vs = _stores(embedder, store)
    if not query or not query.strip():
        return []
    qvec = emb.embed_one(query)
    ns_filter = {"namespace": namespace} if namespace else None
    fetch_k = k
    if filters or hybrid:
        # Filtering and hybrid re-ranking need to see the whole matching candidate pool, not just the dense
        # top-k — otherwise a hit that a plain dense query would rank below k (but that satisfies the metadata
        # filter, or wins on lexical overlap) could never surface.
        fetch_k = max(vs.count(user_id, filter=ns_filter), k)
    hits: list[Hit] = vs.query(user_id, qvec, k=fetch_k, filter=ns_filter)
    if filters:
        hits = [h for h in hits if _matches_filters(h.meta, filters)]
    if hybrid and hits:
        bm25 = Bm25Index()
        bm25.fit([h.text for h in hits])
        bm25_scores = _normalise_scores(bm25.score(query))
        dense_scores = np.asarray([h.score for h in hits], dtype=np.float64)
        combined = 0.5 * dense_scores + 0.5 * bm25_scores
        order = np.argsort(-combined, kind="stable")
        hits = [replace(hits[i], score=float(combined[i])) for i in order]
    hits = hits[:k]
    out = []
    for h in hits:
        if min_score is not None and h.score < min_score:
            continue
        out.append(h.to_dict())
    return out
