"""Vector store: persist embedded chunks per ``(user, namespace)`` and cosine-rank them.

:class:`LocalVectorStore` is SQLModel-backed (the platform DB — SQLite local, Postgres cloud) and does the cosine
ranking in numpy. Vectors are stored JSON-encoded (portable, no pgvector needed); for a user's working-set sizes
(conversation memory + a handful of uploaded docs) an in-process numpy scan is more than fast enough.

:class:`PgVectorStore` records the cloud deployment contract for server-side
ANN search with pgvector or a managed vector DB.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sqlmodel import Session, select

from ..storage.db import get_engine
from .models import VectorItem

_tables_ready = False


def _ensure_tables() -> None:
    """Idempotently create the RAG tables (so the layer works without editing ``storage.db.init_db``)."""
    global _tables_ready
    if _tables_ready:
        return
    from sqlmodel import SQLModel

    from .models import Document  # noqa: F401  (register both RAG tables)

    SQLModel.metadata.create_all(
        get_engine(), tables=[VectorItem.__table__, Document.__table__]
    )
    _tables_ready = True


@dataclass
class Hit:
    """A retrieved item: the ranked chunk plus its cosine score and metadata."""

    id: str
    text: str
    score: float
    namespace: str
    source_id: str | None
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "namespace": self.namespace,
            "source_id": self.source_id,
            "meta": self.meta,
        }


class VectorStore(ABC):
    """Store embedded chunks and cosine-rank them, scoped by user."""

    @abstractmethod
    def add(self, user_id: str, items: Sequence[dict[str, Any]]) -> list[str]:
        """Add items ``{id?, text, vector, meta?, namespace?, source_id?}``; returns the stored ids."""
        ...

    @abstractmethod
    def query(
        self,
        user_id: str,
        vector: np.ndarray,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Return the top-``k`` cosine-nearest items for ``user_id`` matching ``filter`` (e.g. ``namespace``)."""
        ...

    @abstractmethod
    def delete_source(self, user_id: str, source_id: str) -> int:
        """Drop every chunk for a source (re-index / delete a conversation or document). Returns rows removed."""
        ...

    @abstractmethod
    def count(self, user_id: str, filter: dict[str, Any] | None = None) -> int:
        ...


def _as_unit(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class LocalVectorStore(VectorStore):
    """SQLModel-backed store over the platform DB; cosine ranking in numpy."""

    def __init__(self) -> None:
        _ensure_tables()

    def add(self, user_id: str, items: Sequence[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        with Session(get_engine()) as session:
            for it in items:
                vec = _as_unit(it["vector"]).tolist()
                row = VectorItem(
                    user_id=user_id,
                    namespace=it.get("namespace", "default"),
                    source_id=it.get("source_id"),
                    text=it.get("text", ""),
                    vector_json=json.dumps(vec),
                    meta_json=json.dumps(it["meta"]) if it.get("meta") else None,
                )
                if it.get("id"):
                    row.id = it["id"]
                session.add(row)
                ids.append(row.id)
            session.commit()
        return ids

    def query(
        self,
        user_id: str,
        vector: np.ndarray,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Hit]:
        filter = filter or {}
        q = _as_unit(vector)
        if q.size == 0 or not np.any(q):
            return []
        with Session(get_engine()) as session:
            stmt = select(VectorItem).where(VectorItem.user_id == user_id)
            if "namespace" in filter and filter["namespace"] is not None:
                stmt = stmt.where(VectorItem.namespace == filter["namespace"])
            if "source_id" in filter and filter["source_id"] is not None:
                stmt = stmt.where(VectorItem.source_id == filter["source_id"])
            rows = list(session.exec(stmt))
        if not rows:
            return []
        mat = np.asarray([json.loads(r.vector_json) for r in rows], dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != q.shape[0]:
            # dimensionality mismatch (e.g. embedder changed) → score only the matching-dim rows
            keep = [i for i, r in enumerate(rows) if len(json.loads(r.vector_json)) == q.shape[0]]
            if not keep:
                return []
            rows = [rows[i] for i in keep]
            mat = np.asarray([json.loads(r.vector_json) for r in rows], dtype=np.float64)
        scores = mat @ q                       # rows are unit vectors (normalised on add) → cosine
        order = np.argsort(-scores)[: max(k, 0)]
        return [
            Hit(
                id=rows[i].id,
                text=rows[i].text,
                score=float(scores[i]),
                namespace=rows[i].namespace,
                source_id=rows[i].source_id,
                meta=rows[i].meta(),
            )
            for i in order
        ]

    def delete_source(self, user_id: str, source_id: str) -> int:
        with Session(get_engine()) as session:
            rows = list(
                session.exec(
                    select(VectorItem).where(
                        VectorItem.user_id == user_id, VectorItem.source_id == source_id
                    )
                )
            )
            for r in rows:
                session.delete(r)
            session.commit()
            return len(rows)

    def count(self, user_id: str, filter: dict[str, Any] | None = None) -> int:
        filter = filter or {}
        with Session(get_engine()) as session:
            stmt = select(VectorItem).where(VectorItem.user_id == user_id)
            if filter.get("namespace") is not None:
                stmt = stmt.where(VectorItem.namespace == filter["namespace"])
            return len(list(session.exec(stmt)))


class PgVectorStore(VectorStore):
    """Cloud vector-store contract for Postgres + ``pgvector`` or a managed vector DB.

    A cloud deployment should create an ``embedding vector(dim)`` column with an
    IVFFlat/HNSW index and translate :meth:`query` to
    ``ORDER BY embedding <=> :q LIMIT k``. The local-first build uses
    :class:`LocalVectorStore` until a deployment wires server-side ANN search.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "PgVectorStore requires a configured pgvector or managed vector backend."
        )

    def add(self, user_id, items):  # pragma: no cover - abstract backend method
        raise NotImplementedError

    def query(self, user_id, vector, k=5, filter=None):  # pragma: no cover - abstract backend method
        raise NotImplementedError

    def delete_source(self, user_id, source_id):  # pragma: no cover - abstract backend method
        raise NotImplementedError

    def count(self, user_id, filter=None):  # pragma: no cover - abstract backend method
        raise NotImplementedError


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Process-wide vector store. Local fs/SQLite now; ``PgVectorStore`` in a wired cloud deployment."""
    global _store
    if _store is None:
        _store = LocalVectorStore()
    return _store


def reset_vector_store() -> None:
    """Test hook: drop the cached store (and re-create tables on next use)."""
    global _store, _tables_ready
    _store = None
    _tables_ready = False
