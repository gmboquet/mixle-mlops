"""E8 -- federated retrieval: physics + docs.

``ToolRegistry.rag_search`` is the model-callable entry point a hosted model already reaches for
document context. This adds a ``location`` argument so the *same* call can also federate a physics
:class:`mixle_pde.reasoning.SpatialFieldStore` -- a location-anchored request comes back with both
nearby document chunks (still carrying their D2 artifact ref / selector) and nearby field tiles,
each a typed, source-native record. The store's ``retrieve`` returns bare corpus indices; those must
be dereferenced (payload + key) before they ever leave this module -- an integer index is not
evidence a downstream ranker (M1b, IC-13) can use.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_pde.reasoning import SpatialFieldStore


def _build_field_store() -> SpatialFieldStore:
    xs, ys = np.meshgrid(np.arange(3.0), np.arange(3.0))
    cells = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(9)])
    data = np.arange(9, dtype=float)
    return SpatialFieldStore(cells, data, tile_radius=1.5)


@pytest.fixture
def retrieve_spy(monkeypatch):
    """Replace ``mixle_mlops.rag.index.retrieve`` with a spy that records its call args and returns one
    D2-style structured hit (artifact_ref + selector living under ``meta``)."""
    calls: list[dict] = []

    def fake_retrieve(user_id, query, k=5, **kwargs):
        calls.append({"user_id": user_id, "query": query, "k": k, **kwargs})
        return [
            {
                "id": "chunk-1",
                "text": "assay table near the anomaly",
                "score": 0.83,
                "namespace": "document",
                "source_id": "doc-1",
                "meta": {
                    "artifact_ref": "blob://sha256/" + "a" * 64,
                    "selector": {"row": 4, "column": "cu_pct"},
                },
            }
        ]

    import mixle_mlops.rag.index as index_mod

    monkeypatch.setattr(index_mod, "retrieve", fake_retrieve)
    return calls


def test_federated_rag_search_returns_doc_and_field_records(retrieve_spy):
    reg = ModelRegistry()
    tools = ToolRegistry(
        reg,
        user_id="tenant-1",
        include_mcp=False,
        include_mixle=False,
        field_store=_build_field_store(),
    )

    query_text = "what does the ore body look like near here?"
    out = asyncio.run(
        tools.dispatch(
            "rag_search",
            {
                "query": query_text,
                "k": 3,
                "location": [1.0, 1.0, 0.0],
            },
        )
    )
    assert "error" not in out, out
    results = out["results"]

    # doc RAG actually received tenant + text (never swallowed, never rewritten to something else)
    assert retrieve_spy, "retrieve() was never called"
    assert retrieve_spy[0]["user_id"] == "tenant-1"
    assert retrieve_spy[0]["query"] == query_text

    docs = [r for r in results if r["source_kind"] == "document"]
    tiles = [r for r in results if r["source_kind"] == "field_tile"]

    assert len(docs) >= 1
    doc = docs[0]
    assert doc["source_id"] == "doc-1"
    assert doc["artifact_ref"] == "blob://sha256/" + "a" * 64
    assert doc["selector"] == {"row": 4, "column": "cu_pct"}

    assert len(tiles) >= 1
    tile = tiles[0]
    assert isinstance(tile["source_index"], int)  # index kept, but never the *whole* record
    assert "location" in tile and len(tile["location"]) == 3
    assert "grid" in tile and tile["grid"]["cell_indices"]  # dereferenced .payloads[index], not a bare int

    # no returned payload is a bare int -- every record is a typed, source-native dict
    assert all(isinstance(r, dict) for r in results)
    assert all("source_kind" in r and "score" in r and "provenance" in r for r in results)


def test_rag_search_without_location_is_doc_only(retrieve_spy):
    reg = ModelRegistry()
    tools = ToolRegistry(
        reg, user_id="tenant-1", include_mcp=False, include_mixle=False, field_store=_build_field_store()
    )
    out = asyncio.run(tools.dispatch("rag_search", {"query": "no location here", "k": 2}))
    assert all(r["source_kind"] == "document" for r in out["results"])


def test_rag_search_location_without_field_store_degrades_to_doc_only(retrieve_spy):
    reg = ModelRegistry()
    tools = ToolRegistry(reg, user_id="tenant-1", include_mcp=False, include_mixle=False)
    out = asyncio.run(
        tools.dispatch(
            "rag_search",
            {
                "query": "location given but no store configured",
                "k": 2,
                "location": [0.0, 0.0, 0.0],
            },
        )
    )
    assert all(r["source_kind"] == "document" for r in out["results"])


def test_bbox_filter_degrades_gracefully_against_pre_filter_retrieve(monkeypatch):
    """``retrieve()`` on this branch predates the geoscience ``filters=``/``hybrid=`` kwargs (D3); a bbox
    request must not crash -- it degrades to a plain tenant+text call rather than raising TypeError."""

    def old_style_retrieve(user_id, query, k=5, *, namespace=None, embedder=None, store=None, min_score=None):
        return [{"id": "c", "text": "t", "score": 1.0, "namespace": "document", "source_id": "d", "meta": {}}]

    import mixle_mlops.rag.index as index_mod

    monkeypatch.setattr(index_mod, "retrieve", old_style_retrieve)
    reg = ModelRegistry()
    tools = ToolRegistry(reg, user_id="tenant-1", include_mcp=False, include_mixle=False)
    out = asyncio.run(
        tools.dispatch(
            "rag_search",
            {
                "query": "q",
                "k": 1,
                "bbox": [-1.0, -1.0, 1.0, 1.0],
            },
        )
    )
    assert "error" not in out, out
    assert out["results"][0]["source_kind"] == "document"
