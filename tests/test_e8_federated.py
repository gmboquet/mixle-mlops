"""E8 -- federated `rag_search`: physics field tiles + document/conversation retrieval, side by side.

A location-anchored `rag_search` call must return two source-native record groups: at least one document
hit carrying its D2 artifact ref/selector through untouched, and at least one field-tile hit dereferenced
from the `SpatialFieldStore`'s `CrossModalStore` (never a bare retrieved index). Document retrieval and
field-tile retrieval are spied on independently so neither store's query drives the other.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import mixle_mlops.rag.index as index_mod
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_pde.reasoning import SpatialFieldStore


def _fake_document_hit():
    """One D2/D3-shaped hit: an artifact ref + selector must survive retrieval untouched."""
    return {
        "id": "chunk-1",
        "text": "assay results for hole DDH-12, 1.8% Cu at 40-46m",
        "score": 0.83,
        "namespace": "document",
        "source_id": "doc-9",
        "meta": {
            "artifact_ref": "blob://sha256/" + "7" * 64,
            "selector": {"sheet": "assays", "row_start": 3, "row_end": 5, "columns": ["hole_id", "cu_pct"]},
        },
    }


@pytest.fixture
def spied_retrieve(monkeypatch):
    """Replace the real D3 `retrieve` with a spy that records the exact call args and returns one fixed hit."""
    calls = []

    def fake_retrieve(user_id, query, k=5, **kwargs):
        calls.append({"user_id": user_id, "query": query, "k": k, **kwargs})
        return [_fake_document_hit()]

    monkeypatch.setattr(index_mod, "retrieve", fake_retrieve)
    return calls


@pytest.fixture
def field_store():
    """A tiny synthetic physics volume: 5 cells on a line, tiled with radius 0.6 (one neighbor either side)."""
    cells = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return SpatialFieldStore(cells, data, tile_radius=0.6)


def test_federated_search_returns_document_and_field_records(spied_retrieve, field_store):
    tools = ToolRegistry(
        ModelRegistry(),
        user_id="tenant-a",
        include_mcp=False,
        include_mixle=False,
        field_store=field_store,
    )
    assert "rag_search" in {t.function.name for t in tools.specs()}

    out = asyncio.run(
        tools.dispatch(
            "rag_search",
            {"query": "Cu assay near the hole", "location": [2.0, 0.0], "k": 3},
        )
    )
    assert "error" not in out
    results = out["results"]

    doc_records = [r for r in results if r["source_kind"] == "document"]
    field_records = [r for r in results if r["source_kind"] == "field_tile"]
    assert doc_records and field_records

    # -- document group: D3 received tenant + text, D2 ref/selector survived verbatim --
    assert spied_retrieve == [{"user_id": "tenant-a", "query": "Cu assay near the hole", "k": 3}]
    doc = doc_records[0]
    assert doc["artifact_ref"] == "blob://sha256/" + "7" * 64
    assert doc["selector"] == {"sheet": "assays", "row_start": 3, "row_end": 5, "columns": ["hole_id", "cu_pct"]}
    assert doc["source_id"] == "doc-9"

    # -- field-tile group: every hit is dereferenced (grid/location metadata), never a bare int --
    assert not any(isinstance(r, int) for r in results)
    for rec in field_records:
        assert isinstance(rec["source_index"], int)
        assert isinstance(rec["selector"], dict)
        assert "centroid" in rec["selector"] and "member_cell_indices" in rec["selector"]
        assert rec["selector"]["n_members"] == len(rec["selector"]["member_cell_indices"])
        assert rec["artifact_ref"] is None
        assert 0.0 < rec["score"] <= 1.0

    # the tile centred on the queried cell itself must be the closest hit
    best = max(field_records, key=lambda r: r["score"])
    assert best["selector"]["centroid"] == [2.0, 0.0]


def test_field_search_omitted_without_location(spied_retrieve, field_store):
    """No `location` -> document-only federation; the field store is never touched."""
    tools = ToolRegistry(
        ModelRegistry(),
        user_id="tenant-a",
        include_mcp=False,
        include_mixle=False,
        field_store=field_store,
    )
    out = asyncio.run(tools.dispatch("rag_search", {"query": "Cu assay", "k": 2}))
    kinds = {r["source_kind"] for r in out["results"]}
    assert kinds == {"document"}


def test_field_store_lazy_factory_is_built_once(spied_retrieve, field_store):
    """A zero-arg factory is only invoked once (built, then reused across calls)."""
    builds = []

    def factory():
        builds.append(1)
        return field_store

    tools = ToolRegistry(
        ModelRegistry(),
        user_id="tenant-a",
        include_mcp=False,
        include_mixle=False,
        field_store=factory,
    )
    for _ in range(3):
        asyncio.run(tools.dispatch("rag_search", {"query": "q", "location": [0.0, 0.0], "k": 2}))
    assert len(builds) == 1
