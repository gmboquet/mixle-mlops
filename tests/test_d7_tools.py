"""D7 -- expose the platform's rich capabilities (substrate retrieval, calibrated UQ, simulation) as
model-callable tools on `ToolRegistry`, gated by `include_platform`. `substrate_retrieve` must return an
IC-13-valid `KnowledgeBundle` whose graph payload, table schema/cells, media ref and relation links match
the stored substrate items -- not a flattened text blob -- and `uq` must return calibrated intervals."""
import asyncio

import numpy as np
import pytest
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.substrate.core import Substrate, SubstrateItem
from mixle_knowledge.contracts import (
    PROPERTY_GRAPH_SCHEMA,
    SPATIAL_MEDIA_SCHEMA,
    TYPED_TABLE_SCHEMA,
    KnowledgeBundle,
)

from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_mlops.models.mixle_model import MixleAdapter


def _seeded_substrate() -> tuple[Substrate, SubstrateItem, SubstrateItem, SubstrateItem]:
    """A tiny substrate: a property graph, a typed assay table (linked to the graph), and a georeferenced
    raster (linked to the table) -- so relation links, table cells and the media ref all round-trip."""
    substrate = Substrate()
    graph_item = SubstrateItem(
        kind="graph", text="ore body network for site alpha", tags=["site-alpha"],
        payload={
            "nodes": [{"id": "n1", "type": "deposit"}],
            "edges": [{"id": "e1", "source": "n1", "target": "n1", "type": "refines"}],
        },
    )
    substrate.put(graph_item)
    table_item = SubstrateItem(
        kind="record", text="assay table for site alpha", tags=["site-alpha"],
        payload={
            "primary_key": ["sample_id"],
            "columns": [{"name": "sample_id", "type": "string"}, {"name": "cu_pct", "type": "float", "unit": "%"}],
            "rows": [{"sample_id": "A-1", "cu_pct": 1.8}],
        },
        links=[graph_item.id],
    )
    substrate.put(table_item)
    image_item = SubstrateItem(
        kind="image", text="georeferenced raster for site alpha", tags=["site-alpha"],
        payload={
            "ref": "blob://sha256/" + "3" * 64,
            "crs": "EPSG:32611",
            "extent": [500000, 4100000, 501000, 4101000],
            "pixel_to_crs": [1, 0, 500000, 0, -1, 4101000],
        },
        links=[table_item.id],
    )
    substrate.put(image_item)
    return substrate, graph_item, table_item, image_item


def test_tool_registry_exposes_platform_tools():
    reg = ModelRegistry()
    tools = ToolRegistry(reg, user_id=None)
    names = {t.function.name for t in tools.specs()}
    assert {"substrate_retrieve", "uq", "simulate", "synthesize"} <= names


def test_platform_tools_absent_when_disabled():
    reg = ModelRegistry()
    tools = ToolRegistry(reg, user_id=None, include_platform=False)
    names = {t.function.name for t in tools.specs()}
    assert not ({"substrate_retrieve", "uq", "simulate", "synthesize"} & names)


def test_substrate_retrieve_returns_ic13_bundle_without_flattening():
    substrate, graph_item, table_item, image_item = _seeded_substrate()
    tools = ToolRegistry(ModelRegistry(), user_id=None, substrate=substrate)

    raw = asyncio.run(tools.dispatch(
        "substrate_retrieve", {"query": "site alpha", "k": 3, "project_id": "proj-1"}
    ))
    assert "error" not in raw

    bundle = KnowledgeBundle.model_validate(raw)         # IC-13 validity, including cross-field validators
    assert bundle.project_id == "proj-1"
    by_id = {item.id: item for item in bundle.items}
    assert {graph_item.id, table_item.id, image_item.id} <= set(by_id)

    graph = by_id[graph_item.id]
    assert graph.schema_uri == PROPERTY_GRAPH_SCHEMA
    assert graph.payload["nodes"] == graph_item.payload["nodes"]
    assert graph.payload["edges"][0]["id"] == "e1"

    table = by_id[table_item.id]
    assert table.schema_uri == TYPED_TABLE_SCHEMA
    assert table.payload["columns"][1]["type"] == "float"
    assert table.payload["rows"][0]["cu_pct"] == pytest.approx(1.8)
    assert any(r.target_id == graph_item.id for r in table.relations)   # relation link preserved

    image = by_id[image_item.id]
    assert image.schema_uri == SPATIAL_MEDIA_SCHEMA
    assert image.artifact_ref == "blob://sha256/" + "3" * 64             # media ref preserved, not inlined
    assert image.payload["crs"] == "EPSG:32611"
    assert image.payload["extent"] == image_item.payload["extent"]
    assert any(r.target_id == table_item.id for r in image.relations)

    # the rendered text is only a compatibility view -- canonical payloads are what round-trips
    assert "legacy_text" in raw["renderings"]


def test_substrate_retrieve_reuses_supplied_substrate_across_calls():
    substrate, *_ = _seeded_substrate()
    tools = ToolRegistry(ModelRegistry(), user_id=None, substrate=substrate)
    asyncio.run(tools.dispatch("substrate_retrieve", {"query": "alpha", "k": 1}))
    assert tools.substrate is substrate                     # never silently replaced with a fresh empty store


def test_uq_tool_returns_calibrated_intervals():
    reg = ModelRegistry()
    reg.register(MixleAdapter("g", model=GaussianDistribution(5.0, 1.0)))
    tools = ToolRegistry(reg, user_id=None)

    out = asyncio.run(tools.dispatch("uq", {"model": "g", "records": [None, None], "level": 0.5}))
    assert out["interval_level"] == pytest.approx(0.5)
    assert len(out["records"]) == 2
    lo, hi = out["records"][0]["interval"]
    assert lo < hi

    wide = asyncio.run(tools.dispatch("uq", {"model": "g", "records": [None], "level": 0.95}))
    narrow = asyncio.run(tools.dispatch("uq", {"model": "g", "records": [None], "level": 0.5}))
    wide_lo, wide_hi = wide["records"][0]["interval"]
    narrow_lo, narrow_hi = narrow["records"][0]["interval"]
    assert (wide_hi - wide_lo) > (narrow_hi - narrow_lo)    # a higher confidence level is a wider interval


def test_uq_unknown_model_errors_in_band():
    tools = ToolRegistry(ModelRegistry(), user_id=None)
    out = asyncio.run(tools.dispatch("uq", {"model": "nope", "records": []}))
    assert "error" in out


def test_simulate_tool_draws_synthetic_records():
    reg = ModelRegistry()
    reg.register(MixleAdapter("g", model=GaussianDistribution(5.0, 1.0)))
    tools = ToolRegistry(reg, user_id=None)

    out = asyncio.run(tools.dispatch("simulate", {"spec": {"model": "g", "n": 50, "seed": 0}}))
    assert "error" not in out
    records = out["records"]
    assert len(records) == 50
    assert abs(np.mean(records) - 5.0) < 1.0

    synth = asyncio.run(tools.dispatch("synthesize", {"spec": {"model": "g", "n": 10, "seed": 1}}))
    assert len(synth["records"]) == 10
