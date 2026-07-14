"""L4 DoD -- external/domain models (climate, physics, ...) register through the identical MCP/tool-registry
seam physics tools use, every call comes back as a content-addressed, hydratable knowledge item, and the
item's canonical data never depends on the compact JSON ``preview`` a caller may discard.
"""

from __future__ import annotations

import asyncio
import json
import string

from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_mlops.mcp.domain_tools import InMemoryKnowledgeStore, register_domain_tools
from mixle_mlops.models.domain_adapter import (
    ClimateProjectionStub,
    DomainModelAdapter,
    ModelManifest,
    ProvenancedResult,
)


class _RockPhysicsEmulatorStub(DomainModelAdapter):
    """A second, disjoint domain connector standing in for a "physics"-flavored external model (e.g. a
    hosted Gassmann rock-physics emulator). Proves an unrelated `DomainModelAdapter` registers, dispatches
    and hydrates through the exact same surface as the climate stub (Non-goals: "real adapters register
    the same way") -- this class is test-local, not part of the L4 public surface."""

    manifest = ModelManifest(
        name="rock-physics-emulator-stub",
        io_schema={
            "input": {
                "type": "object",
                "properties": {"porosity": {"type": "number"}, "sw": {"type": "number"}},
                "required": ["porosity", "sw"],
            },
            "output": {"type": "object", "properties": {"vp": {"type": "number"}}},
        },
        cost=0.02,
        reliability=0.9,
    )
    version = "synthetic-physics-v1"

    async def call(self, inputs: dict) -> ProvenancedResult:
        porosity = float(inputs["porosity"])
        sw = float(inputs["sw"])
        vp = 4500.0 - 3000.0 * porosity + 200.0 * sw  # deterministic synthetic rock-physics stand-in
        return self._provenanced({"vp": round(vp, 3)}, inputs, uncertainty={"std": 50.0})


def _is_hex64(value: str) -> bool:
    return len(value) == 64 and all(c in string.hexdigits for c in value)


def test_climate_stub_and_physics_one_turn():
    store = InMemoryKnowledgeStore()
    climate = ClimateProjectionStub()
    physics = _RockPhysicsEmulatorStub()

    tool_reg = ToolRegistry(
        ModelRegistry(),
        include_mcp=False,
        include_rag=False,
        include_mixle=False,
        include_platform=False,
        model_id="turn-model-x",
    )
    register_domain_tools(tool_reg, [climate, physics], knowledge_store=store)

    climate_args = {"lat": 34.0, "lon": -118.0, "year": 2060}
    physics_args = {"porosity": 0.22, "sw": 0.6}

    climate_name = f"domain__{climate.name}"
    physics_name = f"domain__{physics.name}"
    assert tool_reg.has(climate_name) and tool_reg.has(physics_name)

    climate_raw = asyncio.run(tool_reg.dispatch(climate_name, climate_args))
    physics_raw = asyncio.run(tool_reg.dispatch(physics_name, physics_args))
    climate_resp = json.loads(climate_raw)
    physics_resp = json.loads(physics_raw)

    # both calls return item ids + 64-hex content hashes
    for resp in (climate_resp, physics_resp):
        assert resp["knowledge_item_id"]
        assert _is_hex64(resp["content_hash"])
        assert _is_hex64(resp["source_result_hash"])
        assert resp["schema_uri"]
        assert resp["preview"]

    # ... attributed in the trace: one TraceStep per call, stamped with the calling model
    assert len(tool_reg.trace_steps) == 2
    assert {step.tool for step in tool_reg.trace_steps} == {climate_name, physics_name}
    for step in tool_reg.trace_steps:
        assert step.args["_model"] == "turn-model-x"
        recorded = json.loads(step.result)
        assert _is_hex64(recorded["content_hash"])

    # hydrating each item reproduces the typed value/uncertainty
    climate_item = store.get(climate_resp["knowledge_item_id"])
    physics_item = store.get(physics_resp["knowledge_item_id"])
    expected_climate = asyncio.run(ClimateProjectionStub().call(climate_args))
    expected_physics = asyncio.run(_RockPhysicsEmulatorStub().call(physics_args))
    assert climate_item["payload"] == expected_climate.value
    assert climate_item["uncertainty"] == expected_climate.uncertainty
    assert physics_item["payload"] == expected_physics.value
    assert physics_item["uncertainty"] == expected_physics.uncertainty

    # corrupting/removing the JSON preview does not alter canonical data
    corrupted = dict(climate_resp)
    del corrupted["preview"]
    corrupted["content_hash"] = "0" * 64
    still_there = store.get(climate_resp["knowledge_item_id"])
    assert still_there["payload"] == expected_climate.value
    assert still_there["content_hash"] == climate_resp["content_hash"]
    assert still_there["content_hash"] != corrupted["content_hash"]
