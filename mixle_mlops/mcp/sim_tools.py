"""IC-11 MCP registration: the unified physics `simulate` tool (workstream P1).

Wraps `mixle_pde.simulation_service.simulate` -- the single forward-simulation surface a frontier model
drives for what-ifs -- as one `mcp.server.Tool` named ``"simulate"``, and registers it onto an existing
`ToolRegistry` through the pre-existing additive `ToolRegistry._add` seam (the same seam
`ToolRegistry._build` itself uses for the platform's own MCP tools). This file never edits
`build_model_tools` (mcp/server.py), `ToolRegistry._build` (gateway/tool_registry.py), or
`mcp/domain_tools.py` -- it is a disjoint registration module, by construction not colliding with E4's or
L4's tool-registration work.

`ToolRegistry` already exposes a generic, mixle-model-sampling ``"simulate"``/``"synthesize"`` pair
(`_build`'s `include_platform` branch). Calling `register_sim_tools` after construction intentionally
*replaces* that generic ``"simulate"`` entry with this IC-11 physics one -- the "unified forward-
simulation `simulate` tool" this task delivers is meant to be *the* `simulate` tool a model reaches for
physics what-ifs. `"synthesize"` is left as-is (still the generic mixle-model alias); only the name
`"simulate"` is unified onto the physics surface.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from mixle_pde.simulation_service import Scenario, ScenarioStep, SimResult, simulate

from .schema_bridge import mcp_tool_to_tooldef
from .server import Tool

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing gateway.tool_registry at runtime
    from ..gateway.tool_registry import ToolRegistry

__all__ = ["build_sim_tools", "register_sim_tools"]

SCENARIO_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenario": {
            "type": "object",
            "description": "IC-11 Scenario: {steps: [{op, inputs_ref, params}], couplings, provenance}",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "ordered forward-simulation steps",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "description": "registered forward operator name"},
                            "inputs_ref": {
                                "type": "string",
                                "description": "content-hashed artifact handle or field id",
                            },
                            "params": {"type": "object", "description": "op-specific parameters"},
                        },
                        "required": ["op", "inputs_ref"],
                    },
                },
                "couplings": {
                    "type": "array",
                    "description": "(from_step, to_step, field_name) edges for a coupled DAG scenario",
                    "items": {"type": "array"},
                },
                "provenance": {"type": "object"},
            },
            "required": ["steps"],
        },
    },
    "required": ["scenario"],
}


def _scenario_step_from_dict(raw: dict[str, Any]) -> ScenarioStep:
    return ScenarioStep(op=raw["op"], inputs_ref=raw["inputs_ref"], params=dict(raw.get("params") or {}))


def _scenario_from_dict(raw: dict[str, Any]) -> Scenario:
    steps = [_scenario_step_from_dict(s) for s in raw.get("steps", [])]
    couplings = [tuple(c) for c in raw.get("couplings", [])]
    provenance = dict(raw.get("provenance") or {})
    return Scenario(steps=steps, couplings=couplings, provenance=provenance)


async def _simulate_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Deserialise `args["scenario"]` -> IC-11 `Scenario`, run it, return the `SimResult` as a plain dict.

    Returned as a dict (not a `json.dumps` string) because this handler is dispatched through
    `ToolRegistry.dispatch`, which returns a handler's result verbatim -- exactly like the platform's own
    dict-returning handlers (`_rag_search`, `_mixle_predict`, `_substrate_retrieve`, ...). When this same
    `Tool` is instead served over the raw MCP JSON-RPC transport (`mcp/server.py`'s `MCPServer._call_tool`
    -> `tools/call`), that layer places the handler's return value into the response's text content block
    and JSON-encodes the whole response regardless of its type, so this dict return value round-trips
    there too.
    """
    result: SimResult = simulate(_scenario_from_dict(args.get("scenario") or {}))
    return dataclasses.asdict(result)


def build_sim_tools() -> dict[str, Tool]:
    """Build the single IC-11 ``"simulate"`` MCP tool."""
    tool = Tool(
        name="simulate",
        description="Run a (possibly coupled) forward-simulation scenario against a registered mixle_pde "
        "physics forward operator (transport, dispersion, wave, flow, em, poroelastic, ...); "
        "returns a content-hashed result handle (IC-11 SimResult).",
        input_schema=SCENARIO_INPUT_SCHEMA,
        handler=_simulate_handler,
    )
    return {tool.name: tool}


def register_sim_tools(tool_reg: "ToolRegistry") -> None:
    """Register the IC-11 ``"simulate"`` tool onto an already-built `ToolRegistry` via its additive
    `_add` seam -- the same mechanism `ToolRegistry._build` uses for the platform's own MCP tools."""
    for tool in build_sim_tools().values():
        tool_reg._add(mcp_tool_to_tooldef(tool), tool.handler)
