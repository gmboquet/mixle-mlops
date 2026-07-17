"""Registration seam for the unified forward-simulation ``simulate`` tool (P1, mlops half).

The simulation *logic* -- ``Scenario``/``ScenarioStep``/``SimResult``, ``simulate``, ``register_forward``, and
the registered pde forwards (transport/dispersion/wave/flow/em/poroelastic/model) -- lives in
``mixle_pde.simulation_service`` (IC-11, a separate package/PR). This module only wraps that frozen dataclass
+ function pair into this platform's ``mcp.server.Tool``, exactly the way ``mcp/physics_tools.py`` (E4) wraps
``mixle_pde.tools`` (IC-3), so a :class:`~mixle_mlops.gateway.tool_registry.ToolRegistry` can list/dispatch a
single ``simulate`` tool -- the digital twin G/H/K/L/N drive for what-ifs.

``mixle_pde.simulation_service`` is an optional, lazily-imported dependency: a deployment without the IC-11
module installed still boots (``build_sim_tools()`` returns ``{}``), and installing/landing it later makes the
tool appear on the very next ``build_sim_tools``/``register_sim_tools`` call, no restart -- the same story
``mcp/physics_tools.py`` and ``MCPServer.tools()`` already tell for newly-registered capabilities.

Note on a pre-existing name: ``ToolRegistry._build`` (``include_platform=True``) already registers a
*different* tool named ``"simulate"`` -- forward-sampling synthetic records from a hosted mixle model's fitted
generative distribution via ``mixle.inference.simulate.simulate`` (``gateway/tool_registry.py`` around line
165). That registration is untouched here (P1's file boundary forbids editing ``tool_registry.py``/
``server.py``). ``register_sim_tools`` is opt-in and purely additive via the pre-existing ``ToolRegistry._add``
(``tool_registry.py:35``): when a caller invokes it against a registry, it replaces whatever handler is
currently bound to the name ``"simulate"`` with the IC-11 physics-scenario tool -- the intended precedence for
a registry a frontier model uses to drive physics what-ifs. Callers who want the platform's generic
model-sampling ``simulate`` tool instead simply do not call ``register_sim_tools`` on that registry.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

from .schema_bridge import mcp_tool_to_tooldef
from .server import INVALID_PARAMS, MCPError, Tool

_SCENARIO_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "description": "wave|flow|em|transport|dispersion|poroelastic|climate_rcm|exposure|habitat|model",
        },
        "inputs_ref": {
            "type": "string",
            "description": "content-hashed artifact handle (IC-2) or a registered field id",
        },
        "params": {"type": "object", "description": "op-specific forward parameters"},
    },
    "required": ["op", "inputs_ref"],
}

SIMULATE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenario": {
            "type": "object",
            "description": "an IC-11 Scenario: one or more forward steps, optionally DAG-coupled",
            "properties": {
                "steps": {"type": "array", "items": _SCENARIO_STEP_SCHEMA},
                "couplings": {
                    "type": "array",
                    "description": "(from_step, to_step, field_name) edges; empty = single-step/no coupling",
                    "items": {"type": "array"},
                },
                "provenance": {"type": "object"},
            },
            "required": ["steps"],
        },
    },
    "required": ["scenario"],
}


class SimToolsUnavailable(RuntimeError):
    """Raised (and swallowed by ``build_sim_tools``) when ``mixle_pde.simulation_service`` isn't installed."""


def _load_simulation_service() -> Any:
    """Import ``mixle_pde.simulation_service`` (IC-11); raise :class:`SimToolsUnavailable` with an actionable
    message if the pde package/module isn't available yet. Not cached at module scope so a later install/land
    is picked up immediately (mirrors ``mcp/physics_tools.py::_load_pde_tools``).

    Deliberately ``importlib.import_module("mixle_pde.simulation_service")`` rather than
    ``from mixle_pde import simulation_service``: the ``from`` form resolves the submodule via
    ``_handle_fromlist``, which skips ``sys.modules`` entirely when the parent package already has a
    ``simulation_service`` attribute (``hasattr(mixle_pde, "simulation_service")``). Once anything in the
    process has imported the real submodule once — including a test's own probe
    ``from mixle_pde import simulation_service`` used only to decide whether to skip — that attribute is set
    for the rest of the process, so a later ``monkeypatch.setitem(sys.modules, "mixle_pde.simulation_service",
    fake)`` is silently ignored and the real module wins instead of the fake. ``importlib.import_module``
    always resolves the fully-qualified name through ``sys.modules`` first, so it honors such a monkeypatch
    regardless of import history/order (same rationale as ``mcp/physics_tools.py::_load_pde_tools``)."""
    try:
        simulation_service = importlib.import_module("mixle_pde.simulation_service")
    except ImportError as exc:  # pragma: no cover - exercised only when mixle_pde lacks IC-11
        raise SimToolsUnavailable(
            "the simulate tool requires mixle_pde.simulation_service (IC-11); install/update mixle_pde "
            "(see the mixle-pde repo, workstream P) to enable the unified forward-simulation `simulate` tool"
        ) from exc
    return simulation_service


def _validate_scenario(raw: Any) -> None:
    """Validate the raw ``scenario`` arg against :data:`SIMULATE_TOOL_SCHEMA`; raises :class:`MCPError` on
    violation. Uses the full ``jsonschema`` package when importable, otherwise a minimal required-field check
    (same graceful-fallback shape as ``mcp/physics_tools.py::_validate``)."""
    if not isinstance(raw, dict):
        raise MCPError(INVALID_PARAMS, "'scenario' (object) is required")
    try:
        import jsonschema
    except ImportError:
        if "steps" not in raw or not isinstance(raw["steps"], list):
            raise MCPError(INVALID_PARAMS, "scenario.steps (array) is required")
        for i, step in enumerate(raw["steps"]):
            step_missing = [k for k in ("op", "inputs_ref") if k not in step]
            if step_missing:
                raise MCPError(INVALID_PARAMS, f"scenario.steps[{i}] missing required field(s): {step_missing}")
        return
    try:
        jsonschema.validate({"scenario": raw}, SIMULATE_TOOL_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise MCPError(INVALID_PARAMS, f"schema violation: {exc.message}") from exc


def _scenario_from_dict(sim: Any, raw: dict[str, Any]) -> Any:
    """Deserialise a plain-dict scenario (as received over the wire/tool-call args) into IC-11
    ``Scenario``/``ScenarioStep`` dataclasses."""
    steps = [
        sim.ScenarioStep(op=step["op"], inputs_ref=step["inputs_ref"], params=dict(step.get("params") or {}))
        for step in (raw.get("steps") or [])
    ]
    couplings = [tuple(c) for c in (raw.get("couplings") or [])]
    return sim.Scenario(steps=steps, couplings=couplings, provenance=dict(raw.get("provenance") or {}))


def build_sim_tools() -> dict[str, Tool]:
    """Build the single IC-11 ``simulate`` tool, wired to ``mixle_pde.simulation_service``. Returns ``{}`` (no
    error) if ``mixle_pde`` doesn't have the IC-11 module importable yet, so a deployment without physics
    forward-simulation support still boots."""
    try:
        sim = _load_simulation_service()
    except SimToolsUnavailable:
        return {}

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        raw_scenario = args.get("scenario")
        _validate_scenario(raw_scenario)
        scenario = _scenario_from_dict(sim, raw_scenario)
        result = sim.simulate(scenario)
        return dataclasses.asdict(result)

    return {
        "simulate": Tool(
            name="simulate",
            description=(
                "Run a (possibly coupled multiphysics) forward scenario -- wave/flow/em/transport/dispersion/"
                "poroelastic/model -- and return a content-hashed result handle. The single surface a frontier "
                "model drives for physics what-ifs (IC-11)."
            ),
            input_schema=SIMULATE_TOOL_SCHEMA,
            handler=handler,
        )
    }


def register_sim_tools(tool_reg: Any) -> None:
    """Register the IC-11 ``simulate`` tool onto an existing ``ToolRegistry`` via its pre-existing additive
    ``_add`` (``tool_registry.py:35``) -- no edit to ``ToolRegistry._build``/``build_model_tools``. A no-op if
    ``mixle_pde.simulation_service`` isn't importable yet (``build_sim_tools`` degrades to ``{}``)."""
    for tool in build_sim_tools().values():
        tool_reg._add(mcp_tool_to_tooldef(tool), tool.handler)


def catalog_entries() -> list[Any]:
    """One IC-10 ``CatalogEntry`` for ``simulate``, so the router/decomposer (M3) enumerates the physics
    forward-simulation capability uniformly alongside the physics/economic/climate/external ones. Returns
    ``[]`` if ``mixle.task.catalog`` (IC-10) or ``mixle_pde.simulation_service`` (IC-11) aren't importable
    yet -- best-effort, never raises."""
    try:
        from mixle.task.catalog import CatalogEntry
    except ImportError:
        return []
    if not build_sim_tools():
        return []
    return [CatalogEntry(id="simulate", schema=SIMULATE_TOOL_SCHEMA, owner="physics", verifier="physical")]


def register_sim_catalog(catalog: Any) -> int:
    """Register :func:`catalog_entries` into ``catalog`` (an IC-10 ``ToolCatalog``); returns how many were
    added. A no-op returning ``0`` if IC-10 or IC-11 aren't available yet."""
    entries = catalog_entries()
    for entry in entries:
        catalog.register(entry)
    return len(entries)
