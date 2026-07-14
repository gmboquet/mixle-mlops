"""Registration seam for the pde-side physics tools (E4, mlops half).

The physics *logic* — ``run_inversion``, ``query_posterior``, ``gassmann``, ``forward_model`` and their
JSON-schemas (``PHYSICS_TOOL_SCHEMAS``) — lives in ``mixle_pde.tools`` (IC-3, a separate package/PR). This
module only wraps that frozen schema + handler pair into this platform's ``mcp.server.Tool`` objects, exactly
as ``_chat_tool``/``_score_tool`` wrap a hosted model today, so ``build_model_tools`` and ``ToolRegistry`` can
list/dispatch them uniformly alongside the chat/score/MCP tools.

``mixle_pde`` is an optional, lazily-imported dependency: a deployment without the pde package installed still
boots (``build_physics_tools()`` returns ``{}``), and installing it later makes the tools appear on the very
next ``tools/list``/``ToolRegistry`` construction with no restart, the same story ``MCPServer.tools()`` already
tells for newly-registered models.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from .server import INVALID_PARAMS, MCPError, Tool

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "run_inversion": (
        "Fit a subsurface posterior from an observation dataset (gravity/magnetics/dc/ip/mt/csem/aem/seismic) "
        "under a chosen prior/regulariser; returns a content-hashed posterior_ref plus fit diagnostics."
    ),
    "query_posterior": (
        "Read a decision quantity (region_mass, prob_exceed, net_pay, drill_target, marginal, section) off a "
        "saved posterior; always reports whether the answer is prior_dominated (data vs. regulariser driven)."
    ),
    "gassmann": "Run Gassmann rock-physics fluid substitution, propagating input uncertainty through to vp/vs/rho.",
    "forward_model": (
        "Simulate observation data from a model + survey geometry via the requested physics forward operator."
    ),
}


class PhysicsToolsUnavailable(RuntimeError):
    """Raised (and swallowed by ``build_physics_tools``) when ``mixle_pde`` isn't installed."""


def _load_pde_tools() -> Any:
    """Import ``mixle_pde.tools`` (IC-3); raise :class:`PhysicsToolsUnavailable` with an actionable message if
    the pde package isn't installed. Not cached at module scope so a later install is picked up immediately."""
    try:
        from mixle_pde import tools as pde_tools
    except ImportError as exc:  # pragma: no cover - exercised only when mixle_pde is absent
        raise PhysicsToolsUnavailable(
            "the physics tools require the mixle_pde package; install mixle_pde (see the mixle-pde repo) "
            "to enable run_inversion/query_posterior/gassmann/forward_model"
        ) from exc
    return pde_tools


def _validate(args: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate ``args`` against a physics tool's JSON-schema; raises :class:`MCPError` on violation. Uses the
    full ``jsonschema`` package when importable, otherwise a minimal required-field check (same graceful-fallback
    shape as ``gateway.constrained``)."""
    try:
        import jsonschema
    except ImportError:
        missing = [k for k in (schema.get("required") or []) if k not in args]
        if missing:
            raise MCPError(INVALID_PARAMS, f"missing required field(s): {missing}")
        return
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as exc:
        raise MCPError(INVALID_PARAMS, f"schema violation: {exc.message}") from exc


async def _call(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Call a pde handler, off the event loop when it is a plain sync function (an inversion can be slow and
    must never block the server)."""
    if inspect.iscoroutinefunction(fn):
        return await fn(**kwargs)
    return await asyncio.to_thread(fn, **kwargs)


def _make_tool(name: str, pde_tools: Any) -> Tool:
    schema = pde_tools.PHYSICS_TOOL_SCHEMAS[name]
    fn = getattr(pde_tools, name)

    async def handler(args: dict[str, Any]) -> Any:
        _validate(args, schema)
        return await _call(fn, args)

    return Tool(name=name, description=_TOOL_DESCRIPTIONS[name], input_schema=schema, handler=handler)


def build_physics_tools() -> dict[str, Tool]:
    """Build the four IC-3 physics tools, wired to ``mixle_pde.tools``. Returns ``{}`` (no error) if the
    ``mixle_pde`` package isn't importable yet, so a deployment without physics support still boots."""
    try:
        pde_tools = _load_pde_tools()
    except PhysicsToolsUnavailable:
        return {}
    return {name: _make_tool(name, pde_tools) for name in pde_tools.PHYSICS_TOOL_SCHEMAS}


def catalog_entries() -> list[Any]:
    """One IC-10 ``CatalogEntry`` per physics tool, for the router/decomposer (M3) to enumerate uniformly
    alongside economic/climate/external capabilities. Returns ``[]`` if ``mixle.task.catalog`` (IC-10) or
    ``mixle_pde.tools`` aren't importable yet — best-effort, never raises."""
    try:
        from mixle.task.catalog import CatalogEntry
    except ImportError:
        return []
    try:
        pde_tools = _load_pde_tools()
    except PhysicsToolsUnavailable:
        return []
    return [
        CatalogEntry(id=name, schema=schema, owner="physics", verifier="physical")
        for name, schema in pde_tools.PHYSICS_TOOL_SCHEMAS.items()
    ]


def register_physics_catalog(catalog: Any) -> int:
    """Register every physics :class:`catalog_entries` entry into ``catalog`` (an IC-10 ``ToolCatalog``);
    returns how many were added. A no-op returning ``0`` if IC-10 or the pde tools aren't available yet."""
    entries = catalog_entries()
    for entry in entries:
        catalog.register(entry)
    return len(entries)
