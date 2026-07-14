"""E4 (mlops half) — the physics tools (IC-3) are wired into `build_model_tools` and `ToolRegistry`.

The physics *logic* (``run_inversion``/``query_posterior``/``gassmann``/``forward_model`` and their frozen
JSON-schemas) lives in ``mixle_pde.tools`` (IC-3), a sibling package landing under its own PR. To exercise the
mlops-side wiring end-to-end without depending on that PR's landing order, this suite installs a minimal
in-memory stand-in for ``mixle_pde.tools`` that honors the IC-3 signatures/return-shapes — the same technique
as mocking any not-yet-deployed collaborator service. Once the real ``mixle_pde.tools`` lands, this wiring picks
it up unchanged (`_load_pde_tools` does a plain ``from mixle_pde import tools``).
"""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_mlops.mcp.server import build_model_tools
from mixle_mlops.models import EchoAdapter

PHYSICS_TOOL_SCHEMAS = {
    "run_inversion": {
        "type": "object",
        "properties": {
            "dataset_ref": {"type": "string"},
            "modality": {
                "type": "string",
                "enum": ["gravity", "magnetics", "dc", "ip", "mt", "csem", "aem", "seismic"],
            },
            "prior": {"type": "string", "enum": ["smooth", "blocky", "compact", "anisotropic"]},
            "config": {"type": "object"},
        },
        "required": ["dataset_ref", "modality", "prior"],
    },
    "query_posterior": {
        "type": "object",
        "properties": {
            "posterior_ref": {"type": "string"},
            "query": {"type": "string",
                      "enum": ["region_mass", "prob_exceed", "net_pay", "drill_target", "marginal", "section"]},
            "params": {"type": "object"},
        },
        "required": ["posterior_ref", "query"],
    },
    "gassmann": {
        "type": "object",
        "properties": {"inputs": {"type": "object"}},
        "required": ["inputs"],
    },
    "forward_model": {
        "type": "object",
        "properties": {
            "modality": {"type": "string"},
            "model_ref": {"type": "string"},
            "geometry_ref": {"type": "string"},
        },
        "required": ["modality", "model_ref", "geometry_ref"],
    },
}


def _install_fake_pde_tools(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a fake ``mixle_pde.tools`` module satisfying the IC-3 contract, and return the in-memory
    posterior store it uses so a test can inspect it."""
    posteriors: dict[str, dict[str, Any]] = {}
    fake = types.ModuleType("mixle_pde.tools")
    fake.PHYSICS_TOOL_SCHEMAS = PHYSICS_TOOL_SCHEMAS

    def run_inversion(dataset_ref: str, modality: str, prior: str, config: dict | None = None) -> dict:
        ref = f"posterior:{dataset_ref}:{modality}:{prior}"
        posteriors[ref] = {"modality": modality, "prior": prior, "dataset_ref": dataset_ref}
        return {"posterior_ref": ref, "diagnostics": {"modality": modality, "prior": prior, "converged": True}}

    def query_posterior(posterior_ref: str, query: str, params: dict | None = None) -> dict:
        if posterior_ref not in posteriors:
            raise KeyError(f"no such posterior: {posterior_ref}")
        # a smooth prior is data-starved for this toy fixture; anything else is well-constrained
        prior_dominated = posteriors[posterior_ref]["prior"] == "smooth"
        return {"value": 42.0, "prior_dominated": prior_dominated}

    def gassmann(inputs: dict) -> dict:
        return {"vp": 3000.0, "vs": 1500.0, "rho": 2200.0, "uncertainty": {"vp": 50.0}}

    def forward_model(modality: str, model_ref: str, geometry_ref: str) -> dict:
        return {"data_ref": f"data:{modality}:{model_ref}:{geometry_ref}"}

    fake.run_inversion = run_inversion
    fake.query_posterior = query_posterior
    fake.gassmann = gassmann
    fake.forward_model = forward_model

    monkeypatch.setitem(sys.modules, "mixle_pde.tools", fake)
    return posteriors


def _registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    return reg


def test_build_physics_tools_absent_is_a_graceful_noop():
    """Without mixle_pde installed, physics tools simply don't appear — nothing crashes."""
    from mixle_mlops.mcp.physics_tools import build_physics_tools

    assert build_physics_tools() == {}


def test_build_physics_tools_wraps_all_four(monkeypatch):
    _install_fake_pde_tools(monkeypatch)
    from mixle_mlops.mcp.physics_tools import build_physics_tools

    tools = build_physics_tools()
    assert set(tools) == {"run_inversion", "query_posterior", "gassmann", "forward_model"}
    for name, tool in tools.items():
        assert tool.input_schema == PHYSICS_TOOL_SCHEMAS[name]


def test_build_model_tools_merges_physics_tools(monkeypatch):
    _install_fake_pde_tools(monkeypatch)
    tools = build_model_tools(_registry())
    assert {"list_models", "chat__echo", "run_inversion", "query_posterior"} <= set(tools)


def test_build_model_tools_can_exclude_physics(monkeypatch):
    _install_fake_pde_tools(monkeypatch)
    tools = build_model_tools(_registry(), include_physics=False)
    assert "run_inversion" not in tools
    assert "list_models" in tools


def test_tool_registry_has_physics_tools_by_default(monkeypatch):
    _install_fake_pde_tools(monkeypatch)
    registry = ToolRegistry(_registry(), user_id=None)
    assert registry.has("run_inversion")
    assert registry.has("query_posterior")
    assert registry.has("gassmann")
    assert registry.has("forward_model")


def test_tool_registry_include_physics_flag_gates_them(monkeypatch):
    _install_fake_pde_tools(monkeypatch)
    registry = ToolRegistry(_registry(), user_id=None, include_physics=False)
    assert not registry.has("run_inversion")
    assert not registry.has("query_posterior")
    # unaffected: the ordinary MCP tools are still there
    assert registry.has("list_models")


def test_tool_registry_include_mcp_false_still_exposes_physics(monkeypatch):
    """The two flags are independent: dropping the chat/score/list tools must not drop the physics ones."""
    _install_fake_pde_tools(monkeypatch)
    registry = ToolRegistry(_registry(), user_id=None, include_mcp=False, include_mixle=False)
    assert registry.has("run_inversion")
    assert not registry.has("chat__echo")


def test_run_inversion_then_query_posterior_round_trip_carries_prior_dominated(monkeypatch):
    _install_fake_pde_tools(monkeypatch)
    registry = ToolRegistry(_registry(), user_id=None)

    async def _turn():
        inversion = await registry.dispatch(
            "run_inversion", {"dataset_ref": "ds-1", "modality": "gravity", "prior": "blocky"}
        )
        assert isinstance(inversion, dict)
        posterior_ref = inversion["posterior_ref"]
        assert posterior_ref

        queried = await registry.dispatch(
            "query_posterior", {"posterior_ref": posterior_ref, "query": "region_mass"}
        )
        return queried

    result = asyncio.run(_turn())
    assert isinstance(result, dict)
    assert "prior_dominated" in result
    assert result["prior_dominated"] is False  # "blocky" prior, well-constrained per the fixture


def test_query_posterior_missing_arg_is_reported_in_band_not_a_crash(monkeypatch):
    _install_fake_pde_tools(monkeypatch)
    registry = ToolRegistry(_registry(), user_id=None)

    out = asyncio.run(registry.dispatch("query_posterior", {"posterior_ref": "nope"}))  # missing "query"
    assert "error" in out


def test_catalog_entries_absent_without_ic10():
    from mixle_mlops.mcp.physics_tools import catalog_entries

    assert catalog_entries() == []


def test_catalog_entries_register_when_ic10_and_pde_tools_present(monkeypatch):
    _install_fake_pde_tools(monkeypatch)

    class _FakeCatalogEntry:
        def __init__(self, *, id, schema, owner, verifier=None, cost=0.0, reliability=1.0):
            self.id = id
            self.schema = schema
            self.owner = owner
            self.verifier = verifier

    fake_catalog_mod = types.ModuleType("mixle.task.catalog")
    fake_catalog_mod.CatalogEntry = _FakeCatalogEntry
    monkeypatch.setitem(sys.modules, "mixle.task.catalog", fake_catalog_mod)

    from mixle_mlops.mcp.physics_tools import catalog_entries, register_physics_catalog

    entries = catalog_entries()
    assert {e.id for e in entries} == {"run_inversion", "query_posterior", "gassmann", "forward_model"}
    assert all(e.owner == "physics" for e in entries)

    class _FakeCatalog:
        def __init__(self):
            self.registered: dict[str, Any] = {}

        def register(self, entry):
            self.registered[entry.id] = entry

    catalog = _FakeCatalog()
    count = register_physics_catalog(catalog)
    assert count == 4
    assert set(catalog.registered) == {"run_inversion", "query_posterior", "gassmann", "forward_model"}
