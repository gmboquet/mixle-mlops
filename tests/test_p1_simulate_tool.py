"""P1 DoD: the unified IC-11 `simulate` MCP tool, end to end through `ToolRegistry`.

Builds a `ToolRegistry`, registers the physics `simulate` tool via `register_sim_tools` (the additive
`_add` seam), dispatches a single-step "transport" scenario against a tiny synthetic plume `.npz`, and
verifies the returned `result_ref` is a genuine content-hashed handle: 64 hex chars, backed by
`{store}/{ref}.npz` on disk, and `sha256_of_arrays` of the read-back arrays reproduces `ref` exactly
(content-addressed read-back of the plume-transport result)."""

import asyncio

import numpy as np

from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_mlops.mcp.sim_tools import register_sim_tools
from mixle_pde.io.artifacts import sha256_of_arrays
from mixle_pde.simulation_service import read_result_artifact


def _is_hex64(s: str) -> bool:
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def test_p1_simulate_tool_transport_round_trip(tmp_path, monkeypatch):
    store = tmp_path / "sim_store"
    monkeypatch.setenv("MIXLE_PDE_SIM_STORE_DIR", str(store))

    n = 64
    plume = tmp_path / "plume.npz"
    x = np.arange(n)
    field0 = np.exp(-((x - n / 3.0) ** 2) / 20.0)
    np.savez(plume, field=field0)

    registry = ModelRegistry()
    reg = ToolRegistry(registry, include_mixle=False)
    register_sim_tools(reg)
    assert reg.has("simulate")

    scenario = {
        "steps": [
            {
                "op": "transport",
                "inputs_ref": str(plume),
                "params": {"diffusivity": 1e-3, "velocity": 0.2, "n": n, "steps": 20},
            }
        ],
        "couplings": [],
        "provenance": {},
    }

    async def _run():
        return await reg.dispatch("simulate", {"scenario": scenario})

    result = asyncio.run(_run())

    ref = result["result_ref"]
    assert _is_hex64(ref)
    npz_path = store / f"{ref}.npz"
    assert npz_path.exists()

    arrays = read_result_artifact(ref, store_dir=str(store))
    assert sha256_of_arrays(arrays) == ref


def test_p1_simulate_tool_replaces_generic_simulate_name(tmp_path, monkeypatch):
    """`register_sim_tools` intentionally overrides `ToolRegistry`'s generic mixle-model `simulate` tool
    with the IC-11 physics one -- a bare `simulate` dispatch must go through the physics forward path, not
    the generic model-sampling handler, once `register_sim_tools` has run."""
    monkeypatch.setenv("MIXLE_PDE_SIM_STORE_DIR", str(tmp_path / "sim_store"))
    registry = ModelRegistry()
    reg = ToolRegistry(registry)  # include_platform=True by default -> generic "simulate" exists first
    assert reg.has("simulate")
    register_sim_tools(reg)
    assert reg.has("simulate")

    async def _run():
        # a scenario dispatch would fail against the generic handler (no `spec.model`); success here
        # proves the physics `simulate` tool is the one actually installed.
        n = 16
        plume = tmp_path / "plume2.npz"
        np.savez(plume, field=np.ones(n))
        scenario = {
            "steps": [
                {
                    "op": "transport",
                    "inputs_ref": str(plume),
                    "params": {"diffusivity": 1e-3, "velocity": 0.1, "n": n, "steps": 1},
                }
            ],
            "couplings": [],
            "provenance": {},
        }
        return await reg.dispatch("simulate", {"scenario": scenario})

    result = asyncio.run(_run())
    assert "result_ref" in result and _is_hex64(result["result_ref"])
