"""P1 (mlops half) -- the unified forward-simulation `simulate` tool (IC-11) is wired into ToolRegistry.

The simulation *logic* (``Scenario``/``ScenarioStep``/``SimResult``, ``simulate``, ``register_forward``, and the
registered pde forwards such as the real ``dynamics.AdvectionDiffusionOperator``) lives in
``mixle_pde.simulation_service`` (IC-11), a sibling package landing under its own PR (workstream P, mixle-pde).
To exercise the mlops-side wiring end-to-end without depending on that PR's landing order, this suite installs
a minimal in-memory stand-in for ``mixle_pde.simulation_service`` that honors the frozen IC-11
dataclasses/functions plus the frozen IC-2 content-hashing rule -- the same technique
``test_e4_physics_tools.py`` uses for ``mixle_pde.tools``. Once the real ``mixle_pde.simulation_service``
lands, ``sim_tools._load_simulation_service``'s ``importlib.import_module("mixle_pde.simulation_service")``
picks it up unchanged; nothing in ``mixle_mlops`` needs to change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest

from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_mlops.mcp.sim_tools import register_sim_tools


def _sha256_of_arrays(arrays: dict[str, Any]) -> str:
    """The frozen IC-2 hashing rule: sha256 over ``arrays[k].tobytes()`` for ``k`` in ``sorted(arrays)``."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode("utf-8"))
        h.update(memoryview(arrays[k]).tobytes() if hasattr(arrays[k], "tobytes") else bytes(arrays[k]))
    return h.hexdigest()


def _install_fake_simulation_service(monkeypatch: pytest.MonkeyPatch, store_dir: str) -> types.ModuleType:
    """Install a fake ``mixle_pde.simulation_service`` (IC-11) satisfying the frozen dataclasses/functions,
    with real content-addressed artifact I/O under ``store_dir`` and a toy "transport" forward standing in for
    the real ``dynamics.AdvectionDiffusionOperator`` (out of scope for this mlops-side PR)."""

    @dataclass
    class ScenarioStep:
        op: str
        inputs_ref: str
        params: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Scenario:
        steps: list[ScenarioStep]
        couplings: list[tuple] = field(default_factory=list)
        provenance: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class SimResult:
        result_ref: str
        uncertainty: Any = None
        provenance: dict[str, Any] = field(default_factory=dict)

    _forwards: dict[str, Callable[[str, dict], dict[str, np.ndarray]]] = {}

    def register_forward(op: str, fn: Callable[[str, dict], dict[str, np.ndarray]]) -> None:
        _forwards[op] = fn

    def _transport(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
        """Toy stand-in for the real ``AdvectionDiffusionOperator`` -- a plain explicit diffusion smoothing of
        the plume field over ``params["steps"]`` iterations, scaled by ``params["diffusivity"]``. Real enough
        to exercise a genuine, round-trippable numeric artifact end to end."""
        with np.load(inputs_ref) as data:
            field_arr = np.asarray(data["field"], dtype=float)
        diffusivity = float(params.get("diffusivity", 1e-3))
        n_steps = int(params.get("steps", 1))
        out = field_arr.copy()
        for _ in range(n_steps):
            lap = (
                np.roll(out, 1, axis=0)
                + np.roll(out, -1, axis=0)
                + np.roll(out, 1, axis=1)
                + np.roll(out, -1, axis=1)
                - 4 * out
            )
            out = out + diffusivity * lap
        return {"field": out}

    register_forward("transport", _transport)
    register_forward("dispersion", _transport)

    def write_result_artifact(arrays, *, grid, units, provenance, store_dir):
        os.makedirs(store_dir, exist_ok=True)
        ref = _sha256_of_arrays(arrays)
        np.savez(os.path.join(store_dir, f"{ref}.npz"), **arrays)
        header = {
            "schema": "mixle_pde.sim_result/v1",
            "content_hash": ref,
            "crs": None,
            "grid": grid,
            "units": units,
            "provenance": provenance,
            "created": None,
        }
        with open(os.path.join(store_dir, f"{ref}.json"), "w") as fh:
            json.dump(header, fh)
        return ref

    def read_result_artifact(ref, *, store_dir):
        with np.load(os.path.join(store_dir, f"{ref}.npz")) as data:
            return {k: np.asarray(data[k]) for k in data.files}

    def _run_coupled_dag(scenario):
        raise NotImplementedError("P2")

    def simulate(scenario) -> "SimResult":
        if scenario.couplings:
            return _run_coupled_dag(scenario)
        step = scenario.steps[0]
        arrays = _forwards[step.op](step.inputs_ref, step.params)
        ref = write_result_artifact(arrays, grid={}, units="", provenance={"op": step.op}, store_dir=store_dir)
        return SimResult(result_ref=ref, uncertainty=None, provenance={"op": step.op, "content_hash": ref})

    fake = types.ModuleType("mixle_pde.simulation_service")
    fake.ScenarioStep = ScenarioStep
    fake.Scenario = Scenario
    fake.SimResult = SimResult
    fake.register_forward = register_forward
    fake._FORWARDS = _forwards
    fake.write_result_artifact = write_result_artifact
    fake.read_result_artifact = read_result_artifact
    fake._run_coupled_dag = _run_coupled_dag
    fake.simulate = simulate
    fake.sha256_of_arrays = _sha256_of_arrays

    monkeypatch.setitem(sys.modules, "mixle_pde.simulation_service", fake)
    return fake


def test_build_sim_tools_absent_is_a_graceful_noop():
    """Without the IC-11 module, the simulate tool simply doesn't appear -- nothing crashes."""
    try:
        from mixle_pde import simulation_service  # noqa: F401

        pytest.skip("mixle_pde.simulation_service is present in this environment")
    except ImportError:
        pass
    import mixle_mlops.mcp.sim_tools as sim_tools_mod

    assert sim_tools_mod.build_sim_tools() == {}


def test_register_sim_tools_wires_simulate_onto_registry(monkeypatch, tmp_path):
    _install_fake_simulation_service(monkeypatch, str(tmp_path / "store"))
    reg = ToolRegistry(ModelRegistry(), include_mixle=False)
    register_sim_tools(reg)
    assert reg.has("simulate")


def test_simulate_dispatch_content_addressed_round_trip(monkeypatch, tmp_path):
    """The literal P1 DoD: register the tool, dispatch a one-step "transport" scenario, and confirm the
    returned ref is a 64-hex content hash whose artifact exists and re-hashes to itself on read-back."""
    store_dir = str(tmp_path / "store")
    _install_fake_simulation_service(monkeypatch, store_dir)
    from mixle_pde.simulation_service import read_result_artifact, sha256_of_arrays

    plume_path = tmp_path / "plume.npz"
    rng = np.random.default_rng(0)
    np.savez(plume_path, field=rng.normal(size=(64, 64)))

    reg = ToolRegistry(ModelRegistry(), include_mixle=False)
    register_sim_tools(reg)
    assert reg.has("simulate")

    async def _turn():
        return await reg.dispatch(
            "simulate",
            {
                "scenario": {
                    "steps": [
                        {
                            "op": "transport",
                            "inputs_ref": str(plume_path),
                            "params": {"diffusivity": 1e-3, "velocity": 0.2, "n": 64, "steps": 20},
                        }
                    ],
                    "couplings": [],
                    "provenance": {},
                }
            },
        )

    out = asyncio.run(_turn())
    assert isinstance(out, dict) and "result_ref" in out, out
    ref = out["result_ref"]

    assert isinstance(ref, str) and len(ref) == 64 and all(c in "0123456789abcdef" for c in ref)
    assert os.path.exists(os.path.join(store_dir, f"{ref}.npz"))
    assert sha256_of_arrays(read_result_artifact(ref, store_dir=store_dir)) == ref


def test_simulate_dispatch_reports_invalid_scenario_in_band(monkeypatch, tmp_path):
    _install_fake_simulation_service(monkeypatch, str(tmp_path / "store"))
    reg = ToolRegistry(ModelRegistry(), include_mixle=False)
    register_sim_tools(reg)

    out = asyncio.run(reg.dispatch("simulate", {"scenario": {}}))  # missing "steps"
    assert isinstance(out, dict) and "error" in out


def test_register_sim_tools_overrides_generic_platform_simulate_tool(monkeypatch, tmp_path):
    """`ToolRegistry._build` (untouched by this PR) already registers a different `"simulate"` tool -- forward
    sampling records from a hosted mixle model. `register_sim_tools` is opt-in/additive via the pre-existing
    `_add`; when a caller invokes it, the IC-11 physics-scenario tool takes precedence on that registry, which
    is the intended behavior for a registry a frontier model uses to drive physics what-ifs."""
    _install_fake_simulation_service(monkeypatch, str(tmp_path / "store"))
    reg = ToolRegistry(ModelRegistry(), include_mixle=False)
    assert reg.has("simulate")  # the pre-existing generic model-sampling tool
    register_sim_tools(reg)
    assert reg.has("simulate")  # now the IC-11 physics-scenario tool

    out = asyncio.run(reg.dispatch("simulate", {"scenario": {}}))  # the generic tool expects "spec", not
    assert isinstance(out, dict) and "error" in out  # "scenario" -- confirms ours answered, not it


def test_load_simulation_service_honors_a_monkeypatched_fake_even_after_a_real_import_cached_the_attribute(
    monkeypatch, tmp_path
):
    """Regression test for the mixle_pde submodule import-caching bug (distinct from #61's TreeLogitProvider
    KV-cache bug) -- mirrors ``test_e4_physics_tools.py``'s analogous regression test, but for
    ``_load_simulation_service``/``mixle_pde.simulation_service``.

    A real ``from mixle_pde import simulation_service`` anywhere earlier in the process (e.g. this file's own
    ``test_build_sim_tools_absent_is_a_graceful_noop``, whose probe import still runs for real even though that
    test then skips) sets a ``simulation_service`` attribute on the ``mixle_pde`` package module. ``from
    mixle_pde import simulation_service`` elsewhere then resolves via ``_handle_fromlist``, which skips
    ``sys.modules`` entirely once that attribute already exists -- silently ignoring a later
    ``monkeypatch.setitem(sys.modules, "mixle_pde.simulation_service", fake)`` and returning the real module
    instead. Forces that poisoning inline -- rather than relying on incidental ordering against another test --
    and asserts ``_load_simulation_service`` (which uses ``importlib.import_module`` precisely to sidestep
    this) still picks up the fake regardless.
    """
    real_service = pytest.importorskip("mixle_pde.simulation_service")
    import mixle_pde

    assert mixle_pde.simulation_service is real_service  # sanity: the poisoning precondition holds

    _install_fake_simulation_service(monkeypatch, str(tmp_path / "store"))
    fake = sys.modules["mixle_pde.simulation_service"]

    import mixle_mlops.mcp.sim_tools as sim_tools_mod

    loaded = sim_tools_mod._load_simulation_service()
    assert loaded is fake, "a real import earlier in the process shadowed the monkeypatched fake"
    assert loaded is not real_service
