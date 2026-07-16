"""E11 -- worked end-to-end exploration example: data -> invert -> interpret -> decide -> receipt.

A synthetic, labelled gravity survey walks the full exploration spine in one process:

    1. a synthetic dataset (a small ore-like density anomaly, known ground truth for scoring/calibration)
    2. ``run_inversion`` -- a Bayesian linear-Gaussian inversion (E4 physics-tool surface, IC-3)
    3. ``query_posterior`` -- a calibrated ``region_mass`` tonnage distribution (IC-3/IC-8)
    4. ``POST /v1/interpret`` -- a calibrated natural-language claim or an honest abstain (E5, real FastAPI app)
    5. ``decision_receipt`` -- an auditable, hash-linked memo (E7, IC-5 envelope)
    6. an IC-13 knowledge bundle export + persist/reload round-trip (M0b/M1a)

Grounding note (read before touching the "local stand-in" pieces below): this task (E11) lists
E1, E2, E4, M0b, M1a among its declared dependencies. As of this PR, on ``release/0.8.0``:

  * E1 (posterior conformance in ``mixle_pde.latent``) and E2 (``mixle_pde.io.artifacts``) have NOT
    landed -- ``mixle_pde.latent.PosteriorField3D`` still exposes ``sample`` (singular) and
    ``credible_interval(alpha=...)``, and there is no ``mixle_pde/io/`` package to serialize to.
  * E4 (``mixle_pde.tools`` / ``mixle_mlops.mcp.physics_tools``) has NOT landed -- there is no
    ``run_inversion``/``query_posterior`` tool wrapper to call.
  * M1a (knowledge-bundle persistence) has NOT landed in ``mixle-knowledge`` -- only M0's frozen
    ``mixle_knowledge.contracts`` pydantic models exist.
  * E3 (``FieldPosteriorAdapter``), E5 (``POST /v1/interpret``), E7 (provenance receipt pattern), E10
    (physics/calibration verifiers), and M0 (IC-13 contracts) HAVE landed, and this example calls the
    real modules for all of those.

This is the same situation E3/E5/E7/E10's own PRs already found and handled: each of those tasks'
tests build a small local stand-in for the specific still-missing piece they depend on (see e.g.
``mixle-mlops/tests/test_e7_provenance.py``'s ``_decision_receipt``), rather than blocking on an
upstream PR. This example follows the identical, already-established convention: the handful of
``_ic2_*``/``_run_inversion``/``_query_posterior``/``_decision_receipt`` helpers below are thin,
clearly-labelled local stand-ins for the still-missing E1/E2/E4/M1a pieces -- built to the exact frozen
IC-2/IC-3/IC-5 shapes so each swaps for a one-line import the moment the real module lands. Everything
downstream of them (E3's adapter, E5's route, E7's hashing/substrate reuse, E10's verifiers, M0's
contracts) is the real, already-shipped code, called as-is.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from mixle.data.hashing import dataset_hash
from mixle.inference.production.provenance import build_header
from mixle.substrate.core import Substrate
from mixle.substrate.ingest import ingest_artifacts
from mixle_pde.decision_quantities import region_mass as _region_mass
from mixle_pde.geophysics import gravity_point_sensitivity
from mixle_pde.latent import Field3D, PosteriorField3D

try:  # IC-5's frozen envelope validator -- lands with E7's dependency chain; not always present pre-E1.
    from mixle.task.trace_record import STEP_KEYS, TRACE_KEYS, validate_trace_record
except ImportError:  # pragma: no cover - exercised only on checkouts missing IC-5
    TRACE_KEYS = ("prompt", "steps", "outcome", "provenance")
    STEP_KEYS = ("tool", "args", "result", "model", "verdict")

    def validate_trace_record(d: dict[str, Any]) -> None:
        missing = [k for k in TRACE_KEYS if k not in d]
        if missing:
            raise ValueError(f"trace record missing frozen keys: {missing}")
        for i, s in enumerate(d.get("steps") or []):
            for k in ("tool", "args", "result"):
                if k not in s:
                    raise ValueError(f"step {i} missing frozen key {k!r}")


# ================================================================================================
# Local stand-in #1 (E1/E2): an IC-1-conforming wrapper over a real `PosteriorField3D`, plus the
# frozen IC-2 artifact-header shape (`save_posterior`/`load_posterior`/`content_hash`/`sha256_of_arrays`).
# Swap for `from mixle_pde.io.artifacts import save_posterior, load_posterior, content_hash` and drop
# `IC1PosteriorAdapter` (once E1 lands the `.samples`/`credible_interval(level)` rename directly on
# `PosteriorField3D`) the moment those two tasks land -- the call sites below do not otherwise change.
# ================================================================================================

ARTIFACT_SCHEMA = "mixle_pde.field_posterior/v1"
HEADER_KEYS = ("schema", "content_hash", "crs", "grid", "units", "provenance", "created")


def sha256_of_arrays(arrays: dict[str, Any]) -> str:
    """IC-2's frozen hashing rule (notes/exec/contracts.md), copied verbatim: sha256 over each array's
    bytes taken in sorted-key order, so the digest is order-independent and stable."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        v = arrays[k]
        h.update(k.encode("utf-8"))
        h.update(memoryview(v).tobytes() if hasattr(v, "tobytes") else bytes(v))
    return h.hexdigest()


class IC1PosteriorAdapter:
    """Adapts a real ``mixle_pde.latent.PosteriorField3D`` onto the frozen IC-1 ``Posterior`` protocol.

    ``PosteriorField3D`` (pre-E1) exposes ``sample`` (singular) and ``credible_interval(alpha=...)``;
    IC-1 freezes ``samples(n, rng)`` (plural) and ``credible_interval(level)``. This wrapper is exactly
    the adapter E1 is scoped to build directly onto ``PosteriorField3D`` -- until it lands, wrapping
    here keeps every IC-1 consumer (E3's adapter, E5's ``describe_posterior``, E10's calibration
    verifier) working against the real posterior unmodified. Any attribute this class does not define
    itself (``grid``, ``map``, ``precision_factor``, ``slice``, ...) falls through to the wrapped object.
    """

    def __init__(self, field_posterior: PosteriorField3D) -> None:
        self._field = field_posterior

    def __getattr__(self, name: str) -> Any:
        return getattr(self._field, name)

    @property
    def native(self) -> PosteriorField3D:
        """The wrapped, concrete ``PosteriorField3D`` -- needed by IC-8 functions (``region_mass``) that
        are typed against the concrete class rather than the structural IC-1 protocol."""
        return self._field

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._field.sample(n, rng)

    @property
    def mean(self) -> np.ndarray:
        return self._field.mean

    @property
    def cov(self) -> np.ndarray | None:
        return self._field.cov

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        return self._field.credible_interval(alpha=1.0 - level)

    def derived_quantity(self, fn, n: int, rng: np.random.Generator) -> "_DerivedQuantity":
        return _DerivedQuantity(fn(self.samples(n, rng)))


class _DerivedQuantity:
    """A minimal IC-1 ``DerivedQuantity``: draws + a central credible interval + the honesty flag."""

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = np.asarray(samples, dtype=float)
        self.prior_dominated = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


def _ic2_save_posterior(
    field_posterior: PosteriorField3D, path: str, *, parent_content_hash: str | None = None
) -> str:
    """Local stand-in for E2's ``mixle_pde.io.artifacts.save_posterior`` (IC-2): writes ``{path}/posterior.npz``
    (the arrays) + a sibling ``{path}/manifest.json`` header with the frozen ``HEADER_KEYS``."""
    grid = field_posterior.grid
    arrays: dict[str, Any] = {
        "mean": field_posterior.mean,
        "map": field_posterior.map,
        "coordinates": grid.coordinates,
    }
    if field_posterior.cov is not None:
        arrays["cov"] = field_posterior.cov
    elif field_posterior.diag_var is not None:
        arrays["diag_var"] = field_posterior.diag_var
    content_hash = sha256_of_arrays(arrays)

    adir = Path(path)
    adir.mkdir(parents=True, exist_ok=True)
    np.savez(adir / "posterior.npz", **arrays)
    manifest = {
        "mixle_artifact": "field_posterior",
        "schema": ARTIFACT_SCHEMA,
        "content_hash": content_hash,
        "crs": None,  # mesh-local coordinates (no CRS attached in this synthetic worked example)
        "grid": {"shape": [grid.n], "origin": grid.coordinates[0].tolist(), "spacing": grid.spacing},
        "units": grid.units,
        "provenance": {
            "content_hash": parent_content_hash,
            "stage": "inversion",
            "property_name": grid.property_name,
        },
        "created": time.time(),
    }
    (adir / "manifest.json").write_text(json.dumps(manifest))
    return str(adir)


def _ic2_load_posterior(path: str) -> IC1PosteriorAdapter:
    """Local stand-in for E2's ``mixle_pde.io.artifacts.load_posterior``: reconstructs a real
    ``PosteriorField3D`` from the ``{path}/posterior.npz`` + ``manifest.json`` pair, IC-1-wrapped."""
    adir = Path(path)
    manifest = json.loads((adir / "manifest.json").read_text())
    data = np.load(adir / "posterior.npz")
    grid = Field3D(
        coordinates=data["coordinates"],
        spacing=manifest["grid"]["spacing"],
        units=manifest["units"],
        property_name=manifest["provenance"].get("property_name", "field"),
        bounds=None,
    )
    kwargs: dict[str, Any] = {"grid": grid, "mean": data["mean"], "map": data["map"]}
    if "cov" in data:
        kwargs["cov"] = data["cov"]
    elif "diag_var" in data:
        kwargs["diag_var"] = data["diag_var"]
    return IC1PosteriorAdapter(PosteriorField3D(**kwargs))


def _ic2_content_hash(path: str) -> str:
    """Local stand-in for E2's ``mixle_pde.io.artifacts.content_hash``."""
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    return manifest["content_hash"]


# ================================================================================================
# Local stand-in #2 (E4): the ``run_inversion`` / ``query_posterior`` physics-tool functions (IC-3).
# Swap for ``from mixle_pde.tools import run_inversion, query_posterior`` the moment E4 lands -- the
# call/return shapes here are exactly the frozen IC-3 signatures, so `main()` does not change.
# ================================================================================================

_DATASET_STORE: dict[str, dict[str, Any]] = {}


def build_synthetic_survey(rng: np.random.Generator) -> dict[str, Any]:
    """A small, labelled synthetic gravity survey: a 3x3x2 density-contrast grid with one ore-like
    anomaly at depth, surveyed by a 4x4 station grid at the surface. ``true_model`` is the label kept
    for calibration/scoring -- a real deployment would not have it at inversion time."""
    spacing = 20.0
    nx, ny = 3, 3
    xs = (np.arange(nx) - (nx - 1) / 2.0) * spacing
    ys = (np.arange(ny) - (ny - 1) / 2.0) * spacing
    zs = np.array([-25.0, -50.0])  # z is up-positive, so cells sit below the surface (z=0)

    cells = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    n = cells.shape[0]
    volume = spacing**3

    true_model = np.zeros(n)
    shallow_layer = np.arange(nx * ny)  # the z=-25 layer -- the surveyed target horizon
    anomaly_cells = shallow_layer[[4]]  # the center cell: a compact, near-surface ore body
    true_model[anomaly_cells] = 2200.0  # kg/m^3 excess density contrast (a strong sulfide-vs-host signal)

    gx = (np.arange(5) - 2.0) * 15.0
    gy = (np.arange(5) - 2.0) * 15.0
    obs = np.array([[x, y, 1.0] for y in gy for x in gx])  # a station sits directly above the body

    G = gravity_point_sensitivity(obs, cells, volume)
    noise_std = 0.004  # mGal, a good (but realistic) microgravity survey precision
    data = G @ true_model + rng.normal(0.0, noise_std, size=obs.shape[0])

    target_region = np.zeros(n, dtype=bool)
    target_region[shallow_layer] = True  # the target horizon this survey was designed to test

    return {
        "name": "synthetic_gravity_survey",
        "obs": obs,
        "cells": cells,
        "spacing": spacing,
        "volume": volume,
        "G": G,
        "data": data,
        "noise_std": noise_std,
        "true_model": true_model,
        "target_region": target_region,
    }


def register_dataset(dataset: dict[str, Any]) -> str:
    """Fingerprint + persist a synthetic dataset, returning its content-hashed ``dataset_ref`` (the
    string handle ``run_inversion`` takes).

    Writes the frozen-shape minimal JSON bundle ``mixle_pde.tools._load_dataset_bundle`` reads
    (``{grid: {coordinates, spacing, units, property_name, cell_volumes}, observations: [{location,
    value, noise_cov, units}]}``, see that function's docstring) to a real file under a
    content-hash-keyed path, so ``dataset_ref`` resolves for BOTH consumers: the real, already-shipped
    ``mixle_pde.tools.run_inversion`` (E4 -- it ``open()``s ``dataset_ref`` directly, the same way
    ``mixle-pde/tests/tools_test.py``'s own ``_tiny_gravity_dataset`` fixture does; there is no
    dataset-artifact registry for it to resolve a bare hash through) and this module's own
    local-stand-in ``_run_inversion``/``main()`` walkthrough (which still look the full dataset --
    including the precomputed sensitivity matrix ``G`` that no minimal JSON bundle carries -- up in
    ``_DATASET_STORE`` by this same ``dataset_ref`` string).

    Before E4 landed in mixle-pde, this returned a bare :func:`mixle.data.hashing.dataset_hash` digest
    backed by nothing but ``_DATASET_STORE``, on the theory that it mirrors how a real deployment
    would resolve a ``dataset_ref`` through a registry keyed by that hash -- but no such registry has
    ever existed anywhere upstream, and once the real ``run_inversion`` landed (merged ~40 minutes
    after this module, per git history -- the two PRs were in flight at the same time and never
    reconciled), a bare hash stopped being openable, breaking every seam-test path that exercises the
    real tool. Keying the file's path by the hash keeps the handle content-derived (registering an
    identical dataset twice is idempotent: same bytes in, same path out) while also being a real,
    ``open()``-able file, mirroring ``mixle_pde.io.artifacts``'s own posterior_ref pattern (a real path
    plus a separately-derivable content hash) rather than inventing a new convention.
    """
    n = dataset["cells"].shape[0]
    bundle = {
        "grid": {
            "coordinates": dataset["cells"].tolist(),
            "spacing": float(dataset["spacing"]),
            "units": "kg/m^3",
            "property_name": "density_contrast",
            "cell_volumes": [float(dataset["volume"])] * n,
        },
        "observations": [
            {
                "location": dataset["obs"].tolist(),
                "value": dataset["data"].tolist(),
                "noise_cov": [float(dataset["noise_std"]) ** 2] * int(dataset["data"].size),
                "units": "mGal",
            }
        ],
    }
    content_ref = dataset_hash(dataset["data"].tolist())
    workdir = Path(tempfile.gettempdir()) / "mixle_e11_datasets" / content_ref
    workdir.mkdir(parents=True, exist_ok=True)
    path = str(workdir / "dataset.json")
    with open(path, "w") as f:
        json.dump(bundle, f)

    _DATASET_STORE[path] = dataset
    return path


def _run_inversion(dataset_ref: str, modality: str, prior: str, config: dict | None = None) -> dict:
    """Local stand-in for E4's ``mixle_pde.tools.run_inversion`` (IC-3): fits a posterior and returns
    ``{posterior_ref, diagnostics}``, ``posterior_ref`` content-hashed (IC-2)."""
    if modality != "gravity":
        raise NotImplementedError(f"only modality='gravity' is wired in this worked example, got {modality!r}")
    dataset = _DATASET_STORE[dataset_ref]
    cfg = config or {}
    prior_var = float(cfg.get("prior_var", 800.0**2))
    length_scale = float(cfg.get("length_scale", 1.2 * dataset["spacing"]))

    cells = dataset["cells"]
    G = dataset["G"]
    n = cells.shape[0]
    dist = np.linalg.norm(cells[:, None, :] - cells[None, :, :], axis=2)
    if prior == "smooth":
        cov_prior = prior_var * np.exp(-(dist**2) / (2.0 * length_scale**2))
    elif prior == "compact":
        cov_prior = prior_var * np.exp(-dist / length_scale)
    else:  # "blocky" / "anisotropic": this worked example falls back to an independent diagonal prior
        cov_prior = prior_var * np.eye(n)

    noise_var = dataset["noise_std"] ** 2
    prior_precision = np.linalg.inv(cov_prior + 1e-9 * np.eye(n))
    precision_post = (G.T @ G) / noise_var + prior_precision
    cov_post = np.linalg.inv(precision_post)
    mean_post = cov_post @ (G.T @ dataset["data"] / noise_var)

    grid = Field3D(
        coordinates=cells, spacing=dataset["spacing"], units="kg/m^3", property_name="density_contrast", bounds=None
    )
    posterior = PosteriorField3D(grid=grid, mean=mean_post, cov=cov_post)

    workdir = Path(tempfile.mkdtemp(prefix="mixle_e11_artifacts_"))
    posterior_ref = _ic2_save_posterior(posterior, str(workdir / "posterior_001"), parent_content_hash=dataset_ref)
    # Auxiliary honesty-flag inputs (work-plan A2's prior/posterior variance reduction), stored as a
    # sibling file -- NOT part of the content-hashed posterior artifact itself.
    np.savez(
        Path(posterior_ref) / "diagnostics.npz",
        prior_var_diag=np.diag(cov_prior),
        posterior_var_diag=np.diag(cov_post),
    )

    diagnostics = {
        "n_cells": n,
        "n_observations": int(G.shape[0]),
        "prior": prior,
        "data_misfit_rms": float(np.sqrt(np.mean((G @ mean_post - dataset["data"]) ** 2))),
    }
    return {"posterior_ref": posterior_ref, "diagnostics": diagnostics}


def _query_posterior(posterior_ref: str, query: str, params: dict | None = None) -> dict:
    """Local stand-in for E4's ``mixle_pde.tools.query_posterior`` (IC-3): dispatches to an IC-8 decision
    quantity and ALWAYS returns ``prior_dominated``."""
    params = params or {}
    wrapper = _ic2_load_posterior(posterior_ref)
    raw = wrapper.native

    if query != "region_mass":
        raise NotImplementedError(f"query {query!r} is not wired in this worked example")

    region = np.asarray(params.get("region", np.ones(raw.grid.n, dtype=bool)), dtype=bool)
    cell_volumes = params.get("cell_volumes", raw.grid.spacing**3)
    diag_path = Path(posterior_ref) / "diagnostics.npz"
    prior_var = posterior_var = None
    if diag_path.exists():
        diag = np.load(diag_path)
        prior_var, posterior_var = diag["prior_var_diag"], diag["posterior_var_diag"]

    dq = _region_mass(raw, region, cell_volumes, prior_var=prior_var, posterior_var=posterior_var)
    level = float(params.get("level", 0.9))
    lo, hi = dq.credible_interval(level)
    return {
        "distribution": {"mean": float(dq.mean), "std": float(dq.std)},
        "interval": {"level": level, "lo": float(lo), "hi": float(hi)},
        "prior_dominated": bool(dq.prior_dominated),
    }


# ================================================================================================
# E5 -- POST /v1/interpret, the real gateway route (imported, not stubbed)
# ================================================================================================


def run_interpret(posterior_ref: str, *, field: str, tol: float, level: float = 0.9) -> dict:
    """Drive the real ``POST /v1/interpret`` route (E5) in-process via a FastAPI ``TestClient``.

    ``interpret_route.resolve_posterior`` is a swappable module-level hook by design (see the route's
    own docstring): the default tries ``mixle_pde.io.artifacts.load_posterior`` (E2, not yet landed),
    so this points it at the local IC-2 stand-in above -- exactly the intended wiring point, not a hack.
    """
    from fastapi.testclient import TestClient

    import mixle_mlops.storage.db as db
    from mixle_mlops.config import get_settings
    from mixle_mlops.gateway.app import create_app
    from mixle_mlops.gateway.routes import interpret as interpret_route

    with tempfile.TemporaryDirectory(prefix="mixle_e11_gateway_") as data_dir:
        os.environ["MIXLE_DATA_DIR"] = data_dir
        get_settings.cache_clear()
        db._engine = None
        previous_resolver = interpret_route.resolve_posterior
        interpret_route.resolve_posterior = lambda ref: _ic2_load_posterior(ref)
        try:
            app = create_app()
            with TestClient(app) as client:
                signup = client.post(
                    "/auth/signup", json={"email": "e11-explorer@example.com", "password": "pw123456"}
                )
                signup.raise_for_status()
                headers = {"Authorization": f"Bearer {signup.json()['api_key']}"}
                resp = client.post(
                    "/v1/interpret",
                    headers=headers,
                    json={"posterior_ref": posterior_ref, "field": field, "tol": tol, "level": level},
                )
                resp.raise_for_status()
                return resp.json()
        finally:
            interpret_route.resolve_posterior = previous_resolver
            get_settings.cache_clear()
            db._engine = None


# ================================================================================================
# Local stand-in #3 (E7): the decision receipt (IC-5 envelope), following the exact convention
# ``mixle-mlops/tests/test_e7_provenance.py`` already established for the same still-missing upstream
# piece (``mixle_pde.reasoning.decision_receipt``). Swap for
# ``from mixle_pde.reasoning import decision_receipt`` the moment E7's primary-repo half lands.
# ================================================================================================


def _item_hash(ref: str, manifest: dict[str, Any]) -> str:
    """An IC-13-style enclosing item hash over metadata+ref (work-plan M1a)."""
    blob = ref + json.dumps(manifest, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class _DummyModel:
    """Concrete enough for ``build_header`` to introspect; it degrades unknown fields to ``None``."""


def decision_receipt(
    *, dataset_ref: dict, posterior_ref: dict, claim: dict, decision: dict, substrate: Substrate
) -> dict:
    """Local stand-in for E7's public API (``mixle_pde.reasoning.decision_receipt``). Walks
    data -> posterior -> claim -> decision, stamping every edge's ``provenance.content_hash`` with its
    parent's hash, and returns the frozen IC-5 envelope."""
    steps: list[dict[str, Any]] = [
        {
            "tool": "ingest_dataset",
            "args": {"dataset_ref": dataset_ref["name"]},
            "result": {"content_hash": dataset_ref["content_hash"], "n_records": dataset_ref["n_records"]},
            "model": None,
            "verdict": None,
        },
        {
            "tool": "run_inversion",
            "args": {"dataset_ref": dataset_ref["content_hash"], "modality": "gravity", "prior": "smooth"},
            "result": {
                "posterior_ref": posterior_ref["item_id"],
                "content_hash": posterior_ref["content_hash"],
                "item_hash": posterior_ref["item_hash"],
            },
            "model": None,
            "verdict": None,
            "provenance": {"content_hash": dataset_ref["content_hash"]},
        },
        {
            "tool": "query_posterior",
            "args": {"posterior_ref": posterior_ref["item_id"], "query": "region_mass"},
            "result": {"value": claim["value"], "content_hash": claim["content_hash"]},
            "model": claim["model"],
            "verdict": None,
            "provenance": {"content_hash": posterior_ref["content_hash"]},
        },
        {
            "tool": "decide",
            "args": {"claim_ref": claim["content_hash"]},
            "result": dict(decision, content_hash=decision["content_hash"]),
            "model": decision["model"],
            "verdict": decision.get("verdict"),
            "provenance": {"content_hash": claim["content_hash"]},
        },
    ]
    lineage = [
        {"stage": "data", "content_hash": dataset_ref["content_hash"]},
        {"stage": "inversion", "content_hash": posterior_ref["content_hash"], "parent": dataset_ref["content_hash"]},
        {"stage": "interpretation", "content_hash": claim["content_hash"], "parent": posterior_ref["content_hash"]},
        {"stage": "decision", "content_hash": decision["content_hash"], "parent": claim["content_hash"]},
    ]
    return {
        "prompt": f"drill decision for {dataset_ref['name']}",
        "steps": steps,
        "outcome": {k: v for k, v in decision.items() if k not in ("model", "verdict")},
        "provenance": {"lineage": lineage, "content_hash": decision["content_hash"]},
    }


# ================================================================================================
# M0b/M1a -- IC-13 knowledge-bundle export + persist/reload. M0's frozen contracts (``mixle_knowledge
# .contracts``) have landed; M1a's real persistence layer has not, so `persist_bundle`/`reload_bundle`
# below are a thin file round-trip over the frozen pydantic model's own (lossless) JSON serialization --
# the wire format M1a's real store would sit on top of.
# ================================================================================================


def build_knowledge_bundle(*, dataset_ref: str, posterior_ref: str, posterior_hash: str, claim: dict, decision: dict):
    from mixle_knowledge.contracts import KnowledgeBundle, KnowledgeItem, Modality, ResourceKind

    posterior_item = KnowledgeItem(
        id="posterior-001",
        kind=ResourceKind.ARTIFACT,
        modality=Modality.GEOSPATIAL,
        schema_uri="mixle://schema/field-posterior/1",
        content_hash=posterior_hash,
        artifact_ref=posterior_ref,  # arrays stay behind the ref -- never duplicated into the bundle
        metadata={"dataset_ref": dataset_ref},
    )
    claim_item = KnowledgeItem(
        id="claim-001",
        kind=ResourceKind.ARTIFACT,
        modality=Modality.STRUCTURED,
        schema_uri="mixle://schema/uq-claim/1",
        content_hash=claim["content_hash"],
        payload=claim,  # structured UQ payload, deep round-trips (not flattened to text)
    )
    decision_item = KnowledgeItem(
        id="decision-001",
        kind=ResourceKind.ARTIFACT,
        modality=Modality.STRUCTURED,
        schema_uri="mixle://schema/decision/1",
        content_hash=decision["content_hash"],
        payload=decision,
    )
    return KnowledgeBundle(
        id="bundle-e11-exploration",
        project_id="exploration-e2e",
        task="drill decision for synthetic gravity survey",
        target_kind="model",
        items=[posterior_item, claim_item, decision_item],
    )


def persist_bundle(bundle, path: str) -> None:
    """Local stand-in for M1a: the frozen ``KnowledgeBundle`` pydantic model already gives an exact,
    lossless JSON round-trip; M1a's job is a database-backed store with revision semantics on top of
    this wire format, not a new one, so this file round-trip exercises the same contract."""
    Path(path).write_text(bundle.model_dump_json())


def reload_bundle(path: str):
    from mixle_knowledge.contracts import KnowledgeBundle

    return KnowledgeBundle.model_validate_json(Path(path).read_text())


# ================================================================================================
# main(): the worked walkthrough
# ================================================================================================


def main() -> None:
    rng = np.random.default_rng(0)
    print("=== Exploration E2E: synthetic gravity survey -> calibrated drill decision ===\n")

    # 1) synthetic, labelled dataset (true_model is the label, kept for scoring -- not for inversion).
    dataset = build_synthetic_survey(rng)
    dataset_ref = register_dataset(dataset)
    print(
        f"[1/5] dataset: {dataset['obs'].shape[0]} gravity stations, {dataset['cells'].shape[0]} model cells, "
        f"dataset_ref={dataset_ref[:16]}..."
    )

    # 2) run_inversion (E4 tool surface / IC-3): Bayesian linear-Gaussian inversion, smooth prior.
    inversion = _run_inversion(dataset_ref, modality="gravity", prior="smooth")
    posterior_ref = inversion["posterior_ref"]
    diag = inversion["diagnostics"]
    print(f"[2/5] run_inversion: rms misfit={diag['data_misfit_rms']:.4f} mGal, posterior_ref={posterior_ref}")

    # 3) query_posterior region_mass (E4 tool / IC-8 decision quantity): calibrated tonnage over the
    # known target horizon, with the prior_dominated honesty flag ALWAYS present.
    query = _query_posterior(
        posterior_ref,
        "region_mass",
        params={"region": dataset["target_region"], "cell_volumes": dataset["volume"], "level": 0.9},
    )
    interval = query["interval"]
    print(
        f"[3/5] query_posterior(region_mass): mean={query['distribution']['mean']:.1f} kg, "
        f"90% CI=({interval['lo']:.1f}, {interval['hi']:.1f}) kg, prior_dominated={query['prior_dominated']}"
    )

    # 4) POST /v1/interpret (E5, real gateway route): a calibrated natural-language claim about the
    # total anomalous signal, or an honest abstain.
    tol = 600.0  # kg/m^3: the absolute precision a driller needs before acting on this claim
    interpret = run_interpret(posterior_ref, field="total", tol=tol, level=0.9)
    if interpret["abstained"]:
        print(f"[4/5] interpret: ABSTAINED (posterior too diffuse relative to tol={tol:g})")
    else:
        print(f"[4/5] interpret: {interpret['claim']!r}")

    # 5) decision_receipt (E7): an auditable, hash-linked memo from data -> decision.
    observations_hash = dataset_hash(dataset["data"].tolist())
    dataset_ref_record = {
        "name": dataset["name"],
        "content_hash": observations_hash,
        "n_records": int(dataset["data"].size),
    }

    posterior_hash = _ic2_content_hash(posterior_ref)
    substrate = Substrate()
    registry_root = str(Path(posterior_ref).parent)
    ingested_ids = ingest_artifacts(substrate, registry_root)
    posterior_item_id = ingested_ids[0]
    manifest = json.loads((Path(posterior_ref) / "manifest.json").read_text())
    posterior_ref_record = {
        "item_id": posterior_item_id,
        "content_hash": posterior_hash,
        "item_hash": _item_hash(posterior_ref, manifest),
    }

    header = build_header(
        _DummyModel(),
        [query["distribution"]["mean"]],
        training={"query": "region_mass", "parent_content_hash": posterior_hash},
    )
    claim_record = {
        "value": query["distribution"]["mean"],
        "content_hash": header.dataset_hash,
        "model": "physics-tools/query_posterior@v1",
        "interval": interval,
        "prior_dominated": query["prior_dominated"],
    }

    value_per_kg = 0.02
    expected_value = query["distribution"]["mean"] * value_per_kg
    decision_payload = {
        "drill": bool(expected_value > 0 and not query["prior_dominated"]),
        "expected_value": expected_value,
        "risk": 0.18,
    }
    decision_hash = dataset_hash(
        [decision_payload["expected_value"], decision_payload["risk"], claim_record["content_hash"]]
    )
    decision_record = dict(
        decision_payload,
        content_hash=decision_hash,
        model="drill-advisor/v1",
        verdict={
            "passed": bool(decision_payload["drill"]),
            "score": 0.9,
            "reasons": ["expected value positive"],
            "kind": "physical",
        },
    )

    receipt = decision_receipt(
        dataset_ref=dataset_ref_record,
        posterior_ref=posterior_ref_record,
        claim=claim_record,
        decision=decision_record,
        substrate=substrate,
    )
    validate_trace_record(receipt)
    print(
        f"[5/5] decision_receipt: drill={decision_record['drill']}, expected_value={decision_record['expected_value']:.2f}, "
        f"provenance_hash={receipt['provenance']['content_hash'][:16]}..."
    )

    # bonus: IC-13 knowledge bundle export + persist/reload (M0b/M1a).
    bundle = build_knowledge_bundle(
        dataset_ref=dataset_ref,
        posterior_ref=posterior_ref,
        posterior_hash=posterior_hash,
        claim=claim_record,
        decision=decision_record,
    )
    with tempfile.TemporaryDirectory(prefix="mixle_e11_bundle_") as bundle_dir:
        bundle_path = str(Path(bundle_dir) / "bundle.json")
        persist_bundle(bundle, bundle_path)
        reloaded = reload_bundle(bundle_path)
    assert reloaded.items[0].artifact_ref == posterior_ref
    assert reloaded.items[0].content_hash == posterior_hash
    assert reloaded.items[1].payload == claim_record
    assert reloaded.items[2].payload == decision_record

    print("\n=== memo ===")
    print(
        f"target-horizon tonnage: {query['distribution']['mean']:.1f} kg (90% CI {interval['lo']:.1f} - {interval['hi']:.1f} kg)"
    )
    print(f"claim: {'ABSTAINED' if interpret['abstained'] else interpret['claim']}")
    print(
        f"decision: {'DRILL' if decision_record['drill'] else 'DO NOT DRILL'} (expected value {decision_record['expected_value']:.2f})"
    )
    print(f"provenance content_hash: {receipt['provenance']['content_hash']}")
    print(f"knowledge bundle: {len(reloaded.items)} items round-tripped, ids={[i.id for i in reloaded.items]}")


if __name__ == "__main__":
    main()
