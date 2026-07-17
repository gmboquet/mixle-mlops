"""Tests for mixle_mlops.control.pde_surrogate: the MP-N8 bridge from a trained mixle_pde operator
surrogate (MP-N5's ``LinearOperatorSurrogate``) into this platform's durable-job worker and deployment
registry.

Skips cleanly via ``pytest.importorskip`` when ``mixle_pde`` (or, in a bare ``--no-deps`` install such as
``control-ci.yml``'s, ``numpy``) is not installed -- the same cross-repo optional-dependency testing
precedent ``mixle-pde``'s own ``tests/multiphysics_pipeline_integration_test.py`` established
(``pytest.importorskip`` gating a test that genuinely needs a sibling package installed, rather than
mocking it away). Both guards run before any other import in this file so a missing dependency skips
collection cleanly instead of erroring.
"""

from __future__ import annotations

import pickle

import pytest

pytest.importorskip("mixle_pde")
np = pytest.importorskip("numpy")

from mixle_mlops.control import (  # noqa: E402
    PDE_OPERATOR_SURROGATE_CAPABILITY_ID,
    PDE_OPERATOR_SURROGATE_MEDIA_TYPE,
    PDE_OPERATOR_SURROGATE_SEMANTIC_TYPE,
    CapabilityRef,
    DeploymentRegistry,
    DurableLocalRunner,
    InvocationSpec,
    JobSpec,
    LocalArtifactStore,
    OperationalError,
    OwnerScope,
    PromotionPolicy,
    ResourceLimits,
    SurrogateTrainingSpec,
    WorkOutcome,
    check_registry_integrity,
    drain,
    land_pde_artifact,
    read_operator_surrogate_payload,
    register_pde_operator_surrogate,
    train_operator_surrogate_job,
)
from mixle_mlops.control.pde_surrogate import _synthetic_low_rank_snapshots  # noqa: E402
from mixle_pde.artifact_store import ArtifactStore  # noqa: E402
from mixle_pde.operator_surrogate import (  # noqa: E402
    LinearOperatorCalibrationReport,
    calibrate_linear_operator_surrogate,
    fit_linear_operator_surrogate,
)


def _owner() -> OwnerScope:
    return OwnerScope(organization_id="org-pde", project_id="surrogates")


def _training_spec(**overrides: object) -> SurrogateTrainingSpec:
    defaults: dict[str, object] = dict(
        n_dof_in=10,
        n_dof_out=8,
        true_rank=3,
        n_train=30,
        n_calibration=10,
        rank_in=3,
        rank_out=3,
        seed=0,
    )
    defaults.update(overrides)
    return SurrogateTrainingSpec(**defaults)  # type: ignore[arg-type]


def _job_spec(owner: OwnerScope, spec: SurrogateTrainingSpec, *, job_id: str = "surrogate-job-1") -> JobSpec:
    return JobSpec(
        id=job_id,
        owner=owner,
        invocation=InvocationSpec(
            capability=CapabilityRef(
                id=PDE_OPERATOR_SURROGATE_CAPABILITY_ID,
                version="1",
                input_schema="pde-operator-surrogate-training-spec/v1",
                output_schema="pde-operator-surrogate/v1",
            ),
            inputs=(),
            parameters=spec.as_dict(),
        ),
        resources=ResourceLimits(
            timeout_seconds=60,
            memory_bytes=50_000_000,
            cpu_seconds=60,
            output_bytes=50_000_000,
            event_count=100,
        ),
    )


def _policy(factory_issuer: str, harness_issuer: str) -> PromotionPolicy:
    return PromotionPolicy(
        trusted_factory_issuers=(factory_issuer,),
        trusted_harness_issuers=(harness_issuer,),
        required_suites=("pde_operator_surrogate_holdout_calibration",),
    )


def test_train_operator_surrogate_job_runs_as_a_real_durable_job(tmp_path) -> None:
    """The handler actually fits + calibrates a real ``LinearOperatorSurrogate`` via the pre-existing
    durable job machinery (claim/lease/checkpoint/complete) -- not a bespoke standalone script, and not a
    mock. The unpickled surrogate genuinely predicts."""
    owner = _owner()
    runner = DurableLocalRunner(tmp_path / "runner")
    spec = _training_spec()
    runner.submit(_job_spec(owner, spec))

    reports = drain(runner, "worker-1", train_operator_surrogate_job)

    assert len(reports) == 1
    assert reports[0].outcome is WorkOutcome.SUCCEEDED
    record = runner.get("surrogate-job-1", owner)
    assert len(record.checkpoints) == 1  # the fitted-but-not-yet-calibrated durability checkpoint
    assert len(record.results) == 1

    result_artifact = record.results[0]
    assert result_artifact.media_type == PDE_OPERATOR_SURROGATE_MEDIA_TYPE
    assert result_artifact.semantic_type == PDE_OPERATOR_SURROGATE_SEMANTIC_TYPE

    payload = read_operator_surrogate_payload(runner.artifacts.get(owner, result_artifact))
    surrogate = payload["surrogate"]
    calibration = payload["calibration"]
    assert isinstance(calibration, LinearOperatorCalibrationReport)
    assert calibration.imprecise is False
    assert calibration.mean_relative_l2_error < 1e-4

    prediction = surrogate.predict(np.zeros(spec.n_dof_in))
    assert prediction.field.shape == (spec.n_dof_out,)


def test_register_trained_surrogate_is_visible_to_integrity_check(tmp_path) -> None:
    """A durable job's result becomes a registered ModelCandidate that check_registry_integrity sees --
    the concrete answer to "no mixle-mlops registry integration ... found anywhere" (MP-N8)."""
    owner = _owner()
    runner = DurableLocalRunner(tmp_path / "runner")
    runner.submit(_job_spec(owner, _training_spec()))
    reports = drain(runner, "worker-1", train_operator_surrogate_job)
    assert reports[0].outcome is WorkOutcome.SUCCEEDED
    result_artifact = runner.get("surrogate-job-1", owner).results[0]

    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, factory, harness = register_pde_operator_surrogate(
        registry=registry,
        artifacts=runner.artifacts,
        owner=owner,
        result_artifact=result_artifact,
        model_id="pde-operator-surrogate-demo",
        version="v1",
        candidate_id="trained-1",
        factory_issuer="mixle_mlops.control.worker",
        harness_issuer="mixle_pde.operator_surrogate",
    )

    assert factory.passed is True
    assert harness.passed is True  # the good-case fit really did calibrate as precise
    assert registry.candidate("trained-1") == candidate

    report = check_registry_integrity(registry, artifacts=runner.artifacts)
    assert report.clean
    assert report.checked_candidates == 1
    assert report.checked_artifacts == 1


def test_land_pde_artifact_bridges_pde_store_lineage_and_registers(tmp_path) -> None:
    """A surrogate trained entirely outside any mlops job -- via pde's own pre-existing, real
    put/get/lineage ArtifactStore (MP-K1) -- becomes registrable too, carrying its pde-side lineage into
    the candidate's metadata rather than this bridge rebuilding storage."""
    owner = _owner()
    pde_store = ArtifactStore(tmp_path / "pde-store")
    spec = _training_spec(seed=11)
    inputs_train, outputs_train, inputs_calibration, outputs_calibration = _synthetic_low_rank_snapshots(spec)

    # Real lineage: the training snapshot matrix is a genuine parent artifact of the fitted surrogate.
    snapshot_digest = pde_store.put(
        pickle.dumps({"inputs_train": inputs_train, "outputs_train": outputs_train}),
        metadata={"kind": "training_snapshots", "n_train": spec.n_train},
    )
    surrogate = fit_linear_operator_surrogate(
        inputs_train, outputs_train, rank_in=spec.rank_in, rank_out=spec.rank_out, ridge=spec.ridge
    )
    calibration = calibrate_linear_operator_surrogate(
        surrogate, inputs_calibration, outputs_calibration, alpha=spec.alpha
    )
    payload = {"surrogate": surrogate, "calibration": calibration, "training_spec": spec.as_dict()}
    surrogate_digest = pde_store.put(
        pickle.dumps(payload),
        metadata={"kind": "operator_surrogate", "model_id": "standalone-surrogate"},
        parents=(snapshot_digest,),
    )

    artifacts = LocalArtifactStore(tmp_path / "mlops-artifacts")
    landed = land_pde_artifact(pde_store=pde_store, pde_digest=surrogate_digest, artifacts=artifacts, owner=owner)
    # Same bytes, same hash, across two independently-implemented content-addressed stores.
    assert landed.sha256 == surrogate_digest

    registry = DeploymentRegistry(tmp_path / "registry")
    candidate, _factory, _harness = register_pde_operator_surrogate(
        registry=registry,
        artifacts=artifacts,
        owner=owner,
        result_artifact=landed,
        model_id="pde-operator-surrogate-standalone",
        version="v1",
        candidate_id="standalone-1",
        factory_issuer="mixle_pde.artifact_store",
        harness_issuer="mixle_pde.operator_surrogate",
        lineage={
            "pde_artifact_digest": surrogate_digest,
            "pde_artifact_parents": pde_store.parents_of(surrogate_digest),
            "pde_artifact_metadata": dict(pde_store.metadata(surrogate_digest)),
        },
    )

    assert candidate.metadata["pde_lineage"]["pde_artifact_parents"] == (snapshot_digest,)
    assert candidate.metadata["pde_lineage"]["pde_artifact_metadata"]["model_id"] == "standalone-surrogate"
    assert pde_store.children_of(snapshot_digest) == (surrogate_digest,)  # lineage query works both ways

    report = check_registry_integrity(registry, artifacts=artifacts)
    assert report.clean
    assert report.checked_candidates == 1
    assert report.checked_artifacts == 1


def test_promote_and_resolve_trained_operator_surrogate(tmp_path) -> None:
    """The full MP-N8 lifecycle in one pass: train via the durable worker, register, integrity-check,
    evidence-gate a promotion, and resolve the promoted alias -- proving "registrable in the registry" and
    "executable as a durable job" work together, not as two disconnected halves."""
    owner = _owner()
    runner = DurableLocalRunner(tmp_path / "runner")
    runner.submit(_job_spec(owner, _training_spec()))
    reports = drain(runner, "worker-1", train_operator_surrogate_job)
    assert reports[0].outcome is WorkOutcome.SUCCEEDED
    result_artifact = runner.get("surrogate-job-1", owner).results[0]

    registry = DeploymentRegistry(tmp_path / "registry")
    factory_issuer, harness_issuer = "mixle_mlops.control.worker", "mixle_pde.operator_surrogate"
    candidate, factory, harness = register_pde_operator_surrogate(
        registry=registry,
        artifacts=runner.artifacts,
        owner=owner,
        result_artifact=result_artifact,
        model_id="pde-operator-surrogate-demo",
        version="v1",
        candidate_id="trained-1",
        factory_issuer=factory_issuer,
        harness_issuer=harness_issuer,
    )

    assert check_registry_integrity(registry, artifacts=runner.artifacts).clean

    policy = _policy(factory_issuer, harness_issuer)
    registry.promote(candidate.id, "stage", (factory, harness), policy, actor="pde-surrogate-bridge-test")

    resolved = registry.resolve("stage")
    assert resolved == candidate

    post_promotion_report = check_registry_integrity(registry, artifacts=runner.artifacts)
    assert post_promotion_report.clean
    assert post_promotion_report.checked_receipts == 1


def test_imprecise_calibration_blocks_promotion(tmp_path) -> None:
    """The surrogate's own honesty gate genuinely governs promotion: registration alone never claims
    trustworthiness, and an ``imprecise=True`` calibration produces a failing HARNESS receipt that
    ``control.registry``'s evidence-gated ``promote`` refuses, exactly as it would for any other
    candidate."""
    owner = _owner()
    spec = _training_spec(seed=3)
    inputs_train, outputs_train, _inputs_calibration, _outputs_calibration = _synthetic_low_rank_snapshots(spec)
    surrogate = fit_linear_operator_surrogate(
        inputs_train, outputs_train, rank_in=spec.rank_in, rank_out=spec.rank_out, ridge=spec.ridge
    )
    bad_calibration = LinearOperatorCalibrationReport(
        n=10,
        alpha=0.1,
        mean_relative_l2_error=1.2,
        max_relative_l2_error=1.5,
        qhat_relative_l2_error=1.5,
        baseline_relative_l2_error=1.0,
        imprecise=True,
        ood_fraction=0.4,
    )
    payload = {"surrogate": surrogate, "calibration": bad_calibration, "training_spec": spec.as_dict()}

    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    result_artifact = artifacts.put(
        owner,
        pickle.dumps(payload),
        media_type=PDE_OPERATOR_SURROGATE_MEDIA_TYPE,
        semantic_type=PDE_OPERATOR_SURROGATE_SEMANTIC_TYPE,
    )
    registry = DeploymentRegistry(tmp_path / "registry")
    factory_issuer, harness_issuer = "mixle_mlops.control.worker", "mixle_pde.operator_surrogate"
    candidate, factory, harness = register_pde_operator_surrogate(
        registry=registry,
        artifacts=artifacts,
        owner=owner,
        result_artifact=result_artifact,
        model_id="pde-operator-surrogate-demo",
        version="v1",
        candidate_id="imprecise-1",
        factory_issuer=factory_issuer,
        harness_issuer=harness_issuer,
    )
    assert harness.passed is False

    policy = _policy(factory_issuer, harness_issuer)
    with pytest.raises(OperationalError):
        registry.promote(candidate.id, "stage", (factory, harness), policy, actor="tester")


def test_load_operator_surrogate_honors_a_monkeypatched_fake_even_after_a_real_import_cached_the_attribute(
    monkeypatch,
) -> None:
    """Regression test for the mixle_pde submodule import-caching bug PR #61's follow-up
    (``fix(mcp): mixle_pde tool loaders stop losing monkeypatched fakes to a cached real import``) fixed for
    ``mcp/physics_tools.py``/``mcp/sim_tools.py``: a real ``from mixle_pde import operator_surrogate``
    anywhere earlier in the process -- e.g. this very test file's own module-level import above -- sets an
    ``operator_surrogate`` attribute on the ``mixle_pde`` package module as a side effect of CPython's import
    machinery (``_handle_fromlist``). A loader written as ``from mixle_pde import operator_surrogate`` would
    then resolve via that cached attribute and silently ignore a later
    ``monkeypatch.setitem(sys.modules, "mixle_pde.operator_surrogate", fake)``. ``_load_operator_surrogate``
    uses ``importlib.import_module`` specifically to sidestep this; this forces the poisoning inline (rather
    than relying on incidental test-execution order) and asserts it still picks up the fake regardless.
    """
    import sys
    import types

    import mixle_pde

    real_operator_surrogate = mixle_pde.operator_surrogate  # already cached by this file's own module-level import
    assert mixle_pde.operator_surrogate is real_operator_surrogate  # sanity: the poisoning precondition holds

    fake = types.ModuleType("mixle_pde.operator_surrogate")
    monkeypatch.setitem(sys.modules, "mixle_pde.operator_surrogate", fake)

    from mixle_mlops.control.pde_surrogate import _load_operator_surrogate

    loaded = _load_operator_surrogate()
    assert loaded is fake, "a real import earlier in the process shadowed the monkeypatched fake"
    assert loaded is not real_operator_surrogate
