"""MP-N8 bridge: mixle-pde's surrogate lifecycle into this platform's durable-job/registry control plane.

The M2 reconciliation ledger (mixle-pde's ``docs/reconciliation/mp-task-ledger.md``, row ``MP-N8``) records:
"Registry, promotion, monitoring, retraining, rollback | not-started | -- | No mixle-mlops registry
integration, promotion gate, or drift-triggered retraining found anywhere." A direct ``git grep`` over PR
#58 (``control.integrity``) and PR #59 (``control.worker``) confirms neither ever mentions "pde" or
"surrogate": a trained pde surrogate (``mixle_pde.surrogate.Surrogate``, MP-N1; ``mixle_pde
.operator_surrogate.LinearOperatorSurrogate``, MP-N5) had no path into either module.

This module is that path, for the operator surrogate (MP-N5) specifically -- a frozen dataclass of plain
numpy arrays and no torch/closures, so it pickles deterministically without the extra machinery a neural
student (MP-N1's ``Surrogate``, which carries live torch modules and an arbitrary ``teacher`` callable)
would need; that remains later, separate work. Three real seams, not a mock registration:

1. :func:`train_operator_surrogate_job` is a :data:`~.worker.WorkerHandler`: given to ``control.worker
   .run_once``/``drain``/``run_forever``, it actually fits and calibrates a real ``LinearOperatorSurrogate``
   from the job's own ``InvocationSpec.parameters`` -- training a surrogate now runs on the pre-existing,
   unmodified durable-job machinery (claim/lease/heartbeat/checkpoint/retry/recovery), not a bespoke script.
2. :func:`land_pde_artifact` bridges a surrogate already stored in pde's own, pre-existing
   ``mixle_pde.artifact_store.ArtifactStore`` (put/get/lineage -- MP-K1) into this platform's owner-scoped
   ``control.artifacts.LocalArtifactStore``, for a surrogate trained entirely outside any mlops job.
3. :func:`register_pde_operator_surrogate` turns either path's landed artifact into an immutable
   ``control.contracts.ModelCandidate`` with real, calibration-grounded evidence and registers it in a
   ``control.registry.DeploymentRegistry`` -- so ``control.integrity.check_registry_integrity`` can see it,
   and ``registry.promote``/``rollback`` govern it, exactly like any other candidate.

Following the lazy-import precedent already established for this exact cross-repo boundary
(``mcp/physics_tools.py``, ``mcp/sim_tools.py``, ``monitoring.py``'s ``exceedance_probability``,
``models/field_posterior.py``): ``mixle_pde`` is an optional, lazily-imported dependency. A deployment
without it installed still imports this module fine; only calling :func:`train_operator_surrogate_job` or
decoding a stored payload needs it, and both raise :class:`PdeSurrogateUnavailable` with an actionable
message instead of a bare ``ImportError``. ``numpy`` is imported lazily here too, deliberately unlike
``monitoring.py``/``production_serving.py`` (which import it unconditionally): unlike those, ``control/``
today is stdlib-only at module level even though the wider ``mixle_mlops`` package already depends on it
transitively, and ``control-ci.yml`` installs this package with ``--no-deps`` -- so a top-level ``import
numpy`` here would make merely ``import mixle_mlops.control`` fail in that lane. Deferred into
``_synthetic_low_rank_snapshots``, the one function that needs it, which ``train_operator_surrogate_job``
only ever reaches after ``_load_operator_surrogate`` has already confirmed ``mixle_pde`` (and therefore
its own hard ``numpy`` dependency) is installed.
"""

from __future__ import annotations

import importlib
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .artifacts import LocalArtifactStore
from .contracts import ArtifactRef, EvidenceKind, EvidenceReceipt, ModelCandidate, OperationalError, OwnerScope
from .registry import DeploymentRegistry
from .runner import JobRecord
from .worker import HandlerFailure, HandlerResult, WorkerContext

__all__ = [
    "PDE_OPERATOR_SURROGATE_CAPABILITY_ID",
    "PDE_OPERATOR_SURROGATE_MEDIA_TYPE",
    "PDE_OPERATOR_SURROGATE_SEMANTIC_TYPE",
    "PdeSurrogateUnavailable",
    "SurrogateTrainingSpec",
    "land_pde_artifact",
    "read_operator_surrogate_payload",
    "register_pde_operator_surrogate",
    "train_operator_surrogate_job",
]

_PAYLOAD_SCHEMA_VERSION = "1.0.0"

#: ``CapabilityRef.id`` a caller should use when building the ``JobSpec`` this module's handler executes.
PDE_OPERATOR_SURROGATE_CAPABILITY_ID = "mixle_pde.operator_surrogate.fit_linear_operator_surrogate"
#: ``media_type`` every payload this module produces or reads is stored under.
PDE_OPERATOR_SURROGATE_MEDIA_TYPE = "application/x-mixle-pde-operator-surrogate-pickle"
#: ``semantic_type`` every payload this module produces is stored under.
PDE_OPERATOR_SURROGATE_SEMANTIC_TYPE = "pde_operator_surrogate"


class PdeSurrogateUnavailable(RuntimeError):
    """Raised when ``mixle_pde`` isn't installed but real pde surrogate code was just about to run."""


def _load_operator_surrogate() -> Any:
    """Import ``mixle_pde.operator_surrogate`` (MP-N5); raise :class:`PdeSurrogateUnavailable` with an
    actionable message if the pde package isn't installed. Not cached at module scope so a later install
    is picked up on the very next call, matching ``mcp/physics_tools.py``'s ``_load_pde_tools``.

    Resolves via ``importlib.import_module`` rather than ``from mixle_pde import operator_surrogate``,
    matching the fix landed in ``fix(mcp): mixle_pde tool loaders stop losing monkeypatched fakes to a
    cached real import``: a plain ``from mixle_pde import X`` sets a real attribute on the ``mixle_pde``
    package object the first time any process-wide caller imports it for real (CPython's
    ``_handle_fromlist``), so a later ``monkeypatch.setitem(sys.modules, "mixle_pde.operator_surrogate",
    fake)`` in a test would be silently ignored in favor of that cached real attribute.
    ``importlib.import_module`` always resolves the fully-qualified name through ``sys.modules`` directly,
    honoring a monkeypatched fake regardless of what ``mixle_pde``'s own cached attributes say.
    """
    try:
        operator_surrogate = importlib.import_module("mixle_pde.operator_surrogate")
    except ImportError as exc:  # pragma: no cover - exercised only when mixle_pde is absent
        raise PdeSurrogateUnavailable(
            "training a pde operator surrogate requires the mixle_pde package; install mixle_pde "
            "(see the mixle-pde repo) to run train_operator_surrogate_job"
        ) from exc
    return operator_surrogate


@dataclass(frozen=True)
class SurrogateTrainingSpec:
    """A durable job's ``InvocationSpec.parameters`` payload for training + calibrating one
    ``mixle_pde.operator_surrogate.LinearOperatorSurrogate`` (MP-N5) from deterministic synthetic paired
    field snapshots.

    Deliberately not a general "bring your own snapshot matrix" contract: a job's parameters must be
    plain JSON-shaped data (``control.contracts.JobSpec``/``InvocationSpec``), and full snapshot arrays are
    neither small nor JSON-shaped. What is captured here is exactly what
    :func:`_synthetic_low_rank_snapshots` needs to regenerate the same paired input/output snapshots
    deterministically from ``seed`` -- mirroring ``tests/operator_surrogate_test.py``'s own hand-built
    exact-recovery fixture in mixle-pde, a real numerical fitting exercise with a known ground truth, not a
    mocked stand-in. A full production data-loading/DOE contract is separate, unclaimed work (MP-N2 is its
    own not-started ledger row).
    """

    n_dof_in: int
    n_dof_out: int
    true_rank: int
    n_train: int
    n_calibration: int
    seed: int = 0
    rank_in: int | None = None
    rank_out: int | None = None
    energy_threshold_in: float | None = None
    energy_threshold_out: float | None = None
    ridge: float = 1e-6
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if min(self.n_dof_in, self.n_dof_out, self.true_rank, self.n_train, self.n_calibration) < 1:
            raise ValueError("n_dof_in, n_dof_out, true_rank, n_train, and n_calibration must all be positive")
        if self.true_rank > min(self.n_dof_in, self.n_dof_out):
            raise ValueError("true_rank cannot exceed min(n_dof_in, n_dof_out)")
        if (self.rank_in is None) == (self.energy_threshold_in is None):
            raise ValueError("exactly one of rank_in or energy_threshold_in must be set")
        if (self.rank_out is None) == (self.energy_threshold_out is None):
            raise ValueError("exactly one of rank_out or energy_threshold_out must be set")

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_dof_in": self.n_dof_in,
            "n_dof_out": self.n_dof_out,
            "true_rank": self.true_rank,
            "n_train": self.n_train,
            "n_calibration": self.n_calibration,
            "seed": self.seed,
            "rank_in": self.rank_in,
            "rank_out": self.rank_out,
            "energy_threshold_in": self.energy_threshold_in,
            "energy_threshold_out": self.energy_threshold_out,
            "ridge": self.ridge,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SurrogateTrainingSpec:
        return cls(
            n_dof_in=int(value["n_dof_in"]),
            n_dof_out=int(value["n_dof_out"]),
            true_rank=int(value["true_rank"]),
            n_train=int(value["n_train"]),
            n_calibration=int(value["n_calibration"]),
            seed=int(value.get("seed", 0)),
            rank_in=value.get("rank_in"),
            rank_out=value.get("rank_out"),
            energy_threshold_in=value.get("energy_threshold_in"),
            energy_threshold_out=value.get("energy_threshold_out"),
            ridge=float(value.get("ridge", 1e-6)),
            alpha=float(value.get("alpha", 0.1)),
        )


def _synthetic_low_rank_snapshots(
    spec: SurrogateTrainingSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic paired input/output snapshots from a known random low-rank linear map -- the same
    hand-built exact-recovery construction ``tests/operator_surrogate_test.py`` uses in mixle-pde. Returns
    ``(inputs_train, outputs_train, inputs_calibration, outputs_calibration)``.

    Imports ``numpy`` locally (see module docstring): by the time ``train_operator_surrogate_job`` calls
    this, ``_load_operator_surrogate`` has already confirmed ``mixle_pde`` -- and therefore ``numpy``,
    ``mixle_pde``'s own hard dependency -- is installed.
    """
    import numpy as np

    rng = np.random.default_rng(spec.seed)
    basis_true_in, _ = np.linalg.qr(rng.standard_normal((spec.n_dof_in, spec.true_rank)))
    basis_true_out, _ = np.linalg.qr(rng.standard_normal((spec.n_dof_out, spec.true_rank)))
    core = rng.standard_normal((spec.true_rank, spec.true_rank))

    def true_operator(u_column: np.ndarray) -> np.ndarray:
        return basis_true_out @ (core @ (basis_true_in.T @ u_column))

    def _draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        latent = rng.standard_normal((spec.true_rank, n))
        inputs = basis_true_in @ latent
        outputs = np.stack([true_operator(inputs[:, i]) for i in range(n)], axis=1)
        return inputs, outputs

    inputs_train, outputs_train = _draw(spec.n_train)
    inputs_calibration, outputs_calibration = _draw(spec.n_calibration)
    return inputs_train, outputs_train, inputs_calibration, outputs_calibration


def train_operator_surrogate_job(record: JobRecord, context: WorkerContext) -> HandlerResult:
    """A :data:`~.worker.WorkerHandler` that fits and calibrates a real ``LinearOperatorSurrogate`` from
    ``record``'s own ``InvocationSpec.parameters`` (a :class:`SurrogateTrainingSpec`).

    Everything about the job itself -- claim, lease, heartbeat, checkpoint, retry, completion -- is the
    pre-existing, unmodified ``control.runner``/``control.worker`` machinery; this is the first domain
    handler that plugs a real pde training routine into it, closing MP-N8's "no path into the durable-job
    ... system" gap. Checkpoints the fitted-but-not-yet-calibrated surrogate before calibration, a genuine
    mid-job durability point exercising ``WorkerContext.checkpoint`` the way ``docs/operations.md``
    describes a worker doing. A ``ValueError`` from the pde fit/calibration call becomes a typed
    :class:`~.worker.HandlerFailure` (a specific, retry-policy-matchable code) rather than an opaque
    ``handler_exception``.
    """
    operator_surrogate = _load_operator_surrogate()
    spec = SurrogateTrainingSpec.from_dict(record.spec.invocation.parameters)
    inputs_train, outputs_train, inputs_calibration, outputs_calibration = _synthetic_low_rank_snapshots(spec)

    context.progress(stage="fitting", n_train=spec.n_train)
    try:
        surrogate = operator_surrogate.fit_linear_operator_surrogate(
            inputs_train,
            outputs_train,
            rank_in=spec.rank_in,
            energy_threshold_in=spec.energy_threshold_in,
            rank_out=spec.rank_out,
            energy_threshold_out=spec.energy_threshold_out,
            ridge=spec.ridge,
        )
    except ValueError as exc:
        raise HandlerFailure(code="pde_operator_surrogate_fit_failed", detail=str(exc)) from exc

    context.checkpoint(
        pickle.dumps({"schema_version": _PAYLOAD_SCHEMA_VERSION, "stage": "fitted", "surrogate": surrogate}),
        media_type=PDE_OPERATOR_SURROGATE_MEDIA_TYPE,
    )

    context.progress(stage="calibrating", n_calibration=spec.n_calibration)
    try:
        calibration = operator_surrogate.calibrate_linear_operator_surrogate(
            surrogate, inputs_calibration, outputs_calibration, alpha=spec.alpha
        )
    except ValueError as exc:
        raise HandlerFailure(code="pde_operator_surrogate_calibration_failed", detail=str(exc)) from exc

    payload = {
        "schema_version": _PAYLOAD_SCHEMA_VERSION,
        "surrogate": surrogate,
        "calibration": calibration,
        "training_spec": spec.as_dict(),
    }
    return HandlerResult(
        data=pickle.dumps(payload),
        media_type=PDE_OPERATOR_SURROGATE_MEDIA_TYPE,
        semantic_type=PDE_OPERATOR_SURROGATE_SEMANTIC_TYPE,
    )


def read_operator_surrogate_payload(data: bytes) -> dict[str, Any]:
    """Unpickle one ``{"surrogate": LinearOperatorSurrogate, "calibration":
    LinearOperatorCalibrationReport, "training_spec": dict | None, ...}`` payload, as produced by
    :func:`train_operator_surrogate_job` or hand-built the same way for a surrogate trained outside any
    job (see :func:`land_pde_artifact`).

    Trust boundary, stated plainly: this is ``pickle.loads`` over bytes that came from this platform's own
    artifact stores (``control.artifacts.LocalArtifactStore`` or pde's own ``ArtifactStore``), both of
    which already verify content by sha256 digest before returning it -- the same trust boundary
    ``control.runner`` already assumes for every stored result and checkpoint (a durable single-node
    reference, not a multi-tenant sandbox against untrusted bytes). It is not a general-purpose
    untrusted-deserialization boundary and must not be pointed at data from outside this control plane.
    """
    try:
        return pickle.loads(data)
    except ModuleNotFoundError as exc:
        raise PdeSurrogateUnavailable(
            "decoding a stored pde operator surrogate requires the mixle_pde package; install mixle_pde "
            "(see the mixle-pde repo) to read a train_operator_surrogate_job result or land_pde_artifact "
            "payload"
        ) from exc


def land_pde_artifact(
    *,
    pde_store: Any,
    pde_digest: str,
    artifacts: LocalArtifactStore,
    owner: OwnerScope,
    media_type: str = PDE_OPERATOR_SURROGATE_MEDIA_TYPE,
    semantic_type: str = PDE_OPERATOR_SURROGATE_SEMANTIC_TYPE,
) -> ArtifactRef:
    """Re-land one artifact already stored in a pde-side ``mixle_pde.artifact_store.ArtifactStore``
    (``pde_store``, MP-K1's real put/get/lineage store) into this platform's owner-scoped
    ``control.artifacts.LocalArtifactStore`` (``artifacts``), so a surrogate trained entirely outside any
    mlops job becomes registrable the same way a durable job's own result is (see
    :func:`register_pde_operator_surrogate`).

    Bridges storage, does not replace it: ``pde_store.get`` is pde's own real, digest-verified read;
    ``artifacts.put`` is this platform's own real, digest-verified write. Both stores hash the identical
    convention (sha256 hex digest of raw content bytes -- ``mixle_pde.artifact_store.digest_of`` and
    ``LocalArtifactStore.put`` agree exactly), so the returned ``ArtifactRef.sha256`` is checked equal to
    ``pde_digest``: the same bytes, addressed identically by two independently-implemented
    content-addressed stores. Raises :class:`~.contracts.OperationalError` if they ever disagreed.
    """
    content = pde_store.get(pde_digest)
    artifact = artifacts.put(owner, content, media_type=media_type, semantic_type=semantic_type)
    if artifact.sha256 != pde_digest:
        raise OperationalError(
            f"pde artifact {pde_digest} landed as {artifact.sha256} in the platform artifact store; "
            "the two content-addressed stores disagree on identical bytes"
        )
    return artifact


def register_pde_operator_surrogate(
    *,
    registry: DeploymentRegistry,
    artifacts: LocalArtifactStore,
    owner: OwnerScope,
    result_artifact: ArtifactRef,
    model_id: str,
    version: str,
    candidate_id: str,
    factory_issuer: str,
    harness_issuer: str,
    lineage: Mapping[str, Any] | None = None,
) -> tuple[ModelCandidate, EvidenceReceipt, EvidenceReceipt]:
    """Register one trained pde operator surrogate as an immutable ``ModelCandidate`` in ``registry`` --
    the MP-N8 bridge closing the ledger's "no mixle-mlops registry integration ... found anywhere" gap.

    ``result_artifact`` is either a durable job's own completed result (``JobRecord.results[-1]``, after
    :func:`train_operator_surrogate_job` runs via ``control.worker``) or the return value of
    :func:`land_pde_artifact` for a surrogate trained outside any job -- either way, the artifact must
    already be present in ``artifacts`` (re-verified here via ``artifacts.get``, never trusted blindly).

    Evidence is real, not fabricated: the FACTORY receipt attests that the training procedure itself
    produced this artifact (this function is only ever reached for a job that already reached
    ``WorkOutcome.SUCCEEDED``, or an artifact that already landed successfully); the HARNESS receipt's
    ``passed`` is exactly ``not calibration.imprecise`` -- the surrogate's own held-out split-conformal
    calibration gate (``mixle_pde.operator_surrogate.calibrate_linear_operator_surrogate``), the same
    honesty gate the module itself reports, never re-derived or loosened here. Whether either issuer is
    actually *trusted* is a decision for the caller's own ``control.contracts.PromotionPolicy`` at
    ``registry.promote`` time -- registration alone asserts nothing about trust.

    ``lineage``, when given (typically ``{"pde_artifact_digest": ..., "pde_artifact_parents": (...),
    "pde_artifact_metadata": {...}}`` from :func:`land_pde_artifact`'s source store), is copied verbatim
    into the candidate's ``metadata`` under the ``pde_lineage`` key.
    """
    data = artifacts.get(owner, result_artifact)
    payload = read_operator_surrogate_payload(data)
    calibration = payload["calibration"]
    imprecise = bool(calibration.imprecise)

    metadata: dict[str, Any] = {
        "pde_module": "mixle_pde.operator_surrogate",
        "training_spec": payload.get("training_spec"),
        "calibration": {
            "n": calibration.n,
            "alpha": calibration.alpha,
            "mean_relative_l2_error": calibration.mean_relative_l2_error,
            "max_relative_l2_error": calibration.max_relative_l2_error,
            "qhat_relative_l2_error": calibration.qhat_relative_l2_error,
            "baseline_relative_l2_error": calibration.baseline_relative_l2_error,
            "imprecise": calibration.imprecise,
            "ood_fraction": calibration.ood_fraction,
        },
    }
    if lineage:
        metadata["pde_lineage"] = dict(lineage)

    factory_receipt = EvidenceReceipt(
        id=f"{candidate_id}-factory",
        kind=EvidenceKind.FACTORY,
        issuer=factory_issuer,
        subject_sha256=result_artifact.sha256,
        passed=True,
        suites=("pde_operator_surrogate_training",),
    )
    harness_receipt = EvidenceReceipt(
        id=f"{candidate_id}-harness",
        kind=EvidenceKind.HARNESS,
        issuer=harness_issuer,
        subject_sha256=result_artifact.sha256,
        passed=not imprecise,
        suites=("pde_operator_surrogate_holdout_calibration",),
    )
    candidate = ModelCandidate(
        id=candidate_id,
        model_id=model_id,
        version=version,
        artifact=result_artifact,
        factory_receipt_id=factory_receipt.id,
        harness_receipt_ids=(harness_receipt.id,),
        metadata=metadata,
    )
    registry.register(candidate)
    return candidate, factory_receipt, harness_receipt
