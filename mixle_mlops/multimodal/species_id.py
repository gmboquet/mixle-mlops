"""N3 -- cross-modal species identification: the choke point between a camera-trap/bioacoustic/eDNA
detection claim and workstream N's species-distribution model (N1). Nothing becomes a typed IC-12
`SpeciesObservation` occurrence without first clearing an IC-6 `Verifier` -- exactly I7's
`ingest_extraction` pattern (write nothing at all on rejection, no half-typed observation, just a
`Verdict` and its reasons a caller can log), applied here to a species-identification claim instead of a
map/log/table extraction.

`identify_species` routes a payload already bound to its `DomainModelAdapter` (IC-7) through `call`,
builds the IC-6 verifier context, and applies an additional score-floor abstain rule on top of the
verdict: a detection is only ever accepted when the verifier passes it AND its score clears
``abstain_below``. An accepted detection's `SpeciesObservation` is the occurrence row N1's `fit_sdm`
ingests; a rejected/abstained one writes nothing, with its reasons and content hash preserved in
`SpeciesIDResult.provenance` for audit.

IC-12's `SpeciesObservation` lives in `mixle.analysis.sdm` (N1, core mixle). N1 had not landed in this
checkout as of this PR (confirmed: no ``mixle/analysis/sdm.py`` on the shared core-mixle tree this task's
PYTHONPATH resolves against), so this module imports it lazily and falls back to a field-for-field
identical IC-12 shim when the import fails -- the same "build the contract's own shape locally until the
real module lands" pattern I7 already applies to `mixle_pde.observations.Observation`/`mixle_knowledge`.
Once N1 merges, the real class is picked up automatically with no code change here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..models.domain_adapter import DomainModelAdapter
from ..verification.base import Verifier, Verdict

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

try:  # IC-12 (N1, mixle/analysis/sdm.py) -- may not have landed in this checkout yet
    from mixle.analysis.sdm import SpeciesObservation
except ImportError:  # pragma: no cover - exercised only when N1 hasn't landed locally

    @dataclass
    class SpeciesObservation:  # type: ignore[no-redef]
        """IC-12 shim: field-for-field identical to the frozen `mixle.analysis.sdm.SpeciesObservation`
        contract (notes/exec/contracts.md, IC-12). Used only until N1 lands the real module."""

        species_id: str
        detection: bool
        location: "np.ndarray"
        crs: str | None = None
        covariates: dict[str, Any] = field(default_factory=dict)
        modality: str = "occurrence"
        provenance: dict[str, Any] = field(default_factory=dict)


# Payload keys that describe the detection's context (site/geometry/verification inputs) rather than the
# model's own classification inputs -- excluded before `inputs` is handed to `adapter.call`.
_CONTEXT_ONLY_KEYS = {"location", "crs", "modality", "plausible_species", "source_ref"}


@dataclass
class SpeciesIDResult:
    """The outcome of one cross-modal species-identification attempt: whether it was accepted, the
    resulting IC-12 occurrence (``None`` on abstain/reject), the IC-6 `Verdict` that gated it, and a
    provenance dict (model id/version, content hash, verifier kind, score, reasons) preserved either way
    so an abstained detection is still auditable even though it wrote no occurrence."""

    accepted: bool
    observation: SpeciesObservation | None
    verdict: Verdict
    provenance: dict[str, Any] = field(default_factory=dict)


async def identify_species(
    payload: dict[str, Any],
    *,
    adapter: DomainModelAdapter,
    verifier: Verifier,
    abstain_below: float = 0.6,
) -> SpeciesIDResult:
    """Run one cross-modal detection ``payload`` (``{modality, blob_ref, location, crs, ...}``) through
    ``adapter`` and gate it behind ``verifier``.

    1. ``await adapter.call(inputs)`` returns an IC-7 `ProvenancedResult{value={species_id, score},
       model_id, version, content_hash}` -- every detection carries model attribution + a content hash.
    2. The IC-6 verifier ``context`` names the site's plausible-range species list, the claimed score, and
       the modality; ``verifier.verify(claim, context) -> Verdict`` is E10's gate, applied exactly the way
       I7 applies it to an extraction claim.
    3. Abstain rule: if ``not verdict.passed`` OR ``score < abstain_below``, nothing is written --
       ``accepted=False``, ``observation=None`` -- and the reasons + content hash are preserved in
       ``provenance`` for logging.
    4. On accept, an IC-12 `SpeciesObservation` is constructed and returned as the occurrence row N1's
       `fit_sdm` ingests, provenanced with its source ref, content hash, model id, verifier kind and score.
    """
    modality = str(payload.get("modality", "occurrence"))
    inputs = {k: v for k, v in payload.items() if k not in _CONTEXT_ONLY_KEYS}

    result = await adapter.call(inputs)
    value = result.value or {}
    species_id = str(value["species_id"])
    score = float(value["score"])

    claim = {"species_id": species_id, "score": score, "modality": modality}
    context = {
        "location": payload.get("location"),
        "crs": payload.get("crs"),
        "modality": modality,
        "plausible_species": payload.get("plausible_species"),
    }
    verdict = verifier.verify(claim, context)

    reasons = list(verdict.reasons)
    below_floor = score < abstain_below
    if below_floor:
        reasons.append(f"score {score:.3f} is below the abstain_below floor {abstain_below:.3f}")

    source_ref = payload.get("source_ref") or {"blob_ref": payload.get("blob_ref"), "modality": modality}
    base_provenance = {
        "source_ref": source_ref,
        "content_hash": result.content_hash,
        "model_id": result.model_id,
        "model_version": result.version,
        "verifier_kind": verdict.kind,
        "score": score,
        "species_id": species_id,
        "reasons": reasons,
    }

    accepted = bool(verdict.passed) and not below_floor
    if not accepted:
        return SpeciesIDResult(accepted=False, observation=None, verdict=verdict, provenance=base_provenance)

    observation = SpeciesObservation(
        species_id=species_id,
        detection=True,
        location=payload["location"],
        crs=payload.get("crs"),
        modality=modality,
        provenance={
            "source_ref": source_ref,
            "blob_content_hash": result.content_hash,
            "model_id": result.model_id,
            "verifier_kind": verdict.kind,
            "score": score,
        },
    )
    return SpeciesIDResult(accepted=True, observation=observation, verdict=verdict, provenance=base_provenance)


def attach_species_receipt(result: SpeciesIDResult, *, sink: Any = None) -> str | None:
    """Persist ``result`` as a substrate ``kind="record"`` item (mirrors ``gateway/trace_capture.py``'s
    `_persist`), so an accepted occurrence -- or a rejected/abstained detection's reasons -- traces back to
    its source image/audio/sequence, model, and verdict (IC-5-shaped provenance via the substrate sink).
    ``sink=None`` is a no-op: `identify_species` itself never requires a live substrate to run."""
    if sink is None:
        return None

    from mixle.substrate.core import SubstrateItem

    observation_payload = None
    if result.observation is not None:
        observation_payload = {
            "species_id": result.observation.species_id,
            "detection": result.observation.detection,
            "crs": result.observation.crs,
            "modality": result.observation.modality,
            "provenance": dict(result.observation.provenance),
        }
    payload = {
        "accepted": result.accepted,
        "verdict": {
            "passed": result.verdict.passed,
            "score": result.verdict.score,
            "reasons": list(result.verdict.reasons),
            "kind": result.verdict.kind,
        },
        "observation": observation_payload,
    }
    item = SubstrateItem(
        kind="record",
        text=f"species-id {result.provenance.get('species_id')!r} accepted={result.accepted}",
        payload=payload,
        provenance=dict(result.provenance),
    )
    return sink.put(item)
