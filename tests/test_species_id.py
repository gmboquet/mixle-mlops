"""N3 -- cross-modal species identification: a camera-trap image and an audio clip each pass through
`identify_species` with a high-confidence LABELED-synthetic adapter + a bounds/score verifier and come out
as a verified, model-attributed IC-12 `SpeciesObservation`; a low-confidence clip abstains and writes no
occurrence at all.

The verifier here stands in for E10's real physical verifier (Non-goals: "no verifier algorithm (E10)") --
it only checks that the claimed species is on the site's plausible-range list, which is enough to exercise
`identify_species`'s own abstain/accept wiring independent of whichever real `Verifier` implementation is
plugged in later.
"""

from __future__ import annotations

import asyncio
import string

import numpy as np

from mixle_mlops.models.species_adapters import BioacousticAdapter, CameraTrapAdapter, EdnaAdapter
from mixle_mlops.multimodal.species_id import SpeciesIDResult, identify_species
from mixle_mlops.verification.base import Verdict, Verifier


class _BoundsScoreVerifier:
    """DoD-only stub `Verifier`: passes when the claimed species is in the site's plausible-range list
    (a `physical`-kind bounds check), regardless of score -- `identify_species` applies its own
    `abstain_below` score floor on top of this verdict."""

    def verify(self, claim: dict, context: dict) -> Verdict:
        plausible = context.get("plausible_species") or []
        species_id = claim["species_id"]
        ok = species_id in plausible
        reasons = [] if ok else [f"{species_id!r} is not on the site plausible-range list {plausible!r}"]
        return Verdict(passed=ok, score=1.0 if ok else 0.0, reasons=reasons, kind="physical")


def _is_hex64(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in string.hexdigits for c in value)


def _payload(*, modality: str, blob_ref: str, species_id: str, score: float) -> dict:
    return {
        "modality": modality,
        "blob_ref": blob_ref,
        "species_id": species_id,
        "score": score,
        "location": np.array([34.05, -118.25]),
        "crs": "EPSG:4326",
        "plausible_species": ["puma_concolor", "canis_latrans", "lynx_rufus"],
    }


def test_camera_trap_high_confidence_is_accepted_with_verified_observation():
    payload = _payload(modality="camera", blob_ref="blob://camera/img-001", species_id="puma_concolor", score=0.94)
    result = asyncio.run(identify_species(payload, adapter=CameraTrapAdapter(), verifier=_BoundsScoreVerifier()))

    assert isinstance(result, SpeciesIDResult)
    assert result.accepted is True
    assert result.verdict.passed is True
    assert result.observation is not None
    assert result.observation.species_id == "puma_concolor"
    assert result.observation.detection is True
    assert result.observation.modality == "camera"
    assert result.observation.crs == "EPSG:4326"
    assert np.allclose(result.observation.location, payload["location"])

    # every detection carries model attribution + a content hash (IC-7 ProvenancedResult)
    assert result.observation.provenance["model_id"] == "camera-trap-species-id-stub"
    assert _is_hex64(result.observation.provenance["blob_content_hash"])
    assert _is_hex64(result.provenance["content_hash"])


def test_bioacoustic_high_confidence_is_accepted_with_verified_observation():
    payload = _payload(modality="acoustic", blob_ref="blob://audio/clip-002", species_id="canis_latrans", score=0.81)
    result = asyncio.run(identify_species(payload, adapter=BioacousticAdapter(), verifier=_BoundsScoreVerifier()))

    assert result.accepted is True
    assert result.verdict.passed is True
    assert result.observation is not None
    assert result.observation.species_id == "canis_latrans"
    assert result.observation.modality == "acoustic"
    assert result.observation.provenance["model_id"] == "bioacoustic-species-id-stub"
    assert _is_hex64(result.observation.provenance["blob_content_hash"])


def test_low_confidence_clip_abstains_and_writes_no_occurrence():
    payload = _payload(modality="acoustic", blob_ref="blob://audio/clip-003", species_id="lynx_rufus", score=0.3)
    result = asyncio.run(identify_species(payload, adapter=BioacousticAdapter(), verifier=_BoundsScoreVerifier()))

    assert result.accepted is False
    assert result.observation is None
    # abstained for being below the score floor even though the species itself is plausible
    assert result.verdict.passed is True
    assert any("0.3" in reason or "abstain" in reason.lower() for reason in result.provenance["reasons"])
    assert _is_hex64(result.provenance["content_hash"])


def test_edna_adapter_also_conforms_to_the_gate():
    payload = _payload(modality="edna", blob_ref="blob://edna/read-004", species_id="puma_concolor", score=0.7)
    result = asyncio.run(identify_species(payload, adapter=EdnaAdapter(), verifier=_BoundsScoreVerifier()))
    assert result.accepted is True
    assert result.observation.modality == "edna"


def test_unlisted_species_is_rejected_by_the_verifier_regardless_of_score():
    payload = _payload(modality="camera", blob_ref="blob://camera/img-005", species_id="ursus_arctos", score=0.99)
    result = asyncio.run(identify_species(payload, adapter=CameraTrapAdapter(), verifier=_BoundsScoreVerifier()))
    assert result.accepted is False
    assert result.observation is None
    assert result.verdict.passed is False


def test_adapters_are_domain_model_adapters_and_registered_via_l4():
    from mixle_mlops.gateway.tool_registry import ToolRegistry
    from mixle_mlops.core.registry import ModelRegistry
    from mixle_mlops.models.domain_adapter import DomainModelAdapter
    from mixle_mlops.models.species_adapters import SPECIES_ADAPTERS, register_species_adapters

    for adapter in SPECIES_ADAPTERS:
        assert isinstance(adapter, DomainModelAdapter)

    tool_reg = ToolRegistry(
        ModelRegistry(),
        include_mcp=False,
        include_rag=False,
        include_mixle=False,
        include_platform=False,
        model_id="test-model",
    )
    register_species_adapters(tool_reg)
    for adapter in SPECIES_ADAPTERS:
        assert tool_reg.has(f"domain__{adapter.name}")


def test_verifier_protocol_is_satisfied_by_the_dod_stub():
    assert isinstance(_BoundsScoreVerifier(), Verifier)
