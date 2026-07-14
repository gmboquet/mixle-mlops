"""N3 -- example `DomainModelAdapter` connectors for cross-modal species identification: a camera-trap
image classifier, a bioacoustic classifier, and an eDNA/barcode classifier. Each is a LABELED synthetic
stub (Non-goals: no real trained camera/acoustic/eDNA models in the MVP) -- the "classification" is the
``species_id``/``score`` label the caller already supplies (a curated fixture, a human-reviewed field
log, ...), and the adapter's only real job is to stamp that label with model attribution, a version, and
a content hash, exactly the honesty `ClimateProjectionStub` (L4, ``domain_adapter.py``) already
establishes for a deterministic, clearly-synthetic external connector: no network call, no real
CNN/acoustic/barcode inference. A real trained classifier registers through this identical
`DomainModelAdapter` base with no other change to `identify_species` or the registration seam below.
"""

from __future__ import annotations

from typing import Any

from ..models.domain_adapter import DomainModelAdapter, ModelManifest, ProvenancedResult


def _species_io_schema(blob_kind: str) -> dict[str, Any]:
    """The shared IO schema shape every species-id stub advertises (IC-7 `ModelManifest.io_schema`), only
    the ``blob_kind`` description differs per modality."""
    return {
        "input": {
            "type": "object",
            "properties": {
                "blob_ref": {
                    "type": "string",
                    "description": f"content-addressed reference to the source {blob_kind}",
                },
                "species_id": {
                    "type": "string",
                    "description": "candidate species id -- labeled input in this synthetic MVP stub",
                },
                "score": {
                    "type": "number",
                    "description": "classifier confidence in [0, 1] -- labeled input in this synthetic MVP stub",
                },
            },
            "required": ["blob_ref", "species_id", "score"],
        },
        "output": {
            "type": "object",
            "properties": {
                "species_id": {"type": "string"},
                "score": {"type": "number"},
            },
        },
    }


class CameraTrapAdapter(DomainModelAdapter):
    """image blob -> ``{species_id, score}``; a LABELED synthetic stand-in for a real camera-trap CNN
    species classifier."""

    manifest = ModelManifest(
        name="camera-trap-species-id-stub",
        io_schema=_species_io_schema("camera-trap image"),
        cost=0.0,
        reliability=0.85,
    )
    version = "synthetic-camera-v1"

    async def call(self, inputs: dict) -> ProvenancedResult:
        value = {"species_id": str(inputs["species_id"]), "score": round(float(inputs["score"]), 6)}
        return self._provenanced(value, inputs)


class BioacousticAdapter(DomainModelAdapter):
    """audio blob -> ``{species_id, score}``; a LABELED synthetic stand-in for a real bioacoustic
    (e.g. BirdNET-style) species classifier."""

    manifest = ModelManifest(
        name="bioacoustic-species-id-stub",
        io_schema=_species_io_schema("audio clip"),
        cost=0.0,
        reliability=0.8,
    )
    version = "synthetic-acoustic-v1"

    async def call(self, inputs: dict) -> ProvenancedResult:
        value = {"species_id": str(inputs["species_id"]), "score": round(float(inputs["score"]), 6)}
        return self._provenanced(value, inputs)


class EdnaAdapter(DomainModelAdapter):
    """sequence read -> ``{species_id, score}``; a LABELED synthetic stand-in for a real eDNA/barcode
    species classifier."""

    manifest = ModelManifest(
        name="edna-species-id-stub",
        io_schema=_species_io_schema("eDNA sequence read"),
        cost=0.0,
        reliability=0.75,
    )
    version = "synthetic-edna-v1"

    async def call(self, inputs: dict) -> ProvenancedResult:
        value = {"species_id": str(inputs["species_id"]), "score": round(float(inputs["score"]), 6)}
        return self._provenanced(value, inputs)


SPECIES_ADAPTERS: tuple[DomainModelAdapter, ...] = (CameraTrapAdapter(), BioacousticAdapter(), EdnaAdapter())


def register_species_adapters(tool_reg: Any, *, knowledge_store: Any = None) -> None:
    """Register all three species-id adapters into ``tool_reg`` through L4's own `register_domain_tools`
    (``mcp/domain_tools.py`` -- the identical seam E4's physics tools and L4's climate/rock-physics stubs
    use); this module never edits `register_domain_tools`/`ToolRegistry._add` itself."""
    from ..mcp.domain_tools import register_domain_tools

    register_domain_tools(tool_reg, list(SPECIES_ADAPTERS), knowledge_store=knowledge_store)
