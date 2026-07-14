"""IC-7 -- `DomainModelAdapter` for non-chat external/domain models (work-plan L4, the cross-model keystone).

Extends the gateway's `ModelAdapter` (core/adapters.py) so a climate-projection / hydrology / emissions-factor
model -- or any other external, non-chat domain model -- is a first-class, provenanced, callable tool
alongside the platform's own physics tools (E4). `call(inputs)` returns a `ProvenancedResult` carrying the
value, the model id + version, a content hash of the response, and optional uncertainty; `manifest` (a
classvar) advertises name/io-schema/cost/reliability for the router (IC-10). `ClimateProjectionStub` is a
deterministic, clearly-**synthetic** worked example: no network call, no real CMIP/weather integration
(Non-goals) -- a real external connector (a hosted CMIP6 emulator, a weather-API wrapper, ...) registers
through this identical base class.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar

from ..core.adapters import CapabilityError, ChatCompletionChunk, ChatRequest, ModelAdapter


@dataclass(frozen=True)
class ProvenancedResult:
    """A domain-model output with the provenance every external claim must carry (IC-7)."""

    value: Any
    model_id: str
    version: str
    content_hash: str
    uncertainty: Any | None = None


@dataclass(frozen=True)
class ModelManifest:
    """Static advert for the router (IC-10): identity, IO schema, price, and a prior reliability weight.

    Frozen per IC-7 -- ``io_schema`` is deliberately one combined dict (not separate input/output fields).
    `DomainManifest` below is the richer, catalog-facing view built *from* a `ModelManifest`.
    """

    name: str
    io_schema: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    reliability: float = 1.0


@dataclass(frozen=True)
class DomainManifest:
    """Catalog-facing view of a domain model's manifest: separate ``input_schema``/``output_schema`` (rather
    than IC-7's combined ``io_schema``) so a JSON-schema tool wrapper (``mcp/domain_tools.py``) and the IC-10
    catalog can advertise each half on its own. Derived from a `ModelManifest` via `from_model_manifest`;
    `catalog_entry` renders the IC-10 `CatalogEntry` shape (``{id, schema, owner, cost, reliability,
    verifier}``).

    Note: IC-7 freezes ``DomainModelAdapter.manifest`` as a ``ModelManifest`` (single ``io_schema`` dict);
    this class does not replace that classvar -- it is the additional, IC-10-shaped view every concrete
    adapter exposes via ``adapter.domain_manifest``, reconciling this task's own richer catalog-registration
    ask with the frozen IC-7 shape instead of redefining it.
    """

    name: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    reliability: float = 1.0

    @classmethod
    def from_model_manifest(cls, manifest: ModelManifest) -> "DomainManifest":
        io = manifest.io_schema or {}
        return cls(
            name=manifest.name,
            input_schema=io.get("input", {}) or {},
            output_schema=io.get("output", {}) or {},
            cost=manifest.cost,
            reliability=manifest.reliability,
        )

    def catalog_entry(self, *, owner: str = "external", verifier: str | None = None) -> dict[str, Any]:
        """Render the IC-10 `CatalogEntry` shape (``{id, schema, owner, cost, reliability, verifier}``)."""
        return {
            "id": self.name,
            "schema": {"input": self.input_schema, "output": self.output_schema},
            "owner": owner,
            "cost": self.cost,
            "reliability": self.reliability,
            "verifier": verifier,
        }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


class DomainModelAdapter(ModelAdapter):
    """Base class for non-chat domain/external models. Subclasses set `manifest` (and, optionally,
    `version`) and implement `call`."""

    kind = "domain"
    manifest: ClassVar[ModelManifest]
    version: ClassVar[str] = "1.0.0"  # subclasses override; stamped onto every `ProvenancedResult`

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def domain_manifest(self) -> DomainManifest:
        """The IC-10-facing view of this adapter's manifest (input/output schema split apart)."""
        return DomainManifest.from_model_manifest(self.manifest)

    def capabilities(self) -> set[str]:
        return {"call"}

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """A domain model is not a chat model; the agentic loop reaches it via `call`, not `stream`."""
        raise CapabilityError(self.name, "stream")
        yield  # pragma: no cover  (makes this an async generator so the signature matches ModelAdapter)

    @abstractmethod
    async def call(self, inputs: dict) -> ProvenancedResult:
        """Run the external/domain model on ``inputs`` and return a `ProvenancedResult` (value + provenance)."""
        ...

    def _provenanced(self, value: Any, inputs: dict, uncertainty: Any | None = None) -> ProvenancedResult:
        """Hash ``inputs`` + the typed ``value`` into `ProvenancedResult.content_hash` -- the source-result
        hash a caller later attributes a knowledge item to (``mcp/domain_tools.py``) -- and stamp this
        adapter's model id/version."""
        digest = hashlib.sha256(
            _canonical_json(
                {"model_id": self.name, "version": self.version, "inputs": inputs, "value": value}
            ).encode("utf-8")
        ).hexdigest()
        return ProvenancedResult(
            value=value, model_id=self.name, version=self.version, content_hash=digest, uncertainty=uncertainty
        )


class ClimateProjectionStub(DomainModelAdapter):
    """A deterministic, clearly-**synthetic** stand-in for a real external climate/GCM connector -- no
    network call (Non-goals: no real CMIP/weather API integration in the MVP). A real connector (a hosted
    CMIP6 emulator, a weather-API wrapper, ...) registers through this identical `DomainModelAdapter`
    surface, so nothing downstream (routing, fusion, MCP registration) special-cases "stub vs. real".
    """

    manifest = ModelManifest(
        name="climate-projection-stub",
        io_schema={
            "input": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "latitude, degrees"},
                    "lon": {"type": "number", "description": "longitude, degrees"},
                    "year": {"type": "integer", "description": "projection year"},
                },
                "required": ["lat", "lon", "year"],
            },
            "output": {
                "type": "object",
                "properties": {
                    "tas_anomaly_c": {
                        "type": "number",
                        "description": "near-surface air-temperature anomaly, degC",
                    }
                },
            },
        },
        cost=0.0,
        reliability=0.7,
    )
    version = "synthetic-v1"

    async def call(self, inputs: dict) -> ProvenancedResult:
        lat = float(inputs.get("lat", 0.0))
        lon = float(inputs.get("lon", 0.0))
        year = int(inputs.get("year", 2050))
        # A smooth, bounded, fully deterministic function of the inputs -- labeled synthetic, not a real
        # climate model. Real connectors implement `call` against an actual service/API and register
        # through this same base class with no other change.
        anomaly = 0.02 * (year - 2000) + 0.01 * math.sin(math.radians(lat)) - 0.005 * math.cos(math.radians(lon))
        value = {"tas_anomaly_c": round(anomaly, 6)}
        uncertainty = {"std": 0.15, "kind": "synthetic"}
        return self._provenanced(value, inputs, uncertainty)
