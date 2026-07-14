"""L4 -- the MCP tool wrapper + IC-13 knowledge-item bridge for `DomainModelAdapter` connectors (the
cross-model keystone). Wraps every registered `DomainModelAdapter` (climate, hydrology, emissions-factor,
rock-physics, ...) into an `mcp.server.Tool` so a hosted chat model can reach an external/domain connector
mid-conversation exactly the way it reaches the platform's own physics tools (E4) -- through
`ToolRegistry._add`, never a special case.

Every call normalizes the adapter's `ProvenancedResult` into an IC-13-shaped knowledge item and stores it
(``knowledge_store``), so the *canonical* value/uncertainty always lives behind a content-addressed id. The
MCP transport's compact JSON reply (``{knowledge_item_id, content_hash, source_result_hash, schema_uri,
preview}``) is only a rendering a caller may discard -- the canonical data is resolved by id, never
reconstructed from ``preview``.

M1a is the task that lands the real ``mixle_knowledge.contracts.KnowledgeItem`` construction and its own
enclosing-item content-hash rule; it has not landed yet (and, at the moment this module was written, the
locally-installed ``mixle_knowledge`` checkout does not even carry the IC-13 `KnowledgeItem` shape -- it
lives only on that repo's ``release/0.8.0`` branch). Until M1a lands, this module builds the equivalent
shape locally as a plain dict -- the same "build the IC-13 shape locally until the bridging task lands"
pattern `gateway/tool_registry.py`'s `_knowledge_item_from_substrate` already uses for the analogous
substrate-retrieval bridge -- so external/domain-model calls are already provenanced end-to-end today, and
M1a's landing only has to swap the storage/validation layer, not this module's public shape.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..models.domain_adapter import DomainModelAdapter, ProvenancedResult
from .schema_bridge import mcp_tool_to_tooldef
from .server import Tool

if TYPE_CHECKING:  # pragma: no cover - type-checking only, avoids importing the gateway at module load time
    from ..gateway.tool_registry import ToolRegistry

# A generic, unvalidated schema for a normalized domain-model result -- mirrors the
# `_GENERIC_SCHEMA_URI` fallback `gateway/tool_registry.py` uses for substrate kinds with no dedicated
# IC-13 payload profile; a real per-manifest schema URI can be layered in later without changing callers.
DOMAIN_RESULT_SCHEMA_URI = "mixle://schema/domain-model-result/1"

# Above this many scalar elements a value is treated as "large" and spilled to an artifact ref instead of
# being embedded inline (DR-ALG L4 step 4). The MVP stubs only ever return small structured dicts, so this
# only exercises the payload branch today -- the branch exists so a real array-valued connector (e.g. a
# gridded climate field) has somewhere to go without changing this module's shape.
LARGE_VALUE_THRESHOLD = 4096


@runtime_checkable
class KnowledgeStore(Protocol):
    """Minimal store a knowledge item is hydrated back out of by id -- "put now, get later" is exactly the
    invariant the DoD enforces (the compact JSON ``preview`` a caller sees is never the canonical source)."""

    def put(self, item: dict[str, Any]) -> str: ...

    def get(self, item_id: str) -> dict[str, Any] | None: ...


class InMemoryKnowledgeStore:
    """Process-local `KnowledgeStore` -- sufficient for the MVP/fixture-free DoD. A real deployment backs
    this with the substrate (`mixle.substrate.core.Substrate`) or an M1a-provided store exposing the
    identical `put`/`get` shape; `build_domain_tools`/`register_domain_tools` accept any such store."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def put(self, item: dict[str, Any]) -> str:
        item_id = item["id"]
        self._items[item_id] = dict(item)
        return item_id

    def get(self, item_id: str) -> dict[str, Any] | None:
        item = self._items.get(item_id)
        return dict(item) if item is not None else None


def _item_content_hash(schema_uri: str, payload: Any, artifact_ref: str | None, provenance: dict[str, Any]) -> str:
    """M1a lands the real enclosing-item content-hash rule; until then this mirrors the canonical-JSON
    sha256 pattern `gateway/tool_registry.py`'s `_content_hash` already established for the IC-13 bridge."""
    canonical = json.dumps(
        {"schema_uri": schema_uri, "payload": payload, "artifact_ref": artifact_ref, "provenance": provenance},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _size_of(value: Any) -> int:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return int(value.size)
    except ImportError:  # pragma: no cover - numpy is a real dependency of this platform; defensive only
        pass
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def _normalize_value(value: Any) -> tuple[Any, str | None]:
    """Small structured values stay typed payloads; large arrays/files become artifact refs instead of
    being embedded inline (DR-ALG L4 step 4). Returns ``(payload, artifact_ref)`` -- exactly one is set."""
    if _size_of(value) <= LARGE_VALUE_THRESHOLD:
        return value, None
    try:
        import numpy as np

        raw = value.tobytes() if isinstance(value, np.ndarray) else json.dumps(value, default=str).encode("utf-8")
    except ImportError:  # pragma: no cover - defensive only
        raw = json.dumps(value, default=str).encode("utf-8")
    ref_hash = hashlib.sha256(raw).hexdigest()
    return None, f"artifact://domain-model/{ref_hash}"


def _to_knowledge_item(
    adapter: DomainModelAdapter, result: ProvenancedResult, inputs: dict[str, Any]
) -> dict[str, Any]:
    """Normalize a `ProvenancedResult` into an IC-13-shaped item: model/version/source-result-hash/
    uncertainty become provenance fields, and ``content_hash`` is the *enclosing item's* hash (M1a) --
    distinct from ``result.content_hash`` (the source-result hash the item's provenance points back to)."""
    payload, artifact_ref = _normalize_value(result.value)
    provenance = {
        "model_id": result.model_id,
        "version": result.version,
        "source_result_hash": result.content_hash,
        "inputs": inputs,
    }
    content_hash = _item_content_hash(DOMAIN_RESULT_SCHEMA_URI, payload, artifact_ref, provenance)
    item_id = f"domain-item-{content_hash[:24]}"
    return {
        "id": item_id,
        "kind": "domain_result",
        "modality": "structured",
        "schema_uri": DOMAIN_RESULT_SCHEMA_URI,
        "schema_version": "1.0.0",
        "content_hash": content_hash,
        "payload": payload,
        "artifact_ref": artifact_ref,
        "uncertainty": result.uncertainty,
        "provenance": provenance,
    }


def _preview(value: Any) -> str:
    try:
        return json.dumps(value, default=str)[:500]
    except TypeError:  # pragma: no cover - defensive only
        return str(value)[:500]


def _domain_tool(adapter: DomainModelAdapter, store: KnowledgeStore) -> Tool:
    domain_manifest = adapter.domain_manifest

    async def handler(args: dict[str, Any]) -> str:
        inputs = dict(args["inputs"]) if isinstance(args.get("inputs"), dict) else dict(args)
        result = await adapter.call(inputs)
        item = _to_knowledge_item(adapter, result, inputs)
        item_id = store.put(item)
        response = {
            "knowledge_item_id": item_id,
            "content_hash": item["content_hash"],
            "source_result_hash": result.content_hash,
            "schema_uri": item["schema_uri"],
            "preview": _preview(result.value),
        }
        return json.dumps(response)

    return Tool(
        name=f"domain__{adapter.name}",
        description=(
            f"Call the external/domain model {adapter.name!r} (prior reliability "
            f"{domain_manifest.reliability}) and return a content-addressed knowledge-item handle to its "
            f"provenanced result -- resolve the canonical value/uncertainty by id, not from the preview."
        ),
        input_schema=domain_manifest.input_schema or {"type": "object", "properties": {}},
        handler=handler,
    )


def build_domain_tools(
    adapters: list[DomainModelAdapter], *, knowledge_store: KnowledgeStore | None = None
) -> dict[str, Tool]:
    """Build one MCP `Tool` per `DomainModelAdapter` (a default in-memory knowledge store when none is
    given -- every adapter passed in the same call shares that one store)."""
    store: KnowledgeStore = knowledge_store if knowledge_store is not None else InMemoryKnowledgeStore()
    tools: dict[str, Tool] = {}
    for adapter in adapters:
        tool = _domain_tool(adapter, store)
        tools[tool.name] = tool
    return tools


def register_domain_tools(
    tool_reg: "ToolRegistry", adapters: list[DomainModelAdapter], *, knowledge_store: KnowledgeStore | None = None
) -> None:
    """Register every `build_domain_tools` tool into an existing `ToolRegistry` through its own `_add` --
    the identical seam E4's physics tools use, so external/domain models and physics tools coexist through
    disjoint registration calls with no shared code touched."""
    for tool in build_domain_tools(adapters, knowledge_store=knowledge_store).values():
        tool_reg._add(mcp_tool_to_tooldef(tool), tool.handler)
