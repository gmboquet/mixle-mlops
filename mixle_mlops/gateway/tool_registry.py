"""The model-callable tool catalog — the bridge that lets a hosted model (mid-conversation, ReAct-style) reach
the platform's own capabilities. It assembles OpenAI tool declarations + dispatch handlers from five sources:

  * the platform's MCP tools (``list_models`` + ``chat__<model>`` + ``score__<model>``) via the MCP schema bridge,
  * the user's RAG store, as a callable ``rag_search`` (so the model can *decide* to retrieve, not always pay for it)
    -- federated with a physics :class:`mixle_pde.reasoning.SpatialFieldStore` when one is configured, so a single
    location-anchored call surfaces nearby documents AND nearby field tiles as source-native records,
  * mixle distribution/decision capabilities, as ``mixle_predict`` / ``mixle_decide`` over any hosted mixle model,
  * the platform's richer substrate/UQ/simulation capabilities (``substrate_retrieve``, ``uq``, ``simulate`` /
    ``synthesize``), gated by ``include_platform`` — the same tools a person can already reach through the
    substrate, prediction, and simulation surfaces, now exposed as model-callable so a hosted model can pull
    structured evidence or ask for calibrated uncertainty mid-conversation instead of guessing.

``specs()`` returns the OpenAI ``tools`` array; ``dispatch(name, args)`` executes a tool call and returns a
JSON-serializable result. Errors are returned in-band (``{"error": ...}``) so a bad tool call never crashes the loop.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Awaitable, Callable

import numpy as np
from mixle.task.replay import TraceStep

from ..core.adapters import FunctionDef, ToolDef
from ..core.registry import ModelRegistry
from ..mcp.physics_tools import build_physics_tools
from ..mcp.schema_bridge import mcp_tool_to_tooldef
from ..mcp.server import build_model_tools
from .trace_capture import record_tool_call

Handler = Callable[[dict[str, Any]], Awaitable[Any]]

# --- substrate item kind -> IC-13 (KnowledgeItem) resource/modality/schema mapping -----------------------------
# Only "graph"/"record"/"image" substrate kinds have a validated IC-13 payload profile (property-graph,
# typed-table, spatial-media respectively); every other substrate modality (text, signal, field, event_stream,
# artifact, trace, context) exports under a generic, unvalidated schema so nothing in the substrate is dropped.
_GENERIC_SCHEMA_URI = "mixle://schema/substrate-item/1"

# A ``mixle_pde.reasoning.SpatialFieldStore`` instance, or a zero-arg factory that builds one lazily (so an
# expensive volume isn't materialised until a registry that actually needs field retrieval is built).
FieldStoreLike = Any


class ToolRegistry:
    def __init__(self, registry: ModelRegistry, *, user_id: str | None = None,
                 names: list[str] | None = None, include_mcp: bool = True,
                 include_rag: bool = True, include_mixle: bool = True,
                 include_platform: bool = True, include_physics: bool = True, substrate: Any = None,
                 model_id: str | None = None, verifier: Any = None,
                 field_store: FieldStoreLike | None = None):
        self.registry = registry
        self.user_id = user_id
        self.model_id = model_id      # the calling chat model's id, stamped onto every recorded step (M4a)
        self.verifier = verifier      # an optional IC-6 Verifier gating each dispatched tool's result
        self.substrate = substrate    # an optional pre-populated mixle.substrate.Substrate to reuse (D7);
                                       # lazily constructed on first `substrate_retrieve` dispatch when absent
        self.trace_steps: list[TraceStep] = []   # every call this turn, already ExecutionTrace-ready (M4a)
        self._whitelist = set(names) if names else None      # optional restriction of the exposed catalog
        self._field_store = field_store   # optional SpatialFieldStore (or factory) federated into rag_search (E8)
        self._defs: dict[str, ToolDef] = {}
        self._handlers: dict[str, Handler] = {}
        self._build(include_mcp, include_rag, include_mixle, include_platform, include_physics)

    # --- assembly ---
    def _add(self, tooldef: ToolDef, handler: Handler) -> None:
        name = tooldef.function.name
        if self._whitelist is not None and name not in self._whitelist:
            return
        self._defs[name] = tooldef
        self._handlers[name] = handler

    def _build(self, include_mcp: bool, include_rag: bool, include_mixle: bool, include_platform: bool = True,
               include_physics: bool = True) -> None:
        if include_mcp:
            # physics tools (E4) are added separately below, gated by their own flag, so the two are independent
            # (e.g. include_mcp=False, include_physics=True exposes only the physics tools).
            for tool in build_model_tools(self.registry, include_physics=False).values():
                self._add(mcp_tool_to_tooldef(tool), tool.handler)   # MCP handler(args) -> awaitable[str]
        if include_physics:
            for tool in build_physics_tools().values():
                self._add(mcp_tool_to_tooldef(tool), tool.handler)
        if include_rag and self.user_id:
            self._add(
                ToolDef(function=FunctionDef(
                    name="rag_search",
                    description="Search the user's uploaded documents and past conversations for relevant "
                                "context. Pass 'location' to also retrieve nearby physics field tiles from the "
                                "configured SpatialFieldStore, federated alongside the document hits.",
                    parameters={"type": "object", "properties": {
                        "query": {"type": "string", "description": "what to search for"},
                        "k": {"type": "integer", "description": "number of snippets/tiles", "default": 5},
                        "location": {"type": "array", "items": {"type": "number"},
                                     "description": "optional (east, north[, up]) survey-frame coordinate; when "
                                                     "given, nearby field tiles are retrieved alongside documents"},
                        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                                 "description": "optional (minx, miny, maxx, maxy) geographic bounding box that "
                                                "pre-filters document chunks tagged with lon/lat metadata"},
                    }, "required": ["query"]})),
                self._rag_search)
        if include_mixle:
            self._add(
                ToolDef(function=FunctionDef(
                    name="mixle_predict",
                    description="Predict calibrated distributions / quantiles for records under a hosted mixle model.",
                    parameters={"type": "object", "properties": {
                        "model": {"type": "string", "description": "the hosted mixle model id"},
                        "records": {"type": "array", "items": {}, "description": "records to predict"},
                    }, "required": ["model", "records"]})),
                self._mixle_predict)
            self._add(
                ToolDef(function=FunctionDef(
                    name="mixle_decide",
                    description="Bayes-optimal decision (under a named loss) for records under a hosted mixle model.",
                    parameters={"type": "object", "properties": {
                        "model": {"type": "string"},
                        "records": {"type": "array", "items": {}},
                        "loss": {"type": "string", "description": "squared|absolute|linex|newsvendor"},
                        "actions": {"type": "array", "items": {}},
                    }, "required": ["model", "records"]})),
                self._mixle_decide)
            self._add(
                ToolDef(function=FunctionDef(
                    name="mixle_solve",
                    description="Offload an EXACT computation to mixle's deterministic solver instead of computing "
                                "by hand. ops: 'eval' (arithmetic expression with vars), 'normal_prob' (Gaussian "
                                "tail probability), 'describe' (exact summary stats of data), 'fit_predict' "
                                "(auto-fit a distribution and read off mean/quantile).",
                    parameters={"type": "object", "properties": {
                        "op": {"type": "string", "enum": ["eval", "normal_prob", "describe", "fit_predict"]},
                        "expr": {"type": "string"}, "vars": {"type": "object"},
                        "mean": {"type": "number"}, "std": {"type": "number"}, "x": {"type": "number"},
                        "side": {"type": "string", "enum": ["upper", "lower"]},
                        "data": {"type": "array", "items": {"type": "number"}},
                        "query": {"type": "string"}, "q": {"type": "number"},
                    }, "required": ["op"]})),
                self._mixle_solve)
        if include_platform:
            self._add(
                ToolDef(function=FunctionDef(
                    name="substrate_retrieve",
                    description="Retrieve structured evidence (graphs, tables, spatial media, documents, ...) "
                                "from the knowledge substrate for a query, as a canonical IC-13 knowledge bundle "
                                "-- payloads, relations and media refs, not a flattened text blob.",
                    parameters={"type": "object", "properties": {
                        "query": {"type": "string", "description": "what to retrieve"},
                        "k": {"type": "integer", "description": "max items to include", "default": 5},
                        "project_id": {"type": "string", "description": "the owning project id"},
                        "target_id": {"type": "string", "description": "the requesting model/tool id, if any"},
                    }, "required": ["query"]})),
                self._substrate_retrieve)
            self._add(
                ToolDef(function=FunctionDef(
                    name="uq",
                    description="Calibrated predictive intervals for records under a hosted mixle model, at a "
                                "chosen confidence level (e.g. 0.9 for a central 90% interval).",
                    parameters={"type": "object", "properties": {
                        "model": {"type": "string", "description": "the hosted mixle model id"},
                        "records": {"type": "array", "items": {}, "description": "records to predict"},
                        "level": {"type": "number", "description": "central interval mass, e.g. 0.9"},
                    }, "required": ["model", "records"]})),
                self._uq)
            simulate_params = {"type": "object", "properties": {
                "spec": {"type": "object", "description": "{'model': hosted mixle model id, 'n': draws, "
                                                            "'scenario'?: registered scenario name, "
                                                            "'interventions'?: {field_index: value}, 'seed'?: int}"},
            }, "required": ["spec"]}
            self._add(
                ToolDef(function=FunctionDef(
                    name="simulate",
                    description="Forward-simulate synthetic records from a hosted mixle model's fitted "
                                "generative distribution, optionally under a named/ad-hoc intervention.",
                    parameters=simulate_params)),
                self._simulate)
            self._add(
                ToolDef(function=FunctionDef(
                    name="synthesize",
                    description="Draw synthetic records from a hosted mixle model's fitted generative "
                                "distribution (alias of `simulate` for the no-intervention baseline case).",
                    parameters=simulate_params)),
                self._simulate)

    # --- public surface ---
    def specs(self) -> list[ToolDef]:
        return list(self._defs.values())

    def has(self, name: str) -> bool:
        return name in self._handlers

    async def dispatch(self, name: str, args: dict[str, Any]) -> Any:
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            result = await handler(args)
        except Exception as exc:                              # tool failures reported in-band, never crash the loop
            result = {"error": str(exc)}
        # M4a: every dispatched call becomes a TraceStep with its calling model + (optional) verdict
        # stamped in, so the turn this call was part of can be bound into a Receipt after the fact.
        self.trace_steps.append(
            record_tool_call(name, args, result, model_id=self.model_id or "", verifier=self.verifier)
        )
        return result

    # --- handlers ---
    async def _rag_search(self, args: dict[str, Any]) -> Any:
        query = str(args.get("query", "") or "")
        k = int(args.get("k", 5) or 5)
        location = args.get("location")
        bbox = args.get("bbox")

        results = self._retrieve_documents(query, k, bbox)
        if location is not None:
            results = results + self._retrieve_field_tiles(location, k)
        return {"results": results}

    def _retrieve_documents(self, query: str, k: int, bbox: Any = None) -> list[dict[str, Any]]:
        """Run the tenant's document RAG store (D3's ``retrieve``) and return source-native records.

        Always passes the tenant (``user_id``) and the query text through untouched -- retrieval federation
        never substitutes a different query or drops the tenant scope. ``bbox``, if given, is passed as a D3
        ``filters={"bbox": ...}`` geoscience pre-filter; ``retrieve`` implementations that predate that kwarg
        (a plain ``TypeError`` on the unexpected argument) degrade gracefully to the unfiltered call rather
        than failing the whole federated search.
        """
        from ..rag.index import retrieve

        hits: list[dict[str, Any]]
        if bbox is not None:
            filters = {"bbox": tuple(float(v) for v in bbox)}
            try:
                hits = retrieve(self.user_id, query, k=k, filters=filters)
            except TypeError:
                hits = retrieve(self.user_id, query, k=k)
        else:
            hits = retrieve(self.user_id, query, k=k)

        records = []
        for hit in hits:
            meta = hit.get("meta") or {}
            records.append({
                "source_kind": "document",
                "source_id": hit.get("source_id"),
                "score": hit.get("score"),
                "artifact_ref": meta.get("artifact_ref"),
                "selector": meta.get("selector"),
                "provenance": [{"chunk_id": hit.get("id"), "namespace": hit.get("namespace"), "meta": meta}],
                "text": hit.get("text", ""),
            })
        return records

    def _retrieve_field_tiles(self, location: Any, k: int) -> list[dict[str, Any]]:
        """Retrieve the ``k`` nearest field tiles to ``location`` from the configured SpatialFieldStore.

        ``CrossModalStore.retrieve`` returns bare corpus indices; each is dereferenced here against
        ``.payloads`` (the tile's member cell indices -- the raw sub-volume) and ``.keys`` (the tile centroid,
        used for the distance-based score) before it becomes a record. No bare integer is ever returned as
        evidence.
        """
        if self._field_store is None:
            return []
        store = self._field_store() if callable(self._field_store) else self._field_store
        cross_modal = store.store()

        loc = np.asarray(location, dtype=float).reshape(-1)
        records = []
        for idx in cross_modal.retrieve(loc, k=k):
            idx = int(idx)
            centroid = np.asarray(cross_modal.keys[idx], dtype=float)
            members = np.asarray(cross_modal.payloads[idx]).reshape(-1)   # dereferenced sub-volume, never idx
            distance = float(np.linalg.norm(centroid - loc))
            records.append({
                "source_kind": "field_tile",
                "source_index": idx,
                "score": 1.0 / (1.0 + distance),
                "artifact_ref": None,
                "selector": {"tile_index": idx, "n_cells": int(members.size)},
                "provenance": [{"kind": "spatial_field_store", "distance": distance}],
                "location": [float(v) for v in centroid],
                "grid": {"cell_indices": [int(m) for m in members]},
            })
        return records

    async def _mixle_predict(self, args: dict[str, Any]) -> Any:
        adapter = self.registry.get(args["model"])
        return await adapter.predict(args.get("records") or [])

    async def _mixle_decide(self, args: dict[str, Any]) -> Any:
        adapter = self.registry.get(args["model"])
        opts = {k: v for k, v in args.items() if k in ("loss", "actions")}
        return await adapter.decide(args.get("records") or [], **opts)

    async def _mixle_solve(self, args: dict[str, Any]) -> Any:
        from .program_offload import solve_program

        return solve_program(args)

    async def _substrate_retrieve(self, args: dict[str, Any]) -> Any:
        from mixle.substrate.context import ContextBudget, assemble_context
        from mixle.substrate.core import Substrate

        if self.substrate is None:                            # construct once, then reuse for later calls
            self.substrate = Substrate()
        query = str(args.get("query", "") or "")
        k = int(args.get("k", 5) or 5)
        project_id = str(args.get("project_id", "") or "")
        target_id = args.get("target_id")
        budget = ContextBudget(max_chars=max(k, 1) * 4000, max_items=max(k, 1), shape="passages")
        packet = assemble_context(self.substrate, query, budget=budget)
        return _knowledge_bundle_from_packet(
            packet,
            bundle_id=f"bundle-{uuid.uuid4().hex[:16]}",
            project_id=project_id,
            target_id=target_id if target_id else None,
        )

    async def _uq(self, args: dict[str, Any]) -> Any:
        adapter = self.registry.get(args["model"])
        opts: dict[str, Any] = {}
        level = args.get("level")
        if level is not None:
            opts["interval_level"] = float(level)
        return await adapter.predict(args.get("records") or [], **opts)

    async def _simulate(self, args: dict[str, Any]) -> Any:
        from mixle.inference.simulate import simulate as make_simulator

        spec = args.get("spec") or {}
        adapter = self.registry.get(spec["model"])
        model = getattr(adapter, "_model", None)
        if model is None:
            return {"error": f"model {spec.get('model')!r} has no simulatable fitted distribution"}
        sim = make_simulator(model)
        scenario_name = spec.get("scenario")
        interventions = spec.get("interventions")
        if interventions:
            sim.scenario(scenario_name or "adhoc", {int(k): v for k, v in interventions.items()})
            scenario_name = scenario_name or "adhoc"
        records = sim.run(int(spec.get("n", 100) or 100), scenario=scenario_name, seed=int(spec.get("seed", 0) or 0))
        return {"records": records}


# --- IC-13 knowledge-bundle bridge (M0b bridges core mixle -> mixle_knowledge; until it lands, this file
# builds the same shape locally from a core ContextPacket so `substrate_retrieve` is IC-13-valid today) ---
def _scope_to_access(scope: str) -> dict[str, Any]:
    from mixle_knowledge.contracts import AccessScope

    if scope == "local":
        return {"scope": AccessScope.PRIVATE}
    return {"scope": AccessScope.TEAM, "teams": [scope]}


def _content_hash(schema_uri: str, schema_version: str, payload: Any, artifact_ref: str | None,
                   metadata: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"schema_uri": schema_uri, "schema_version": schema_version, "payload": payload,
         "artifact_ref": artifact_ref, "metadata": metadata},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _knowledge_item_from_substrate(item: Any) -> Any:
    """Map one `mixle.substrate.core.SubstrateItem` to an IC-13 `KnowledgeItem` -- canonical payload,
    artifact ref, relations and access preserved, never flattened to text."""
    from mixle_knowledge.contracts import (
        PROPERTY_GRAPH_SCHEMA,
        SPATIAL_MEDIA_SCHEMA,
        TYPED_TABLE_SCHEMA,
        AccessPolicy,
        KnowledgeItem,
        KnowledgeRelation,
        Modality,
        ResourceKind,
    )

    kind_map = {
        "graph": (ResourceKind.ARTIFACT, Modality.GRAPH, PROPERTY_GRAPH_SCHEMA),
        "record": (ResourceKind.TABLE, Modality.TABLE, TYPED_TABLE_SCHEMA),
        "image": (ResourceKind.IMAGE, Modality.IMAGE, SPATIAL_MEDIA_SCHEMA),
    }
    resource_kind, modality, schema_uri = kind_map.get(
        item.kind, (ResourceKind.ARTIFACT, Modality.STRUCTURED, _GENERIC_SCHEMA_URI)
    )
    schema_version = "1.0.0"
    payload = dict(item.payload) if item.payload else {}
    artifact_ref: str | None = None
    if schema_uri == SPATIAL_MEDIA_SCHEMA:
        artifact_ref = payload.pop("ref", None)
    if not payload and artifact_ref is None:
        payload = {"text": item.text}          # never leave both payload and artifact_ref empty
    metadata = {"tags": list(item.tags), "scope": item.scope, "substrate_kind": item.kind}
    content_hash = _content_hash(schema_uri, schema_version, payload or None, artifact_ref, metadata)
    relations = [KnowledgeRelation(predicate="related_to", target_id=link) for link in item.links]
    return KnowledgeItem(
        id=item.id,
        kind=resource_kind,
        modality=modality,
        schema_uri=schema_uri,
        schema_version=schema_version,
        content_hash=content_hash,
        payload=payload or None,
        artifact_ref=artifact_ref,
        text_surface=item.text or None,
        relations=relations,
        metadata=metadata,
        access=AccessPolicy(**_scope_to_access(item.scope)),
    )


def _knowledge_bundle_from_packet(packet: Any, *, bundle_id: str, project_id: str,
                                   target_kind: str = "model", target_id: str | None = None) -> dict[str, Any]:
    """Export a core `ContextPacket` as a validated IC-13 `KnowledgeBundle` dict: full item payloads, refs
    and relation links stay canonical; `packet.render()` is only a `renderings["legacy_text"]` view."""
    from mixle_knowledge.contracts import KnowledgeBundle

    items = [_knowledge_item_from_substrate(item) for item in packet.items]
    bundle = KnowledgeBundle(
        id=bundle_id,
        project_id=project_id,
        task=packet.task,
        target_kind=target_kind,
        target_id=target_id,
        items=items,
        renderings={"legacy_text": packet.render()},
    )
    return bundle.model_dump(mode="json")
