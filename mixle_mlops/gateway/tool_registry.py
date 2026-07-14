"""The model-callable tool catalog — the bridge that lets a hosted model (mid-conversation, ReAct-style) reach
the platform's own capabilities. It assembles OpenAI tool declarations + dispatch handlers from four sources:

  * the platform's MCP tools (``list_models`` + ``chat__<model>`` + ``score__<model>``) via the MCP schema bridge,
  * the user's RAG store, as a callable ``rag_search`` (so the model can *decide* to retrieve, not always pay for it),
  * mixle distribution/decision capabilities, as ``mixle_predict`` / ``mixle_decide`` over any hosted mixle model.

``specs()`` returns the OpenAI ``tools`` array; ``dispatch(name, args)`` executes a tool call and returns a
JSON-serializable result. Errors are returned in-band (``{"error": ...}``) so a bad tool call never crashes the loop.

``rag_search`` is federated across two source-native stores that share nothing but the tool call: the user's
document/conversation embeddings (``mixle_mlops.rag.index.retrieve``, tenant + free text) and, when the caller
wires one in (``field_store=``), a location-indexed physics volume (a ``mixle_pde.reasoning.SpatialFieldStore``,
or any object exposing the same ``.store()`` -> ``CrossModalStore`` shape). Each group is queried with its own
native signature -- text for documents, a location for the field -- and results come back as typed, source-native
records (``source_kind``, ``score``, ``artifact_ref``, ``selector``, ``provenance``); a downstream normalizer
(mixle-knowledge, IC-13) is what turns these into one ranked bundle, not this module.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

import numpy as np
from mixle.task.replay import TraceStep

from ..core.adapters import FunctionDef, ToolDef
from ..core.registry import ModelRegistry
from ..mcp.schema_bridge import mcp_tool_to_tooldef
from ..mcp.server import build_model_tools
from .trace_capture import record_tool_call

Handler = Callable[[dict[str, Any]], Awaitable[Any]]


class ToolRegistry:
    def __init__(self, registry: ModelRegistry, *, user_id: str | None = None,
                 names: list[str] | None = None, include_mcp: bool = True,
                 include_rag: bool = True, include_mixle: bool = True,
                 model_id: str | None = None, verifier: Any = None,
                 field_store: Any = None):
        self.registry = registry
        self.user_id = user_id
        self.model_id = model_id      # the calling chat model's id, stamped onto every recorded step (M4a)
        self.verifier = verifier      # an optional IC-6 Verifier gating each dispatched tool's result
        self.trace_steps: list[TraceStep] = []   # every call this turn, already ExecutionTrace-ready (M4a)
        # optional location-indexed physics volume for federated `rag_search` (E8): a `SpatialFieldStore`
        # (or bare `CrossModalStore`), or a zero-arg factory building one lazily on first use -- either way
        # it is *reused* across calls, never rebuilt per query.
        self._field_store_source = field_store
        self._field_store_cache: Any = None
        self._whitelist = set(names) if names else None      # optional restriction of the exposed catalog
        self._defs: dict[str, ToolDef] = {}
        self._handlers: dict[str, Handler] = {}
        self._build(include_mcp, include_rag, include_mixle)

    # --- assembly ---
    def _add(self, tooldef: ToolDef, handler: Handler) -> None:
        name = tooldef.function.name
        if self._whitelist is not None and name not in self._whitelist:
            return
        self._defs[name] = tooldef
        self._handlers[name] = handler

    def _build(self, include_mcp: bool, include_rag: bool, include_mixle: bool) -> None:
        if include_mcp:
            for tool in build_model_tools(self.registry).values():
                self._add(mcp_tool_to_tooldef(tool), tool.handler)   # MCP handler(args) -> awaitable[str]
        if include_rag and self.user_id:
            self._add(
                ToolDef(function=FunctionDef(
                    name="rag_search",
                    description="Search the user's uploaded documents and past conversations for relevant context. "
                                "If a `location` is given and a physics field volume is registered, also retrieves "
                                "the nearby field tiles (raw sub-volume evidence, not a lossy embedding).",
                    parameters={"type": "object", "properties": {
                        "query": {"type": "string", "description": "what to search for"},
                        "k": {"type": "integer", "description": "number of snippets/tiles per source", "default": 5},
                        "location": {"type": "array", "items": {"type": "number"},
                                     "description": "optional coordinate anchoring a nearby-field-tile lookup "
                                                     "(units of the registered SpatialFieldStore, not lon/lat)"},
                        "bbox": {"type": "array", "items": {"type": "number"},
                                 "description": "optional [minx,miny,maxx,maxy] geoscience filter for the "
                                                 "document search (independent of `location`)"},
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
    def _resolve_field_store(self) -> Any:
        """Reuse the wired-in field store, building it once if a lazy factory was supplied."""
        if self._field_store_cache is not None:
            return self._field_store_cache
        source = self._field_store_source
        if source is None:
            return None
        store = source() if callable(source) and not hasattr(source, "store") else source
        self._field_store_cache = store
        return store

    def _document_records(self, query: str, k: int, bbox: list[float] | None) -> list[dict[str, Any]]:
        """D3 document retrieval: tenant + free text drive this store; `bbox` never touches the field store."""
        import inspect

        from ..rag.index import retrieve

        kwargs: dict[str, Any] = {}
        if bbox is not None and "filters" in inspect.signature(retrieve).parameters:
            kwargs["filters"] = {"bbox": tuple(bbox)}
        hits = retrieve(self.user_id, query, k=k, **kwargs)
        records = []
        for h in hits:
            meta = h.get("meta") or {}
            records.append({
                "source_kind": "document",
                "source_id": h.get("source_id"),
                "score": float(h.get("score", 0.0)),
                "artifact_ref": meta.get("artifact_ref"),
                "selector": meta.get("selector"),
                "provenance": [{"namespace": h.get("namespace"), "id": h.get("id"), "meta": meta}],
                "text": h.get("text", ""),
            })
        return records

    def _field_tile_records(self, location: list[float], k: int) -> list[dict[str, Any]]:
        """Location-anchored physics retrieval over the registered `SpatialFieldStore`.

        `CrossModalStore.retrieve` returns bare integer tile indices -- an index is a router key, never
        evidence -- so every index is dereferenced against `.payloads` (the tile's member cells) and
        `.keys` (the tile centroid) before it leaves this function as a record.
        """
        field_store = self._resolve_field_store()
        if field_store is None:
            return []
        cross_store = field_store.store() if hasattr(field_store, "store") else field_store
        loc = np.asarray(location, dtype=float)
        records = []
        for index in cross_store.retrieve(loc, k=k):
            index = int(index)
            centroid = np.asarray(cross_store.keys[index], dtype=float)
            members = np.asarray(cross_store.payloads[index]).reshape(-1)
            distance = float(np.linalg.norm(centroid - loc))
            records.append({
                "source_kind": "field_tile",
                "source_index": index,
                "score": 1.0 / (1.0 + distance),
                "artifact_ref": None,
                "selector": {
                    "centroid": centroid.tolist(),
                    "member_cell_indices": members.astype(int).tolist(),
                    "n_members": int(members.size),
                },
                "provenance": [{"store": type(field_store).__name__, "tile_index": index}],
            })
        return records

    async def _rag_search(self, args: dict[str, Any]) -> Any:
        query = str(args.get("query", "") or "")
        k = int(args.get("k", 5) or 5)
        location = args.get("location")
        bbox = args.get("bbox")

        results = self._document_records(query, k, bbox)
        if location is not None:
            results += self._field_tile_records(location, k)
        return {"results": results}

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
