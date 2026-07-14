"""The model-callable tool catalog — the bridge that lets a hosted model (mid-conversation, ReAct-style) reach
the platform's own capabilities. It assembles OpenAI tool declarations + dispatch handlers from four sources:

  * the platform's MCP tools (``list_models`` + ``chat__<model>`` + ``score__<model>``) via the MCP schema bridge,
  * the user's RAG store, as a callable ``rag_search`` (so the model can *decide* to retrieve, not always pay for it),
  * mixle distribution/decision capabilities, as ``mixle_predict`` / ``mixle_decide`` over any hosted mixle model.

``specs()`` returns the OpenAI ``tools`` array; ``dispatch(name, args)`` executes a tool call and returns a
JSON-serializable result. Errors are returned in-band (``{"error": ...}``) so a bad tool call never crashes the loop.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

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
                 model_id: str | None = None, verifier: Any = None):
        self.registry = registry
        self.user_id = user_id
        self.model_id = model_id      # the calling chat model's id, stamped onto every recorded step (M4a)
        self.verifier = verifier      # an optional IC-6 Verifier gating each dispatched tool's result
        self.trace_steps: list[TraceStep] = []   # every call this turn, already ExecutionTrace-ready (M4a)
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
                    description="Search the user's uploaded documents and past conversations for relevant context.",
                    parameters={"type": "object", "properties": {
                        "query": {"type": "string", "description": "what to search for"},
                        "k": {"type": "integer", "description": "number of snippets", "default": 5},
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
    async def _rag_search(self, args: dict[str, Any]) -> Any:
        from ..rag.index import retrieve

        query = str(args.get("query", "") or "")
        k = int(args.get("k", 5) or 5)
        snippets = retrieve(self.user_id, query, k=k)
        return {"results": [
            {"text": s.get("text", ""), "source": s.get("source_id"), "namespace": s.get("namespace")}
            for s in snippets
        ]}

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
