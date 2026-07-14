"""An MCP server that exposes the platform's hosted models as MCP **tools**.

For every model in a :class:`~mixle_mlops.core.registry.ModelRegistry` we expose a ``chat__<model>`` tool that
calls the model's adapter (OpenAI-compatible ``chat``), plus a single ``list_models`` tool that enumerates the
catalog with each model's kind + capabilities. Mixle models that advertise distribution capabilities additionally
get a ``score__<model>`` tool (best-effort; declines cleanly via :class:`CapabilityError`).

Transport-agnostic core: :class:`MCPServer` implements the JSON-RPC 2.0 MCP subset
(``initialize`` / ``tools/list`` / ``tools/call``) as ``async def handle(message)``. A stdio ``run_mcp_server``
loop and a ``__main__`` entry are provided so it is runnable on its own.

If the official ``mcp`` python package is importable we still use this JSON-RPC core for the request handling
(it is the protocol wire format the package itself speaks); ``HAVE_OFFICIAL_MCP`` records availability so the
integrator can swap in the SDK's stdio server transport if desired. The minimal fallback needs no extra deps.
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Awaitable, Callable

from ..core.adapters import CapabilityError, ChatMessage, ChatRequest
from ..core.registry import ModelRegistry

try:  # the official SDK is optional; we speak its wire protocol either way
    import mcp as _mcp  # noqa: F401

    HAVE_OFFICIAL_MCP = True
except Exception:  # pragma: no cover - depends on the host env
    HAVE_OFFICIAL_MCP = False

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "mixle-mlops"
SERVER_VERSION = "0.1.0"

# JSON-RPC 2.0 standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class MCPError(Exception):
    """A JSON-RPC error to return to the caller."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# --- tool definitions: name + JSON-schema + an async handler over the registry ---
class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[str]],
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


def _chat_tool(registry: ModelRegistry, model_id: str) -> Tool:
    async def handler(args: dict[str, Any]) -> str:
        message = args.get("message")
        if not isinstance(message, str) or not message:
            raise MCPError(INVALID_PARAMS, "'message' (non-empty string) is required")
        history = args.get("messages")
        messages: list[ChatMessage] = []
        if isinstance(history, list):
            for m in history:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    messages.append(ChatMessage(role=m["role"], content=m["content"]))
        messages.append(ChatMessage(role="user", content=message))
        req = ChatRequest(model=model_id, messages=messages, temperature=args.get("temperature"))
        adapter = registry.get(model_id)
        completion = await adapter.chat(req)
        if not completion.choices:
            return ""
        return completion.choices[0].message.text()

    return Tool(
        name=f"chat__{model_id}",
        description=f"Chat with the hosted model {model_id!r}. Returns the assistant's reply text.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "the user message to send"},
                "messages": {
                    "type": "array",
                    "description": "optional prior chat turns ({role, content})",
                    "items": {"type": "object"},
                },
                "temperature": {"type": "number"},
            },
            "required": ["message"],
        },
        handler=handler,
    )


def _score_tool(registry: ModelRegistry, model_id: str) -> Tool:
    async def handler(args: dict[str, Any]) -> str:
        records = args.get("records")
        if not isinstance(records, list):
            raise MCPError(INVALID_PARAMS, "'records' (array) is required")
        adapter = registry.get(model_id)
        try:
            result = await adapter.score(records)
        except CapabilityError as exc:
            raise MCPError(INVALID_PARAMS, str(exc))
        return json.dumps(result)

    return Tool(
        name=f"score__{model_id}",
        description=f"Score records under the mixle model {model_id!r} (log-density / proper score).",
        input_schema={
            "type": "object",
            "properties": {"records": {"type": "array", "items": {}}},
            "required": ["records"],
        },
        handler=handler,
    )


def _list_models_tool(registry: ModelRegistry) -> Tool:
    async def handler(_args: dict[str, Any]) -> str:
        out = [
            {"id": info.id, "kind": info.kind, "capabilities": info.capabilities}
            for info in registry.list()
        ]
        return json.dumps(out)

    return Tool(
        name="list_models",
        description="List the platform's hosted models with their kind and capabilities.",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


def build_model_tools(registry: ModelRegistry, *, include_physics: bool = True) -> dict[str, Tool]:
    """Build the tool catalog from a registry: ``list_models`` + a ``chat__`` tool per model (+ ``score__`` for
    models advertising the ``score`` capability), plus the physics tools (E4, IC-3) unless ``include_physics``
    is ``False``. Physics tools are merged in only when ``mixle_pde`` is installed (see
    :func:`mixle_mlops.mcp.physics_tools.build_physics_tools`); their absence never breaks this catalog."""
    tools: dict[str, Tool] = {}
    lm = _list_models_tool(registry)
    tools[lm.name] = lm
    for info in registry.list():
        chat = _chat_tool(registry, info.id)
        tools[chat.name] = chat
        if "score" in info.capabilities:
            sc = _score_tool(registry, info.id)
            tools[sc.name] = sc
    if include_physics:
        from .physics_tools import build_physics_tools

        tools.update(build_physics_tools())
    return tools


class MCPServer:
    """Transport-agnostic MCP server. Hold a registry; rebuild the tool catalog per ``tools/list`` so newly
    registered models appear without a restart."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._initialized = False

    def tools(self) -> dict[str, Tool]:
        return build_model_tools(self.registry)

    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC request/notification. Returns a JSON-RPC response dict, or ``None`` for a
        notification (no ``id``)."""
        if message.get("jsonrpc") != "2.0":
            return self._error(message.get("id"), INVALID_REQUEST, "expected jsonrpc 2.0")
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message
        params = message.get("params") or {}

        try:
            result = await self._dispatch(method, params)
        except MCPError as exc:
            if is_notification:
                return None
            return self._error(msg_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # pragma: no cover - defensive
            if is_notification:
                return None
            return self._error(msg_id, INTERNAL_ERROR, str(exc))

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    async def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            self._initialized = True
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        if method in ("notifications/initialized", "initialized"):
            return {}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [t.spec() for t in self.tools().values()]}
        if method == "tools/call":
            return await self._call_tool(params)
        raise MCPError(METHOD_NOT_FOUND, f"method {method!r} not found")

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        tools = self.tools()
        tool = tools.get(name)
        if tool is None:
            raise MCPError(INVALID_PARAMS, f"unknown tool {name!r}; available: {sorted(tools)}")
        try:
            text = await tool.handler(args)
        except MCPError:
            raise
        except Exception as exc:
            # MCP convention: tool execution failures are reported in-band with isError, not as a protocol error
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        # most handlers return text already; the physics tools (E4) return a JSON-able dict directly so
        # ToolRegistry.dispatch can hand callers structured data (e.g. the `prior_dominated` flag) without a
        # round-trip through json.loads — encode it here for the wire protocol, which is text-only.
        if not isinstance(text, str):
            text = json.dumps(text)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": err}


async def serve_stdio(server: MCPServer, reader=None, writer=None) -> None:
    """Run the server over newline-delimited JSON-RPC on stdio (the common MCP local transport)."""
    loop = asyncio.get_event_loop()

    def _readline() -> str:
        stream = reader if reader is not None else sys.stdin
        return stream.readline()

    def _write(text: str) -> None:
        stream = writer if writer is not None else sys.stdout
        stream.write(text + "\n")
        stream.flush()

    while True:
        line = await loop.run_in_executor(None, _readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write(json.dumps(MCPServer._error(None, PARSE_ERROR, "invalid JSON")))
            continue
        response = await server.handle(message)
        if response is not None:
            _write(json.dumps(response))


def run_mcp_server(registry: ModelRegistry) -> None:
    """Runnable entry point: serve the MCP tools for ``registry`` over stdio until EOF."""
    server = MCPServer(registry)
    asyncio.run(serve_stdio(server))


def _demo_registry() -> ModelRegistry:
    from ..models import EchoAdapter

    registry = ModelRegistry()
    registry.register(EchoAdapter("echo"))
    return registry


if __name__ == "__main__":  # pragma: no cover
    run_mcp_server(_demo_registry())
