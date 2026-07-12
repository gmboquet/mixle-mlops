"""Tool calling + the server-side agentic loop: schema round-trip, the tool registry over the model catalog,
the reason-act-observe loop with a deterministic tool-calling model, and end-to-end agent mode through the chat
route."""
import asyncio

import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    FunctionCall,
    FunctionDef,
    ModelAdapter,
    ToolCall,
    ToolDef,
)
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.agent_loop import run_agent_loop
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_mlops.models import EchoAdapter


class FakeToolAdapter(ModelAdapter):
    """A model that asks for one tool (default: list_models), then answers once it sees the tool result."""
    kind = "llm"

    def __init__(self, name="toolbot", tool="list_models", arguments="{}"):
        self._name = name
        self._tool = tool
        self._arguments = arguments

    @property
    def name(self):
        return self._name

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        tool_msgs = [m for m in req.messages if m.role == "tool"]
        if tool_msgs:                                          # we have a tool result → final answer
            return ChatCompletion(model=req.model, choices=[ChatChoice(
                message=ChatMessage(role="assistant", content=f"final based on: {tool_msgs[-1].text()[:60]}"),
                finish_reason="stop")])
        return ChatCompletion(model=req.model, choices=[ChatChoice(   # first turn → request a tool
            message=ChatMessage(role="assistant", content="",
                                tool_calls=[ToolCall(function=FunctionCall(name=self._tool, arguments=self._arguments))]),
            finish_reason="tool_calls")])

    async def stream(self, req: ChatRequest):
        completion = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(role="assistant", content=completion.choices[0].message.text()),
            finish_reason="stop")])


def test_tool_schema_roundtrip():
    td = ToolDef(function=FunctionDef(name="f", description="d", parameters={"type": "object", "properties": {}}))
    assert td.model_dump(exclude_none=True)["function"]["name"] == "f"
    tc = ToolCall(function=FunctionCall(name="f", arguments='{"x":1}'))
    assert tc.id.startswith("call_") and tc.type == "function"
    # a tool-call assistant message has no content but serializes its calls
    m = ChatMessage(role="assistant", tool_calls=[tc])
    assert m.text() == "" and m.model_dump(exclude_none=True)["tool_calls"][0]["function"]["name"] == "f"


def test_tool_registry_catalog_and_dispatch():
    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    tools = ToolRegistry(reg, user_id="u1")
    names = {t.function.name for t in tools.specs()}
    assert {"list_models", "chat__echo", "rag_search", "mixle_predict", "mixle_decide"} <= names
    out = asyncio.run(tools.dispatch("list_models", {}))
    assert "echo" in out                                       # MCP list_models returns JSON text listing echo
    assert "error" in asyncio.run(tools.dispatch("nope", {}))  # unknown tool → in-band error, no crash


def test_rag_tool_absent_without_user():
    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    tools = ToolRegistry(reg, user_id=None)
    assert "rag_search" not in {t.function.name for t in tools.specs()}


def test_agent_loop_executes_tool_then_answers():
    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    tools = ToolRegistry(reg, user_id=None)
    req = ChatRequest(model="toolbot", messages=[ChatMessage(role="user", content="who is hosted?")])
    completion = asyncio.run(run_agent_loop(FakeToolAdapter(), req, tools, max_iters=4))
    assert "final based on" in completion.choices[0].message.text()
    assert "echo" in completion.choices[0].message.text()      # the executed list_models result reached the model


def test_agent_loop_iteration_guard():
    """A model that always asks for a tool must still terminate with a tool-free final answer."""
    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    tools = ToolRegistry(reg, user_id=None)

    class AlwaysTool(FakeToolAdapter):
        async def chat(self, req):
            return ChatCompletion(model=req.model, choices=[ChatChoice(
                message=ChatMessage(role="assistant", content="ans",
                                    tool_calls=[ToolCall(function=FunctionCall(name="list_models"))]),
                finish_reason="tool_calls")]) if any(m.role != "tool" for m in req.messages[-1:]) or True \
                else None

    # force the final tool-free call to return a plain answer
    class AlwaysThenAnswer(ModelAdapter):
        kind = "llm"
        name = "loop"

        async def chat(self, req):
            if req.tool_choice == "none":
                return ChatCompletion(model=req.model, choices=[ChatChoice(
                    message=ChatMessage(role="assistant", content="forced final"), finish_reason="stop")])
            return ChatCompletion(model=req.model, choices=[ChatChoice(
                message=ChatMessage(role="assistant", content="",
                                    tool_calls=[ToolCall(function=FunctionCall(name="list_models"))]),
                finish_reason="tool_calls")])

        async def stream(self, req):
            yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
                delta=ChoiceDelta(content="x"), finish_reason="stop")])

    req = ChatRequest(model="loop", messages=[ChatMessage(role="user", content="go")])
    completion = asyncio.run(run_agent_loop(AlwaysThenAnswer(), req, tools, max_iters=3))
    assert completion.choices[0].message.text() == "forced final"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    from mixle_mlops.config import get_settings
    from mixle_mlops.gateway.app import create_app

    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        app.state.registry.register(FakeToolAdapter("toolbot"))   # add a tool-calling model to the live registry
        yield c
    get_settings.cache_clear()
    db._engine = None


def test_agent_mode_end_to_end(client):
    raw = client.post("/auth/signup", json={"email": "agent@t.com", "password": "pw12345"}).json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}"}
    r = client.post("/v1/chat/completions", headers=headers,
                    json={"model": "toolbot", "extra": {"agent": True},
                          "messages": [{"role": "user", "content": "what models are hosted?"}]})
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "final based on" in content and "echo" in content


def test_plain_tools_passthrough_does_not_break(client):
    """A request carrying client-side `tools` (no agent mode) must still succeed (echo ignores them)."""
    raw = client.post("/auth/signup", json={"email": "pt@t.com", "password": "pw12345"}).json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}"}
    r = client.post("/v1/chat/completions", headers=headers,
                    json={"model": "echo",
                          "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
