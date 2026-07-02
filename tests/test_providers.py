"""Native Anthropic + Gemini adapters: mixle's OpenAI-shaped ChatRequest is translated to each provider's own wire
format (system separation, content blocks / contents, tool calling) and their responses read back into an
OpenAI-shaped completion. Driven by httpx.MockTransport -- no network -- capturing the outgoing body to assert the
translation and returning provider-shaped payloads to assert the parse.
"""
import asyncio
import json

import httpx
from mixle_mlops.core.adapters import (
    ChatMessage,
    ChatRequest,
    FunctionCall,
    FunctionDef,
    ToolCall,
    ToolDef,
)
from mixle_mlops.models import AnthropicAdapter, GeminiAdapter, OpenAICompatAdapter, make_adapter

_captured: dict = {}


def _transport(handler):
    def wrapped(request: httpx.Request) -> httpx.Response:
        _captured["url"] = str(request.url)
        _captured["headers"] = dict(request.headers)
        _captured["body"] = json.loads(request.content) if request.content else {}
        return handler(request)
    return httpx.MockTransport(wrapped)


def _sse(events: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


# --- Anthropic -----------------------------------------------------------------------------------------------


def test_anthropic_chat_translates_and_parses():
    def handler(_request):
        return httpx.Response(200, json={
            "id": "msg_1", "model": "claude-x", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hello there"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        })
    adapter = AnthropicAdapter("claude", api_key="k", upstream_model="claude-x", transport=_transport(handler))
    req = ChatRequest(model="claude", messages=[
        ChatMessage(role="system", content="be terse"),
        ChatMessage(role="user", content="hi"),
    ], max_tokens=64)
    out = asyncio.run(adapter.chat(req))

    # translation: system pulled out, max_tokens present, only the user turn in messages
    assert _captured["url"].endswith("/v1/messages")
    assert _captured["headers"]["x-api-key"] == "k"
    assert _captured["headers"]["anthropic-version"]
    assert _captured["body"]["system"] == "be terse"
    assert _captured["body"]["max_tokens"] == 64
    assert [m["role"] for m in _captured["body"]["messages"]] == ["user"]
    # parse
    assert out.choices[0].message.content == "hello there"
    assert out.choices[0].finish_reason == "stop"
    assert out.usage.total_tokens == 10


def test_anthropic_tool_use_becomes_tool_call():
    def handler(_request):
        return httpx.Response(200, json={
            "id": "msg_2", "model": "claude-x", "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {"city": "NYC"}}],
            "usage": {"input_tokens": 5, "output_tokens": 8},
        })
    adapter = AnthropicAdapter("claude", api_key="k", transport=_transport(handler))
    tools = [ToolDef(function=FunctionDef(name="get_weather", parameters={"type": "object"}))]
    req = ChatRequest(model="claude", messages=[ChatMessage(role="user", content="weather?")],
                      tools=tools, tool_choice="auto")
    out = asyncio.run(adapter.chat(req))

    assert _captured["body"]["tools"][0]["name"] == "get_weather"
    assert _captured["body"]["tool_choice"] == {"type": "auto"}
    tc = out.choices[0].message.tool_calls
    assert tc and tc[0].function.name == "get_weather"
    assert json.loads(tc[0].function.arguments) == {"city": "NYC"}
    assert out.choices[0].finish_reason == "tool_calls"


def test_anthropic_tool_result_roundtrips_into_messages():
    def handler(_request):
        return httpx.Response(200, json={"id": "m", "model": "c", "stop_reason": "end_turn",
                                         "content": [{"type": "text", "text": "ok"}], "usage": {}})
    adapter = AnthropicAdapter("claude", api_key="k", transport=_transport(handler))
    req = ChatRequest(model="claude", messages=[
        ChatMessage(role="user", content="weather?"),
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            id="toolu_9", function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'))]),
        ChatMessage(role="tool", tool_call_id="toolu_9", content="sunny, 25C"),
    ])
    asyncio.run(adapter.chat(req))
    msgs = _captured["body"]["messages"]
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[2]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_9", "content": "sunny, 25C"}


def test_anthropic_stream_assembles_text():
    def handler(_request):
        return httpx.Response(200, content=_sse([
            {"type": "message_start"},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]))
    adapter = AnthropicAdapter("claude", api_key="k", transport=_transport(handler))
    req = ChatRequest(model="claude", messages=[ChatMessage(role="user", content="hi")], stream=True)

    async def collect():
        return [c async for c in adapter.stream(req)]
    chunks = asyncio.run(collect())
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == "Hello"
    assert any(c.choices[0].finish_reason == "stop" for c in chunks)


# --- Gemini --------------------------------------------------------------------------------------------------


def test_gemini_chat_translates_and_parses():
    def handler(_request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "hi back"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
        })
    adapter = GeminiAdapter("gem", api_key="g", upstream_model="gemini-2.0", transport=_transport(handler))
    req = ChatRequest(model="gem", messages=[
        ChatMessage(role="system", content="be nice"),
        ChatMessage(role="user", content="hey"),
    ], max_tokens=32)
    out = asyncio.run(adapter.chat(req))

    assert ":generateContent" in _captured["url"] and "gemini-2.0" in _captured["url"]
    assert _captured["headers"]["x-goog-api-key"] == "g"
    assert _captured["body"]["systemInstruction"]["parts"][0]["text"] == "be nice"
    assert _captured["body"]["generationConfig"]["maxOutputTokens"] == 32
    assert _captured["body"]["contents"][0]["role"] == "user"
    assert out.choices[0].message.content == "hi back"
    assert out.choices[0].finish_reason == "stop"
    assert out.usage.total_tokens == 6


def test_gemini_function_call_becomes_tool_call():
    def handler(_request):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "lookup", "args": {"q": "x"}}}]}, "finishReason": "STOP"}]})
    adapter = GeminiAdapter("gem", api_key="g", transport=_transport(handler))
    tools = [ToolDef(function=FunctionDef(name="lookup", parameters={"type": "object"}))]
    req = ChatRequest(model="gem", messages=[ChatMessage(role="user", content="q")], tools=tools, tool_choice="auto")
    out = asyncio.run(adapter.chat(req))

    assert _captured["body"]["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert _captured["body"]["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    tc = out.choices[0].message.tool_calls
    assert tc and tc[0].function.name == "lookup"
    assert json.loads(tc[0].function.arguments) == {"q": "x"}
    assert out.choices[0].finish_reason == "tool_calls"


def test_gemini_stream_assembles_text():
    def handler(_request):
        return httpx.Response(200, content=_sse([
            {"candidates": [{"content": {"parts": [{"text": "Good "}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "day"}]}, "finishReason": "STOP"}]},
        ]))
    adapter = GeminiAdapter("gem", api_key="g", transport=_transport(handler))
    req = ChatRequest(model="gem", messages=[ChatMessage(role="user", content="hi")], stream=True)

    async def collect():
        return [c async for c in adapter.stream(req)]
    chunks = asyncio.run(collect())
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Good day"
    assert any(c.choices[0].finish_reason == "stop" for c in chunks)


# --- factory -------------------------------------------------------------------------------------------------


def test_make_adapter_dispatches_by_provider():
    assert isinstance(make_adapter("m", {"provider": "anthropic", "api_key": "k"}), AnthropicAdapter)
    assert isinstance(make_adapter("m", {"provider": "gemini", "api_key": "k"}), GeminiAdapter)
    assert isinstance(make_adapter("m", {"provider": "google", "api_key": "k"}), GeminiAdapter)
    # no provider (or an unknown one) -> the OpenAI-compatible proxy
    assert isinstance(make_adapter("m", {"base_url": "http://x/v1"}), OpenAICompatAdapter)
    assert isinstance(make_adapter("m", {"provider": "openai", "base_url": "http://x/v1"}), OpenAICompatAdapter)
