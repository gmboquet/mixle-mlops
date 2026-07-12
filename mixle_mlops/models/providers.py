"""Native adapters for the frontier providers whose APIs are *not* OpenAI-compatible: Anthropic (``/v1/messages``)
and Google Gemini (``generateContent``). ``OpenAICompatAdapter`` already reaches anything that speaks the OpenAI
chat schema (Ollama, vLLM, OpenAI, most proxies); these two translate mixle's OpenAI-shaped ``ChatRequest`` to and
from each provider's own wire format -- system-prompt separation, content blocks, and tool-calling included -- so a
native-provider model is a first-class ``ModelAdapter`` the cascade/MoA/best-of-N bridge can compose like any other.

Both are prompt-level integrations: hosted APIs don't expose logits, so the logit-level bridge (token PoE, grammar
masking) stays local-only (see ``models/local_engine.py``) -- the deliberate two-tier contract. A small ``transport``
seam lets tests drive these with ``httpx.MockTransport`` instead of the network.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import httpx

from ..core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    FunctionCall,
    ModelAdapter,
    ToolCall,
    ToolCallDelta,
    Usage,
)

ANTHROPIC_VERSION = "2023-06-01"


def _finish(reason: str | None, tool_calls: Any) -> str:
    if tool_calls:
        return "tool_calls"
    return reason or "stop"


# =============================================================================================================
# Anthropic  (POST {base}/v1/messages)
# =============================================================================================================


class AnthropicAdapter(ModelAdapter):
    """Anthropic Messages API as a ``ModelAdapter``. Pulls system turns into the top-level ``system`` field, maps
    assistant tool calls to ``tool_use`` blocks and ``role="tool"`` results to ``tool_result`` blocks, and reads the
    ``content`` block list (text + ``tool_use``) back into an OpenAI-shaped completion."""

    kind = "llm"

    def __init__(
        self,
        name: str,
        *,
        api_key: str = "",
        upstream_model: str | None = None,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 4096,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._name = name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.upstream_model = upstream_model or name
        self.base_url = base_url.rstrip("/")
        self.default_max_tokens = int(max_tokens)
        self.timeout = timeout
        self._transport = transport

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> set[str]:
        return {"chat", "tools"}

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _messages(self, req: ChatRequest) -> tuple[str | None, list[dict[str, Any]]]:
        """Split req.messages into (system prompt, Anthropic message list). Assistant tool calls -> tool_use blocks;
        role='tool' turns -> a user turn with a tool_result block correlated by tool_use_id."""
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        for m in req.messages:
            if m.role == "system":
                if m.text():
                    system_parts.append(m.text())
                continue
            if m.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.text()}],
                })
                continue
            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.text():
                    blocks.append({"type": "text", "text": m.text()})
                for tc in m.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
                out.append({"role": "assistant", "content": blocks})
                continue
            content: list[dict[str, Any]] = []
            if isinstance(m.content, str):
                content.append({"type": "text", "text": m.content})
            else:
                if m.text():
                    content.append({"type": "text", "text": m.text()})
                for url in m.images():  # data:...;base64,<data> or an https URL
                    content.append(_anthropic_image_block(url))
            out.append({"role": m.role, "content": content or [{"type": "text", "text": ""}]})
        return ("\n\n".join(system_parts) or None), out

    def _payload(self, req: ChatRequest, stream: bool) -> dict[str, Any]:
        system, messages = self._messages(req)
        body: dict[str, Any] = {
            "model": self.upstream_model,
            "messages": messages,
            "max_tokens": req.max_tokens or self.default_max_tokens,
            "stream": stream,
        }
        if system:
            body["system"] = system
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.tools:
            body["tools"] = [
                {"name": t.function.name, "description": t.function.description or "",
                 "input_schema": t.function.parameters or {"type": "object", "properties": {}}}
                for t in req.tools
            ]
            if req.tool_choice is not None:
                body["tool_choice"] = _anthropic_tool_choice(req.tool_choice)
        body.update({k: v for k, v in req.extra.items() if k not in body})
        return body

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        async with self._client() as client:
            r = await client.post(f"{self.base_url}/v1/messages", json=self._payload(req, False),
                                  headers=self._headers())
            r.raise_for_status()
            data = r.json()
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []) or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id") or "call_anthropic",
                    function=FunctionCall(name=block.get("name", ""),
                                          arguments=json.dumps(block.get("input", {}))),
                ))
        u = data.get("usage", {}) or {}
        pt, ct = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
        return ChatCompletion(
            id=data.get("id", "chatcmpl-anthropic"),
            model=data.get("model", req.model),
            choices=[ChatChoice(
                message=ChatMessage(role="assistant", content="".join(text_parts),
                                    tool_calls=tool_calls or None),
                finish_reason=_finish(_ANTHROPIC_STOP.get(data.get("stop_reason", "")), tool_calls),
            )],
            usage=Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        cid = "chatcmpl-anthropic"
        async with self._client() as client:
            async with client.stream("POST", f"{self.base_url}/v1/messages",
                                     json=self._payload(req, True), headers=self._headers()) as r:
                r.raise_for_status()
                tool_index: dict[int, bool] = {}
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[len("data:"):].strip())
                    except json.JSONDecodeError:
                        continue
                    etype = ev.get("type")
                    if etype == "content_block_start":
                        block = ev.get("content_block", {}) or {}
                        if block.get("type") == "tool_use":
                            idx = ev.get("index", 0)
                            tool_index[idx] = True
                            yield _chunk(cid, req.model, tool_calls=[ToolCallDelta(
                                index=idx, id=block.get("id"), type="function",
                                function={"name": block.get("name", ""), "arguments": ""})])
                    elif etype == "content_block_delta":
                        delta = ev.get("delta", {}) or {}
                        if delta.get("type") == "text_delta":
                            yield _chunk(cid, req.model, content=delta.get("text", ""))
                        elif delta.get("type") == "input_json_delta":
                            yield _chunk(cid, req.model, tool_calls=[ToolCallDelta(
                                index=ev.get("index", 0),
                                function={"arguments": delta.get("partial_json", "")})])
                    elif etype == "message_delta":
                        stop = (ev.get("delta", {}) or {}).get("stop_reason")
                        if stop:
                            yield _chunk(cid, req.model,
                                         finish_reason=_finish(_ANTHROPIC_STOP.get(stop), tool_index))
                    elif etype == "message_stop":
                        break


_ANTHROPIC_STOP = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop", "tool_use": "tool_calls"}


def _anthropic_tool_choice(choice: Any) -> dict[str, Any]:
    if choice == "auto":
        return {"type": "auto"}
    if choice in ("required", "any"):
        return {"type": "any"}
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "tool", "name": choice.get("function", {}).get("name", "")}
    return {"type": "auto"}


def _anthropic_image_block(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, b64 = url.partition(",")
        media = header[len("data:"):].split(";")[0] or "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}}
    return {"type": "image", "source": {"type": "url", "url": url}}


# =============================================================================================================
# Google Gemini  (POST {base}/v1beta/models/{model}:generateContent)
# =============================================================================================================


class GeminiAdapter(ModelAdapter):
    """Google Gemini generateContent as a ``ModelAdapter``. Maps assistant/user roles to Gemini's ``model``/``user``
    contents, system turns to ``systemInstruction``, tool calls to ``functionCall`` parts and ``role='tool'`` results
    to ``functionResponse`` parts, and reads ``candidates[0].content.parts`` (text + ``functionCall``) back out."""

    kind = "llm"

    def __init__(
        self,
        name: str,
        *,
        api_key: str = "",
        upstream_model: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._name = name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        self.upstream_model = upstream_model or name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> set[str]:
        return {"chat", "tools"}

    def _headers(self) -> dict[str, str]:
        return {"content-type": "application/json", "x-goog-api-key": self.api_key}

    def _contents(self, req: ChatRequest) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for m in req.messages:
            if m.role == "system":
                if m.text():
                    system_parts.append(m.text())
                continue
            if m.role == "tool":
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": m.name or m.tool_call_id or "tool",
                    "response": {"result": m.text()}}}]})
                continue
            if m.role == "assistant" and m.tool_calls:
                parts: list[dict[str, Any]] = []
                if m.text():
                    parts.append({"text": m.text()})
                for tc in m.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    parts.append({"functionCall": {"name": tc.function.name, "args": args}})
                contents.append({"role": "model", "parts": parts})
                continue
            role = "model" if m.role == "assistant" else "user"
            parts = [{"text": m.text()}] if m.text() or not m.images() else []
            for url in m.images():
                parts.append(_gemini_image_part(url))
            contents.append({"role": role, "parts": parts or [{"text": ""}]})
        system = {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
        return system, contents

    def _payload(self, req: ChatRequest) -> dict[str, Any]:
        system, contents = self._contents(req)
        gen: dict[str, Any] = {}
        if req.temperature is not None:
            gen["temperature"] = req.temperature
        if req.max_tokens is not None:
            gen["maxOutputTokens"] = req.max_tokens
        if req.top_p is not None:
            gen["topP"] = req.top_p
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = system
        if gen:
            body["generationConfig"] = gen
        if req.tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t.function.name, "description": t.function.description or "",
                 "parameters": t.function.parameters or {"type": "object", "properties": {}}}
                for t in req.tools]}]
            if req.tool_choice is not None:
                body["toolConfig"] = {"functionCallingConfig": _gemini_tool_mode(req.tool_choice)}
        body.update({k: v for k, v in req.extra.items() if k not in body})
        return body

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    def _parts_to_message(self, parts: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]:
        text_parts, tool_calls = [], []
        for p in parts or []:
            if "text" in p:
                text_parts.append(p["text"])
            elif "functionCall" in p:
                fc = p["functionCall"]
                tool_calls.append(ToolCall(function=FunctionCall(
                    name=fc.get("name", ""), arguments=json.dumps(fc.get("args", {})))))
        return "".join(text_parts), tool_calls

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        url = f"{self.base_url}/v1beta/models/{self.upstream_model}:generateContent"
        async with self._client() as client:
            r = await client.post(url, json=self._payload(req), headers=self._headers())
            r.raise_for_status()
            data = r.json()
        cand = (data.get("candidates") or [{}])[0]
        text, tool_calls = self._parts_to_message((cand.get("content", {}) or {}).get("parts", []))
        u = data.get("usageMetadata", {}) or {}
        pt, ct = int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))
        return ChatCompletion(
            id=data.get("responseId", "chatcmpl-gemini"),
            model=req.model,
            choices=[ChatChoice(
                message=ChatMessage(role="assistant", content=text, tool_calls=tool_calls or None),
                finish_reason=_finish(_GEMINI_FINISH.get(cand.get("finishReason", "")), tool_calls),
            )],
            usage=Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=int(u.get("totalTokenCount", pt + ct))),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        cid = "chatcmpl-gemini"
        url = f"{self.base_url}/v1beta/models/{self.upstream_model}:streamGenerateContent"
        async with self._client() as client:
            async with client.stream("POST", url, params={"alt": "sse"},
                                     json=self._payload(req), headers=self._headers()) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        obj = json.loads(line[len("data:"):].strip())
                    except json.JSONDecodeError:
                        continue
                    cand = (obj.get("candidates") or [{}])[0]
                    text, tool_calls = self._parts_to_message((cand.get("content", {}) or {}).get("parts", []))
                    tcds = [ToolCallDelta(index=i, id=tc.id, type="function",
                                          function={"name": tc.function.name, "arguments": tc.function.arguments})
                            for i, tc in enumerate(tool_calls)] or None
                    finish = cand.get("finishReason")
                    if text or tcds:
                        yield _chunk(cid, req.model, content=text or None, tool_calls=tcds)
                    if finish:
                        yield _chunk(cid, req.model, finish_reason=_finish(_GEMINI_FINISH.get(finish), tool_calls))


_GEMINI_FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "stop"}


def _gemini_tool_mode(choice: Any) -> dict[str, Any]:
    if choice == "none":
        return {"mode": "NONE"}
    if choice in ("required", "any"):
        return {"mode": "ANY"}
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"mode": "ANY", "allowedFunctionNames": [choice.get("function", {}).get("name", "")]}
    return {"mode": "AUTO"}


def _gemini_image_part(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, b64 = url.partition(",")
        media = header[len("data:"):].split(";")[0] or "image/png"
        return {"inlineData": {"mimeType": media, "data": b64}}
    return {"fileData": {"fileUri": url}}


# --- shared streaming-chunk helper ---------------------------------------------------------------------------


def _chunk(cid: str, model: str, *, content: str | None = None,
           tool_calls: list[ToolCallDelta] | None = None, finish_reason: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=cid, model=model,
        choices=[ChatChunkChoice(delta=ChoiceDelta(content=content, tool_calls=tool_calls),
                                 finish_reason=finish_reason)],
    )


# =============================================================================================================
# provider factory
# =============================================================================================================

_PROVIDERS = {"anthropic": AnthropicAdapter, "gemini": GeminiAdapter, "google": GeminiAdapter}


def make_adapter(model_id: str, backend: dict[str, Any], *, default_base_url: str = "",
                 default_api_key: str = "") -> ModelAdapter:
    """Build the right ``ModelAdapter`` for a configured backend. ``backend['provider']`` selects a native provider
    (``anthropic``/``gemini``); anything else (or absent) falls back to ``OpenAICompatAdapter``. Native hosted
    models, OpenAI, Ollama, and vLLM all register into one gateway from the same ``MIXLE_LLM_BACKENDS`` config
    block."""
    from .openai_compat import OpenAICompatAdapter

    provider = (backend.get("provider") or "openai").lower()
    api_key = backend.get("api_key", default_api_key)
    upstream = backend.get("upstream_model")
    if provider in _PROVIDERS:
        cls = _PROVIDERS[provider]
        kwargs: dict[str, Any] = {"api_key": api_key, "upstream_model": upstream}
        if backend.get("base_url"):
            kwargs["base_url"] = backend["base_url"]
        if provider == "anthropic" and backend.get("max_tokens"):
            kwargs["max_tokens"] = int(backend["max_tokens"])
        return cls(model_id, **kwargs)
    return OpenAICompatAdapter(model_id, base_url=backend.get("base_url", default_base_url),
                               api_key=api_key, upstream_model=upstream)
