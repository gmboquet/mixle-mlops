"""Proxy adapter to any OpenAI-compatible chat server: Ollama (:11434/v1), vLLM, llama.cpp, TGI, or a hosted
API. This is how open LLMs (Llama, DeepSeek, ...) are hosted — through their standard server, best-practice."""
from __future__ import annotations

import json
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
    ModelAdapter,
    ToolCall,
    ToolCallDelta,
    Usage,
)


class OpenAICompatAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name: str, *, base_url: str, api_key: str = "", upstream_model: str | None = None,
                 timeout: float = 600.0, transport: httpx.BaseTransport | None = None):
        self._name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.upstream_model = upstream_model or name
        self.timeout = timeout
        self._transport = transport  # injectable for offline tests (httpx.MockTransport); None = real network

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    @property
    def name(self) -> str:
        return self._name

    def _payload(self, req: ChatRequest, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.upstream_model,
            "messages": [m.model_dump(exclude_none=True) for m in req.messages],
            "stream": stream,
        }
        for k in ("temperature", "max_tokens", "top_p"):
            v = getattr(req, k)
            if v is not None:
                body[k] = v
        if req.tools is not None:                       # forward OpenAI tool declarations to the backend
            body["tools"] = [t.model_dump(exclude_none=True) for t in req.tools]
        if req.tool_choice is not None:
            body["tool_choice"] = req.tool_choice
        body.update(req.extra)
        return body

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        async with self._client() as client:
            r = await client.post(f"{self.base_url}/chat/completions",
                                  json=self._payload(req, False), headers=self._headers())
            r.raise_for_status()
            data = r.json()
        choices = data.get("choices") or [{}]
        msg = choices[0].get("message", {}) or {}
        u = data.get("usage", {}) or {}
        tool_calls = None
        if msg.get("tool_calls"):                       # the model wants tools executed
            tool_calls = []
            for tc in msg["tool_calls"]:
                try:
                    tool_calls.append(ToolCall.model_validate(tc))
                except Exception:
                    continue
            tool_calls = tool_calls or None
        finish = choices[0].get("finish_reason") or ("tool_calls" if tool_calls else "stop")
        return ChatCompletion(
            id=data.get("id", "chatcmpl-proxy"),
            model=data.get("model", req.model),
            choices=[ChatChoice(
                message=ChatMessage(role="assistant", content=msg.get("content") or "", tool_calls=tool_calls),
                finish_reason=finish,
            )],
            usage=Usage(**{k: int(u.get(k, 0)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        async with self._client() as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions",
                                     json=self._payload(req, True), headers=self._headers()) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    out = []
                    for ch in obj.get("choices", []):
                        delta = ch.get("delta", {}) or {}
                        tcds = None
                        if delta.get("tool_calls"):     # reassemble index-keyed tool-call fragments
                            tcds = []
                            for t in delta["tool_calls"]:
                                try:
                                    tcds.append(ToolCallDelta.model_validate(t))
                                except Exception:
                                    continue
                            tcds = tcds or None
                        out.append(ChatChunkChoice(
                            index=ch.get("index", 0),
                            delta=ChoiceDelta(role=delta.get("role"), content=delta.get("content"), tool_calls=tcds),
                            finish_reason=ch.get("finish_reason"),
                        ))
                    yield ChatCompletionChunk(id=obj.get("id", "chatcmpl-proxy"),
                                              model=obj.get("model", req.model), choices=out)

    async def list_upstream_models(self) -> list[str]:
        """Discover model ids the backend serves (GET /models)."""
        try:
            async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
                r = await client.get(f"{self.base_url}/models", headers=self._headers())
                r.raise_for_status()
                return [m["id"] for m in r.json().get("data", []) if "id" in m]
        except Exception:
            return []
