"""The model contract. Every backend — a mixle probabilistic model or an open LLM — implements ``ModelAdapter``,
speaking an OpenAI-compatible chat interface plus optional mixle 'distribution/decision' capabilities advertised
through ``capabilities()``. This uniform surface is what lets the gateway host and *compose* both kinds."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


# --- multimodal content parts (a message's content is a string or a list of parts) ---
class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, Any]  # {"url": "data:image/png;base64,..."  or an https URL}


ContentPart = TextPart | ImagePart


# --- tool-calling wire shapes (OpenAI-compatible; all optional → absent means current single-shot behavior) ---
class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON-Schema for the arguments


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class FunctionCall(BaseModel):
    name: str
    arguments: str = ""  # JSON-encoded arguments, per OpenAI


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])
    type: Literal["function"] = "function"
    function: FunctionCall


class ToolCallDelta(BaseModel):
    """A streamed fragment of a tool call; OpenAI keys fragments by ``index`` so they can be reassembled."""

    index: int = 0
    id: str | None = None
    type: str | None = None
    function: dict[str, Any] | None = None  # {"name"?: str, "arguments"?: str-fragment}


class ChatMessage(BaseModel):
    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None  # assistant turn requesting tool execution
    tool_call_id: str | None = None  # a role="tool" result, correlated to the call it answers

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    def images(self) -> list[str]:
        if not isinstance(self.content, list):
            return []
        return [p.image_url.get("url", "") for p in self.content if isinstance(p, ImagePart)]


class ChatRequest(BaseModel):
    model: str = ""  # empty → gateway uses the configured default model
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    user: str | None = None
    tools: list[ToolDef] | None = None  # OpenAI tool/function declarations
    tool_choice: str | dict[str, Any] | None = None  # "auto"|"none"|"required"|{"type":"function",...}
    max_tool_iters: int | None = None  # gateway agentic-loop guard (stripped before upstream)
    extra: dict[str, Any] = Field(default_factory=dict)  # passthrough for backend-specific options


# --- OpenAI-compatible response shapes ---
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class ChoiceDelta(BaseModel):
    role: Role | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta = Field(default_factory=ChoiceDelta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatChunkChoice] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mixle-mlops"
    kind: Literal["llm", "mixle", "composite"] = "llm"
    capabilities: list[str] = Field(default_factory=list)
    # estimation certificate + calibration, when the hosted model is a fitted mixle model (else None).
    # These make the served manifest self-describing: HOW it was estimated (guarantee ladder + why-not-ADAM)
    # and WHETHER its uncertainty was checked honest, so a consumer can gate on provenance, not just id.
    certificate: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None


class CapabilityError(Exception):
    """Raised when a model is asked for a query it does not implement (→ HTTP 422 at the gateway)."""

    def __init__(self, model: str, capability: str):
        super().__init__(f"model {model!r} does not support {capability!r}")
        self.model = model
        self.capability = capability


class ModelAdapter(ABC):
    """Uniform interface over a hosted model.

    LLM backends implement ``stream`` (and may override ``chat``). mixle models additionally implement the
    distribution/decision methods (``predict``/``decide``/``score``/``latent``), advertised via
    ``capabilities()`` so the gateway only routes a query to a model that supports it.
    """

    kind: str = "llm"

    @property
    @abstractmethod
    def name(self) -> str: ...

    def info(self) -> ModelInfo:
        return ModelInfo(id=self.name, kind=self.kind, capabilities=sorted(self.capabilities()))

    def capabilities(self) -> set[str]:
        return {"chat"}

    @abstractmethod
    def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Async-iterate OpenAI-compatible streaming chunks for the request."""
        ...

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        """Non-streaming default: collect the stream into one completion. Override for a native non-stream call."""
        parts: list[str] = []
        async for chunk in self.stream(req):
            for ch in chunk.choices:
                if ch.delta.content:
                    parts.append(ch.delta.content)
        return ChatCompletion(
            model=req.model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content="".join(parts)))],
        )

    # --- optional mixle distribution/decision capabilities (default: unsupported) ---
    async def predict(self, records: list[Any], **opts: Any) -> Any:
        raise CapabilityError(self.name, "predict")

    async def decide(self, records: list[Any], **opts: Any) -> Any:
        raise CapabilityError(self.name, "decide")

    async def score(self, records: list[Any], **opts: Any) -> Any:
        raise CapabilityError(self.name, "score")

    async def latent(self, records: list[Any], **opts: Any) -> Any:
        raise CapabilityError(self.name, "latent")

    async def escalation_decision(self, req: "ChatRequest") -> dict[str, Any] | None:
        """A model that knows its own confidence can drive the cascade directly, instead of best-of-N voting.

        Return ``{"escalate": bool, "answer": str | None, "confidence": float | None}`` to let the cascade router
        use this model's *own* calibrated escalate signal (e.g. a conformal/density gate), or ``None`` to defer
        to the generic self-consistency router. Default: ``None`` (no principled signal)."""
        return None
