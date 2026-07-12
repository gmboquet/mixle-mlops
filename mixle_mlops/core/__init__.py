"""Backend-agnostic model contracts and registry exports.

The core namespace exposes the request, response, adapter, capability, and
registry objects shared by gateway routes and provider implementations. It
should stay independent of a particular storage backend or hosted model so
tests can exercise local, mocked, and OpenAI-compatible providers through the
same public contract.
"""
from .adapters import (
    CapabilityError,
    ChatChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    ChatChunkChoice,
    ModelAdapter,
    ModelInfo,
    Usage,
)
from .registry import ModelRegistry

__all__ = [
    "ModelAdapter", "ModelInfo", "ModelRegistry", "ChatMessage", "ChatRequest", "ChatCompletion",
    "ChatChoice", "ChatCompletionChunk", "ChatChunkChoice", "ChoiceDelta", "Usage", "CapabilityError",
]
