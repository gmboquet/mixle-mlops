"""Model adapters: the echo adapter, the OpenAI-compatible LLM proxy, the native Anthropic/Gemini providers, the
mixle model, and the distilled task cascade."""
from .echo import EchoAdapter
from .openai_compat import OpenAICompatAdapter
from .providers import AnthropicAdapter, GeminiAdapter, make_adapter
from .task_cascade import TaskCascadeAdapter, register_demo_task_model

__all__ = [
    "AnthropicAdapter",
    "EchoAdapter",
    "GeminiAdapter",
    "OpenAICompatAdapter",
    "TaskCascadeAdapter",
    "make_adapter",
    "register_demo_task_model",
]
