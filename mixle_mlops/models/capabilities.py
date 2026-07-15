"""Capability-routed model selection -- the 'flexible pipeline' seam.

Pipeline code should ask for a *capability* (``text``, ``vision``, ``embedding``) and get an adapter,
never hard-code a provider or model name. That's what makes the model layer swappable: route
``text`` to DeepSeek and ``vision`` to a hosted Qwen-VL today, swap either for a local model or a
different provider tomorrow, without touching a single caller.

This sits on top of the machinery that already exists -- ``Settings.llm_backends`` (the per-model
provider/base_url/api_key config block) and :func:`mixle_mlops.models.make_adapter` (which returns an
:class:`~mixle_mlops.models.openai_compat.OpenAICompatAdapter`, forwarding image content parts to any
OpenAI-compatible vision backend, or a native Anthropic/Gemini adapter). It adds only the missing
piece: a *capability -> model* map so callers name what they need, not who provides it.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mixle_mlops.models import make_adapter

__all__ = [
    "TEXT",
    "VISION",
    "EMBEDDING",
    "DEFAULT_CAPABILITY_MODELS",
    "resolve_adapter",
    "resolve_from_settings",
    "providers_from_keys",
    "capability_models_from_env",
]

TEXT = "text"
VISION = "vision"
EMBEDDING = "embedding"

# Sensible defaults for the providers currently wired; every entry is overridable per-call, via the
# MIXLE_CAPABILITY_MODELS env var, or by editing the caller's config -- these are defaults, not locks.
DEFAULT_CAPABILITY_MODELS: dict[str, str] = {
    TEXT: "deepseek-chat",
    VISION: "qwen3-vl-plus",
}

# Known-provider endpoint conveniences: build a llm_backends block straight from a keys dict, so a
# caller with the raw API keys is one call away from a working, capability-routed set of adapters.
# Purely a convenience -- a caller can always build llm_backends by hand for any other provider.
_KNOWN_PROVIDERS: dict[str, dict[str, Any]] = {
    "DEEPSEEK_API_KEY": {
        "models": {"deepseek-chat": "deepseek-chat"},
        "base_url": "https://api.deepseek.com",
    },
    "QWEN_API_KEY": {
        "models": {"qwen3-vl-plus": "qwen3-vl-plus", "qwen3-vl-flash": "qwen3-vl-flash", "qwen3-max": "qwen3-max"},
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
}


def providers_from_keys(keys: dict[str, str]) -> dict[str, dict[str, str]]:
    """Build a ``llm_backends``-shaped dict from a ``{ENV_NAME: api_key}`` mapping, for the known
    providers (DeepSeek, Qwen/DashScope). Unknown key names are ignored. This is the fast path from
    'I have these API keys' to 'capability routing works'; any other provider can be added to the
    returned dict by hand (or configured via ``MIXLE_LLM_BACKENDS``) in exactly the same shape."""
    backends: dict[str, dict[str, str]] = {}
    for env_name, spec in _KNOWN_PROVIDERS.items():
        api_key = keys.get(env_name)
        if not api_key:
            continue
        for model_id in spec["models"]:
            backends[model_id] = {"provider": "openai", "base_url": spec["base_url"], "api_key": api_key}
    return backends


def capability_models_from_env() -> dict[str, str]:
    """Read a capability->model override map from ``MIXLE_CAPABILITY_MODELS`` (a JSON object), e.g.
    ``MIXLE_CAPABILITY_MODELS='{"vision":"qwen3-vl-flash"}'``. Returns ``{}`` if unset/invalid, so it
    degrades to the defaults rather than raising on a malformed env var."""
    raw = os.environ.get("MIXLE_CAPABILITY_MODELS")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def resolve_adapter(
    capability: str,
    backends: dict[str, dict[str, str]],
    *,
    capability_models: dict[str, str] | None = None,
    default_base_url: str = "",
    default_api_key: str = "",
) -> Any:
    """Return an adapter for the model that serves ``capability``.

    ``backends`` is a ``llm_backends``-shaped map (model_id -> provider/base_url/api_key). The
    capability -> model mapping is :data:`DEFAULT_CAPABILITY_MODELS` overlaid with ``capability_models``.
    Raises :class:`KeyError` if the capability has no configured model, or if the model it routes to has
    no backend and no ``default_base_url`` fallback -- an unroutable capability is a loud error, never a
    silent wrong-model fallback.
    """
    cap_map = {**DEFAULT_CAPABILITY_MODELS, **(capability_models or {})}
    model_id = cap_map.get(capability)
    if not model_id:
        raise KeyError(f"no model configured for capability {capability!r}; known capabilities: {sorted(cap_map)}")
    backend = dict((backends or {}).get(model_id, {}))
    if not backend and not default_base_url:
        raise KeyError(
            f"capability {capability!r} routes to model {model_id!r}, but no backend is configured for it "
            f"(known backends: {sorted(backends or {})}) and no default_base_url was given"
        )
    return make_adapter(model_id, backend, default_base_url=default_base_url, default_api_key=default_api_key)


def resolve_from_settings(capability: str, *, settings: Any = None, capability_models: dict[str, str] | None = None) -> Any:
    """Convenience: resolve a capability's adapter using the app's :func:`mixle_mlops.config.get_settings`
    (its ``llm_backends`` and default LLM base_url/api_key), with the capability map taken from
    ``MIXLE_CAPABILITY_MODELS`` overlaid with any explicit ``capability_models`` argument."""
    from mixle_mlops.config import get_settings

    s = settings or get_settings()
    merged = {**capability_models_from_env(), **(capability_models or {})}
    return resolve_adapter(
        capability, s.llm_backends or {}, capability_models=merged,
        default_base_url=s.llm_base_url, default_api_key=s.llm_api_key,
    )
