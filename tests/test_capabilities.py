"""Capability-routed model selection: ask for a capability, get the right adapter, swappably. All
offline (adapter construction only, no network) -- the live end-to-end vision proof is done separately."""

import pytest

from mixle_mlops.models.capabilities import (
    TEXT,
    VISION,
    capability_models_from_env,
    providers_from_keys,
    resolve_adapter,
)
from mixle_mlops.models.openai_compat import OpenAICompatAdapter


def _backends():
    return providers_from_keys({"DEEPSEEK_API_KEY": "sk-deep", "QWEN_API_KEY": "sk-qwen"})


def test_providers_from_keys_builds_expected_backends_and_ignores_unknown():
    backends = providers_from_keys({"DEEPSEEK_API_KEY": "sk-deep", "QWEN_API_KEY": "sk-qwen", "MYSTERY_KEY": "x"})
    assert backends["deepseek-chat"]["base_url"] == "https://api.deepseek.com"
    assert backends["deepseek-chat"]["api_key"] == "sk-deep"
    assert backends["qwen3-vl-plus"]["base_url"].startswith("https://dashscope-intl")
    assert backends["qwen3-vl-plus"]["api_key"] == "sk-qwen"
    assert not any("MYSTERY" in k or v.get("api_key") == "x" for k, v in backends.items())


def test_vision_routes_to_the_qwen_vl_backend():
    adapter = resolve_adapter(VISION, _backends())
    assert isinstance(adapter, OpenAICompatAdapter)
    assert adapter.base_url.startswith("https://dashscope-intl")  # the VLM, not the text model


def test_text_routes_to_deepseek():
    adapter = resolve_adapter(TEXT, _backends())
    assert isinstance(adapter, OpenAICompatAdapter)
    assert adapter.base_url == "https://api.deepseek.com"


def test_capability_map_is_overridable_the_whole_point_of_being_flexible():
    # swap vision to a different model without touching any caller
    backends = _backends()
    adapter = resolve_adapter(VISION, backends, capability_models={VISION: "qwen3-vl-flash"})
    assert adapter.name == "qwen3-vl-flash"


def test_unknown_capability_raises_loudly():
    with pytest.raises(KeyError, match="no model configured for capability"):
        resolve_adapter("audio", _backends())


def test_capability_with_no_backend_and_no_default_raises_not_silently_wrong():
    # vision routes to qwen3-vl-plus, but we give it a backends map missing that model and no default
    with pytest.raises(KeyError, match="no backend is configured"):
        resolve_adapter(VISION, {"deepseek-chat": {"base_url": "https://api.deepseek.com"}})


def test_default_base_url_fallback_lets_an_unconfigured_model_still_resolve():
    adapter = resolve_adapter(VISION, {}, default_base_url="http://localhost:11434/v1", default_api_key="k")
    assert isinstance(adapter, OpenAICompatAdapter)
    assert adapter.base_url == "http://localhost:11434/v1"


def test_capability_models_from_env(monkeypatch):
    monkeypatch.setenv("MIXLE_CAPABILITY_MODELS", '{"vision": "qwen3-vl-flash"}')
    assert capability_models_from_env() == {"vision": "qwen3-vl-flash"}
    monkeypatch.setenv("MIXLE_CAPABILITY_MODELS", "not json")
    assert capability_models_from_env() == {}  # degrades to defaults, never raises


def test_resolve_embedder_routes_to_the_configured_remote_model():
    from mixle_mlops.models.capabilities import resolve_embedder
    emb = resolve_embedder(_backends())
    assert emb.base_url.startswith("https://dashscope-intl")  # the embedding model's backend
    assert emb.allow_remote is True


def test_resolve_embedder_falls_back_to_local_hashing_when_no_backend():
    from mixle_mlops.models.capabilities import resolve_embedder
    emb = resolve_embedder({})  # no backends configured
    assert emb.allow_remote is False  # deterministic local hashing, retrieval still works


def test_embedding_capability_is_overridable_like_the_others():
    from mixle_mlops.models.capabilities import EMBEDDING, resolve_embedder
    backends = {"my-embed": {"provider": "openai", "base_url": "http://x/v1", "api_key": "k"}}
    emb = resolve_embedder(backends, capability_models={EMBEDDING: "my-embed"})
    assert emb.base_url == "http://x/v1"
