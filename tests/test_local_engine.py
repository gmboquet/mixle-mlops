"""Adapter-aware local model loading: a PEFT (LoRA) adapter over a served base model actually changes what
gets served, on an in-process tiny GPT-2 -- no download, no GPU, no real weights."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from mixle_mlops.engines import HFLogitProvider  # noqa: E402
from mixle_mlops.models.local_engine import load_local_engine  # noqa: E402


def _tiny_model():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=32, n_positions=64, n_embd=16, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=1)
    return GPT2LMHeadModel(cfg)


def _fit_tiny_adapter(tmp_path):
    """Build a real LoRA adapter over a fresh tiny GPT-2 and save it, returning the adapter dir. The adapter's
    lora_B is zero-initialized by construction (so a fresh LoRA is a no-op) -- perturbed here so the saved
    adapter has a real, nonzero effect, the way a trained one would."""
    import torch
    from peft import LoraConfig, get_peft_model

    base = _tiny_model()
    peft_model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.5)
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))
    return str(adapter_dir)


def test_adapter_path_changes_served_logits(tmp_path):
    adapter_dir = _fit_tiny_adapter(tmp_path)
    ids = [3, 7, 2]

    unadapted = HFLogitProvider(model=_tiny_model())
    adapted = HFLogitProvider(model=_tiny_model(), adapter_path=adapter_dir)  # same seed -> identical base weights
    assert not np.allclose(unadapted.next_logits(ids), adapted.next_logits(ids))


def test_adapter_path_is_optional_and_defaults_to_unadapted():
    # no adapter_path -> ordinary base-model behavior, unchanged from before this feature existed
    a = HFLogitProvider(model=_tiny_model())
    b = HFLogitProvider(model=_tiny_model())
    np.testing.assert_allclose(a.next_logits([3, 7]), b.next_logits([3, 7]))


def test_load_local_engine_rejects_adapter_with_poe_ensemble():
    with pytest.raises(ValueError, match="exactly one base model"):
        load_local_engine("x", ["a", "b"], adapter_path="/some/adapter")
