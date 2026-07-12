"""Serve a local model through the logit-level decode engine — so token-level Product-of-Experts and grammar
masking are available as a *hosted model*, not just a library call.

One provider → a single local model. Several providers → a Product-of-Experts ensemble decoded token-by-token
(true per-step fusion, the thing the chat-API proxy cannot do). This is how the platform hosts a small local
model with the bridge's logit-level levers turned on."""
from __future__ import annotations

import os
from typing import Any, AsyncIterator

from ..core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    ModelAdapter,
)
from ..engines import decode, decode_iter, speculative_decode


class LocalEngineAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name: str, providers: Any, *, max_new_tokens: int = 128,
                 weights: list[float] | None = None):
        self._name = name
        self._providers = list(providers) if isinstance(providers, (list, tuple)) else [providers]
        self._primary = self._providers[0]
        self.max_new_tokens = max_new_tokens
        self.weights = weights

    @property
    def name(self) -> str:
        return self._name

    def vocab(self) -> dict[int, str]:
        return self._primary.vocab()

    def _prompt_ids(self, req: ChatRequest) -> list[int]:
        tok = getattr(self._primary, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template") and tok.chat_template:
            messages = [{"role": m.role, "content": m.text()} for m in req.messages]
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            return list(ids)
        text = "".join(f"{m.role}: {m.text()}\n" for m in req.messages) + "assistant: "
        return self._primary.encode(text)

    def _eos_ids(self) -> set[int]:
        """All token ids that should stop generation (model eos + chat-template end tokens)."""
        tok = getattr(self._primary, "tokenizer", None)
        eos = set()
        raw = getattr(self._primary, "eos", None)
        if raw is not None:
            eos.update(raw if isinstance(raw, (list, tuple)) else [raw])
        if tok is not None:
            # Add any additional special end tokens (e.g. <|im_end|> for ChatML models)
            for name in ("eos_token", "pad_token"):
                t = getattr(tok, name, None)
                if t:
                    ids = tok.convert_tokens_to_ids([t])
                    eos.update(i for i in ids if i != tok.unk_token_id)
            for tok_id in (tok.additional_special_tokens_ids or []):
                decoded = tok.decode([tok_id])
                if "end" in decoded or "eot" in decoded or "im_end" in decoded:
                    eos.add(tok_id)
        return eos

    def _seed(self, req: ChatRequest) -> int:
        # decode() defaults to seed=0, so without a per-call seed every sampled generation is
        # IDENTICAL (temperature has no effect). Honor an explicit request seed; otherwise draw a
        # fresh one so temperature sampling is actually stochastic (needed for e.g. semantic-entropy UQ).
        s = req.extra.get("seed")
        if s is not None:
            return int(s)
        return int.from_bytes(os.urandom(4), "little")

    def _generate(self, req: ChatRequest) -> str:
        ids = self._prompt_ids(req)
        provs = self._providers if len(self._providers) > 1 else self._primary
        eos_ids = self._eos_ids()
        # decode() accepts a single eos_id; use the first one and strip the rest post-hoc
        first_eos = next(iter(eos_ids), None)
        out = decode(
            provs, prompt_ids=ids, max_new_tokens=req.max_tokens or self.max_new_tokens,
            weights=self.weights, eos_id=first_eos,
            greedy=not req.temperature, temperature=req.temperature or 1.0, top_p=req.top_p or 1.0,
            grammar=req.extra.get("_grammar"), seed=self._seed(req),
        )
        # Trim any trailing EOS tokens before decoding
        while out and out[-1] in eos_ids:
            out = out[:-1]
        return self._primary.decode_text(out)

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        text = self._generate(req)
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=text), finish_reason="stop")])

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        provs = self._providers if len(self._providers) > 1 else self._primary
        ids = self._prompt_ids(req)
        eos_ids = self._eos_ids()
        first_eos = next(iter(eos_ids), None)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(role="assistant"))])
        emitted: list[int] = []
        prev = ""
        for tok_id in decode_iter(provs, prompt_ids=ids, max_new_tokens=req.max_tokens or self.max_new_tokens,
                                  weights=self.weights, eos_id=first_eos,
                                  greedy=not req.temperature, temperature=req.temperature or 1.0,
                                  top_p=req.top_p or 1.0, grammar=req.extra.get("_grammar"),
                                  seed=self._seed(req)):
            if tok_id in eos_ids:
                break
            emitted.append(tok_id)
            text = self._primary.decode_text(emitted)
            delta = text[len(prev):]
            prev = text
            if delta:
                yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(content=delta))])
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(), finish_reason="stop")])


class SpeculativeAdapter(ModelAdapter):
    """Serve a ``(draft, target)`` pair as ONE fast model. Speculative decoding returns the target's exact output
    distribution at lower latency: the cheap draft proposes k tokens, the target verifies them in a single pass.
    The two providers must share a tokenizer/vocabulary."""

    kind = "llm"

    def __init__(self, name: str, draft: Any, target: Any, *, k: int = 4, max_new_tokens: int = 128):
        self._name = name
        self._draft = draft
        self._target = target
        self.k = k
        self.max_new_tokens = max_new_tokens

    @property
    def name(self) -> str:
        return self._name

    def vocab(self) -> dict[int, str]:
        return self._target.vocab()

    def _generate(self, req: ChatRequest) -> str:
        text = "".join(f"{m.role}: {m.text()}\n" for m in req.messages) + "assistant: "
        ids = self._target.encode(text)
        out = speculative_decode(
            self._draft, self._target, prompt_ids=ids, max_new_tokens=req.max_tokens or self.max_new_tokens,
            k=self.k, eos_id=getattr(self._target, "eos", None),
            greedy=not req.temperature, temperature=req.temperature or 1.0)
        return self._target.decode_text(out)

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        text = self._generate(req)
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=text), finish_reason="stop")])

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        text = self._generate(req)                            # speculative decoding is batched; emit the result
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(role="assistant", content=text), finish_reason="stop")])


def load_local_engine(name: str, model_names: list[str], *, max_new_tokens: int = 128,
                       adapter_path: str | None = None) -> LocalEngineAdapter:
    """Load one or more transformers models and wrap them as a local-engine adapter (PoE ensemble if >1).

    ``adapter_path`` loads a PEFT (LoRA/QLoRA) adapter over the base model -- e.g. the output of a
    fine-tune job trained by the generated ``llm_lora_train.py`` script (``compute/jobspec.py``) and
    registered under ``registry_root`` (see ``gateway/routes/models.py``'s ``POST /v1/models/load``).
    Only meaningful for a single base model: a PoE ensemble with per-model adapters isn't supported."""
    from ..engines import HFLogitProvider

    if adapter_path is not None and len(model_names) != 1:
        raise ValueError("adapter_path requires exactly one base model, not a PoE ensemble")
    providers = [
        HFLogitProvider(model_name=m, adapter_path=adapter_path if i == 0 else None)
        for i, m in enumerate(model_names)
    ]
    return LocalEngineAdapter(name, providers, max_new_tokens=max_new_tokens)


def load_speculative_engine(name: str, draft_model: str, target_model: str, *,
                            k: int = 4, max_new_tokens: int = 128) -> SpeculativeAdapter:
    """Load a small draft + a larger target transformers model (shared tokenizer) as a speculative-decoding model."""
    from ..engines import HFLogitProvider

    return SpeculativeAdapter(name, HFLogitProvider(model_name=draft_model), HFLogitProvider(model_name=target_model),
                              k=k, max_new_tokens=max_new_tokens)
