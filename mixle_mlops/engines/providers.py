"""Logit providers for the decode engine: a deterministic toy n-gram (tests) and a transformers backend (real)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class NgramProvider:
    """Deterministic toy LM: next-token logits are a fixed function of the last token (a bigram logit table).

    Useful for fast, exact tests of the decode loop / PoE fusion / grammar masking without loading a real model."""

    def __init__(self, logit_table: np.ndarray, *, initial: np.ndarray | None = None):
        self.table = np.asarray(logit_table, dtype=np.float64)  # (vocab, vocab): row = last token
        self.vocab_size = int(self.table.shape[0])
        self.initial = (
            np.asarray(initial, dtype=np.float64)
            if initial is not None
            else np.zeros(self.vocab_size, dtype=np.float64)
        )

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        if len(token_ids) == 0:
            return self.initial.copy()
        return self.table[int(token_ids[-1])].copy()


class HFLogitProvider:
    """A real transformers ``AutoModelForCausalLM`` exposing per-step next-token logits — the genuine logit-level
    backend that makes token-level PoE + grammar masking work with actual models.

    ``adapter_path``, when given, loads a PEFT adapter (e.g. the LoRA output of the generated
    ``llm_lora_train.py`` script, see ``compute/jobspec.py``) over the base model via
    ``peft.PeftModel.from_pretrained`` -- applies regardless of whether the base model came from
    ``model_name=`` or was passed in directly as ``model=``, so a fitted in-memory model can be adapter-
    tested without a real tokenizer/model_name."""

    def __init__(self, model=None, tokenizer=None, *, model_name: str | None = None, device: str = "cpu",
                 adapter_path: str | None = None):
        import torch

        self._torch = torch
        if model is None:
            if model_name is None:
                raise ValueError("HFLogitProvider needs a model= or a model_name=")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = model
            self.tokenizer = tokenizer
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.vocab_size = int(self.model.config.vocab_size)
        self._bos = getattr(self.model.config, "bos_token_id", None)
        self.eos = getattr(self.model.config, "eos_token_id", None)

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        torch = self._torch
        ids = list(token_ids) or [self._bos if self._bos is not None else 0]
        with torch.no_grad():
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(tensor).logits[0, -1]
        return logits.float().cpu().numpy()

    def seq_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """All-position next-token logits ``(len, vocab)`` in ONE forward pass — what makes speculative
        verification a speedup (the target checks k drafted tokens in a single call)."""
        torch = self._torch
        ids = list(token_ids) or [self._bos if self._bos is not None else 0]
        with torch.no_grad():
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(tensor).logits[0]
        return logits.float().cpu().numpy()

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text)) if self.tokenizer is not None else []

    def decode_text(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids)) if self.tokenizer is not None else ""

    def vocab(self) -> dict[int, str]:
        if getattr(self, "_vocab", None) is None:
            if self.tokenizer is None:
                self._vocab = {}
            else:
                self._vocab = {i: self.tokenizer.decode([i]) for i in range(self.vocab_size)}
        return self._vocab


class TreeLogitProvider:
    """KV-prefix-cached transformers provider for tree-structured workloads (enumeration, tree search).

    ``HFLogitProvider.next_logits`` re-encodes the whole prefix on every call -- O(d^2) attention per node
    when expanding a prefix TREE, where each child differs from its parent by one token. This provider
    keeps a trie of ``past_key_values``: a call finds the deepest cached ancestor of the requested prefix
    and forwards ONLY the suffix tokens against that ancestor's KV state -- one incremental position per
    child in the common case. Numerically it is the same attention computation (float rounding aside).

    The cache is LRU-bounded by ``max_cached_nodes`` (KV memory per node is ~2 * layers * heads * head_dim
    * len floats); evicted nodes are transparently recomputed from their deepest surviving ancestor.
    ``prefix_logprobs`` is the teacher-forcing companion: ONE forward returning the log-softmax rows for
    every prefix of a sequence -- the ``all_position_logprobs`` / ``batch_score_sequences`` contracts of
    :class:`mixle.enumeration.AutoregressiveEnumerable` (see ``enumerate_bridge``).

    Context convention: every prefix is rooted at BOS -- ``next_logits(p)`` scores the context
    ``[bos] + p`` -- so the tree is self-consistent (a child extends its parent's context; the root ``()``
    is the real BOS context). Note :class:`HFLogitProvider` feeds a bare non-empty prefix instead; align
    inputs when comparing the two.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        *,
        model_name: str | None = None,
        device: str = "cpu",
        max_cached_nodes: int = 512,
    ):
        import torch

        self._torch = torch
        if model is None:
            if model_name is None:
                raise ValueError("TreeLogitProvider needs a model= or a model_name=")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.vocab_size = int(self.model.config.vocab_size)
        self.bos = getattr(self.model.config, "bos_token_id", None)
        self.eos = getattr(self.model.config, "eos_token_id", None)
        self.max_cached_nodes = int(max_cached_nodes)
        from collections import OrderedDict

        self._nodes: OrderedDict[tuple, tuple] = OrderedDict()  # prefix -> (stored_kv, last_logits)

    # --- KV (de)hydration across transformers versions ---
    #
    # `stored` must be an immutable snapshot: a plain tuple of `(key_tensor, value_tensor)` per layer. Several
    # tree children can share one ancestor node, so `_fresh_past` has to hand each of them a Cache object that
    # is NOT the live object any sibling's forward call might still be growing -- transformers' `Cache.update`
    # extends a layer's cache by rebinding `layer.keys`/`layer.values` to a new concatenated tensor (see
    # `DynamicLayer.update`), so passing the SAME Cache object into two sibling forward calls silently makes
    # the second one see the first's tokens too (`get_seq_length()` grows out from under it), producing
    # wrong -- too high -- `position_ids` for the second sibling. transformers<5 avoided this because
    # `DynamicCache.from_legacy_cache(stored)` always built a brand-new Cache from the immutable tuple; that
    # classmethod (and its inverse `to_legacy_cache`) were removed in transformers 5, and the old broad
    # `except Exception` here silently swallowed the resulting AttributeError and fell back to returning the
    # live Cache object unchanged, reintroducing exactly the shared-mutable-state bug this method exists to
    # avoid. Rebuild manually from `Cache.layers[i].keys/.values` (5.x's stable public shape) when the legacy
    # helpers aren't available.
    def _fresh_past(self, stored):
        """A fresh Cache object over the stored (immutable) per-layer tensors -- safe to reuse across branches."""
        from transformers.cache_utils import DynamicCache

        from_legacy = getattr(DynamicCache, "from_legacy_cache", None)
        if callable(from_legacy):
            return from_legacy(stored)  # transformers < 5
        cache = DynamicCache()  # transformers >= 5: rebuild layer-by-layer from the stored tensors
        for layer_idx, (keys, values) in enumerate(stored):
            cache.update(keys, values, layer_idx)
        return cache

    @staticmethod
    def _to_stored(past):
        to_legacy = getattr(past, "to_legacy_cache", None)
        if callable(to_legacy):
            return to_legacy()  # transformers < 5
        layers = getattr(past, "layers", None)  # transformers >= 5: Cache.layers (list of CacheLayerMixin)
        if layers is not None:
            return tuple((layer.keys, layer.values) for layer in layers)
        return past  # already a plain tuple, or an unrecognized Cache shape -- pass through

    def _root_ids(self) -> list[int]:
        return [self.bos if self.bos is not None else 0]

    def _forward_from(self, ancestor_key: tuple, key: tuple) -> tuple:
        """Forward the tokens between ``ancestor_key`` and ``key`` on top of the ancestor's KV state."""
        torch = self._torch
        if ancestor_key is None:
            new_ids = self._root_ids() + list(key)
            past = None
        else:
            new_ids = list(key[len(ancestor_key) :])
            past = self._fresh_past(self._nodes[ancestor_key][0])
            self._nodes.move_to_end(ancestor_key)
        with torch.no_grad():
            tensor = torch.tensor([new_ids], dtype=torch.long, device=self.device)
            out = self.model(tensor, past_key_values=past, use_cache=True)
        stored = (self._to_stored(out.past_key_values), out.logits[0, -1].float().cpu().numpy())
        self._nodes[key] = stored
        while len(self._nodes) > self.max_cached_nodes:
            self._nodes.popitem(last=False)  # LRU eviction; descendants recompute from ancestors
        return stored

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        key = tuple(int(t) for t in token_ids)
        hit = self._nodes.get(key)
        if hit is not None:
            self._nodes.move_to_end(key)
            return hit[1].copy()
        ancestor = None
        for k in range(len(key) - 1, -1, -1):  # deepest cached ancestor (root () included)
            if key[:k] in self._nodes:
                ancestor = key[:k]
                break
        return self._forward_from(ancestor, key)[1].copy()

    def prefix_logprobs(self, token_ids: Sequence[int]) -> np.ndarray:
        """Teacher forcing: ``(len, vocab)`` log-softmax rows, row ``d`` = next-token dist after prefix
        ``token_ids[:d]`` -- every prefix of the sequence scored in ONE forward."""
        torch = self._torch
        seq = [int(t) for t in token_ids]
        ids = self._root_ids() + seq[:-1] if seq else self._root_ids()
        with torch.no_grad():
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(tensor).logits[0].float()
            rows = torch.log_softmax(logits, dim=-1).cpu().numpy()
        return rows if seq else rows[:1]

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text)) if self.tokenizer is not None else []

    def decode_text(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids)) if self.tokenizer is not None else ""


class CharProvider:
    """Toy char-level LM over a fixed alphabet: deterministic bigram logits + char encode/decode. Lets the
    LocalEngineAdapter be exercised end-to-end without a model/tokenizer download."""

    def __init__(self, alphabet: str, *, table: np.ndarray | None = None, eos: int | None = None):
        self.alphabet = alphabet
        self.vocab_size = len(alphabet)
        self._c2i = {c: i for i, c in enumerate(alphabet)}
        self.table = (
            np.asarray(table, dtype=np.float64)
            if table is not None
            else np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        )
        self.eos = eos

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        if len(token_ids) == 0:
            return np.zeros(self.vocab_size, dtype=np.float64)
        return self.table[int(token_ids[-1])].copy()

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text if c in self._c2i]

    def decode_text(self, token_ids: Sequence[int]) -> str:
        return "".join(self.alphabet[i] for i in token_ids if 0 <= i < self.vocab_size)

    def vocab(self) -> dict[int, str]:
        return {i: c for i, c in enumerate(self.alphabet)}
