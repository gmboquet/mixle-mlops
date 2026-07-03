"""Bridge a logit provider into mixle's enumeration stack with every speed contract wired.

``mixle.enumeration.AutoregressiveEnumerable`` accepts optional capability callables that this module maps
onto a real transformers backend (:class:`~mixle_mlops.engines.providers.TreeLogitProvider`):

* ``next_logprobs`` -- KV-prefix-cached incremental forwards (one new position per tree node instead of a
  full prefix re-encode);
* ``batch_score_sequences`` -- teacher forcing: ONE forward per sequence scores all its positions (the
  rescoring primitive for speculative enumeration, ``mixle.enumeration.RescoredIndex``);
* ``all_position_logprobs`` -- the same rows as full distributions, harvested into the adapter's forward
  cache (one forward warms L contexts; corpus-calibrated envelopes ride on this).

So the full toolchain -- SeekIndex / AREnvelopeIndex / LatticeEnvelopeIndex / RescoredIndex / branch_cap /
quantized-inference certificates -- runs against an actual causal LM through one constructor call.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    z = logits - m
    return z - float(np.log(np.sum(np.exp(z))))


def autoregressive_enumerable(
    provider: Any,
    *,
    max_len: int | None = None,
    eos: int | None = None,
    branch_cap: int | None = None,
    bin_width_bits: float = 1.0,
    oversample: int = 8,
    count_mode: str = "auto",
):
    """Wrap ``provider`` (a ``TreeLogitProvider``/``HFLogitProvider``-shaped object) as a fully-wired
    :class:`mixle.enumeration.AutoregressiveEnumerable`.

    The provider must expose ``next_logits(token_ids)``; ``prefix_logprobs(token_ids)`` (teacher forcing)
    is used when present to wire the batched scoring + harvest contracts. ``eos`` defaults to the
    provider's own eos when ``max_len`` is not given.
    """
    from mixle.enumeration import AutoregressiveEnumerable

    if max_len is None and eos is None:
        eos = getattr(provider, "eos", None)
        if eos is None:
            raise ValueError("give max_len or eos (provider exposes no eos token)")

    def next_logprobs(prefix: tuple):
        logits = np.asarray(provider.next_logits(list(prefix)), dtype=float)
        lp = _log_softmax(logits)
        return np.arange(lp.size), lp

    batch_score = None
    all_positions = None
    prefix_logprobs = getattr(provider, "prefix_logprobs", None)
    if callable(prefix_logprobs):

        def batch_score(seqs: list[tuple]) -> np.ndarray:  # one teacher-forcing forward per sequence
            out = np.empty(len(seqs), dtype=float)
            for j, s in enumerate(seqs):
                rows = np.asarray(prefix_logprobs(list(s)), dtype=float)
                out[j] = float(sum(rows[d, int(t)] for d, t in enumerate(s)))
            return out

        def all_positions(seq: tuple) -> list:  # full next-token distributions at every prefix, one forward
            rows = np.asarray(prefix_logprobs(list(seq)), dtype=float)
            return [(np.arange(rows.shape[1]), rows[d]) for d in range(min(len(seq), rows.shape[0]))]

    return AutoregressiveEnumerable(
        next_logprobs,
        max_len=max_len,
        eos=eos,
        bin_width_bits=bin_width_bits,
        oversample=oversample,
        count_mode=count_mode,
        branch_cap=branch_cap,
        batch_score_sequences=batch_score,
        all_position_logprobs=all_positions,
    )
