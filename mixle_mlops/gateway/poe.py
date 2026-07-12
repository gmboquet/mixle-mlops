"""Token-level Product-of-Experts fusion — combine several models' next-token distributions geometrically.

The math (exact) is `mixle.ops.product_of_experts`: a fused distribution `p(t) ∝ ∏_k p_k(t)^{w_k}` over the shared
token support. ``fuse_next_token`` builds it from each model's top-logprobs; ``poe_rerank`` is the sequence-level
form (pick the candidate with the highest weighted sum of per-model log-probabilities — PoE in log space).

Boundary: true per-token PoE-decoded generation needs incremental decode control over a shared vocabulary:
feed the fused distribution back, sample, and repeat. The OpenAI-compatible chat API exposes neither forced-token
continuation nor, across heterogeneous tokenizers, an aligned vocabulary. Full token-by-token PoE decoding
therefore requires a logit-level serving integration such as a custom vLLM or llama.cpp sampler with vocabulary
projection. This module provides the exact fusion primitive and the sequence-level reranker supported by the
available API surface."""
from __future__ import annotations

from typing import Any, Awaitable, Callable

import numpy as np


def fuse_next_token(model_logprobs: list[dict[str, float]],
                    weights: list[float] | None = None) -> dict[str, float]:
    """Fuse K models' next-token distributions (each a ``{token: logprob}`` from the backend's top_logprobs) into
    one Product-of-Experts distribution over the shared tokens. Returns ``{token: probability}`` (sums to 1)."""
    from mixle.ops import product_of_experts
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

    dists = []
    for lp in model_logprobs:
        tokens = list(lp.keys())
        vals = np.asarray([lp[t] for t in tokens], dtype=float)
        probs = np.exp(vals - vals.max())
        probs /= probs.sum()                                   # renormalize each model's top-k to a distribution
        dists.append(CategoricalDistribution(pmap={t: float(p) for t, p in zip(tokens, probs)}))

    fused = product_of_experts(dists, weights)                 # exact geometric pool over the shared support
    pmap = fused.pmap
    return {t: float(p) for t, p in pmap.items()}


async def poe_rerank(candidates: list[str], logprob_fn: Callable[[str, str], Awaitable[float]],
                     model_names: list[str], weights: list[float] | None = None
                     ) -> tuple[str, dict[str, Any]]:
    """Sequence-level PoE: score each candidate by the weighted sum of its log-probability under every model, and
    return the highest-scoring one. ``logprob_fn(model, candidate) -> total log-prob`` is supplied by the caller
    (it needs a logprob-scoring backend, e.g. an echo+logprobs completion endpoint)."""
    weights = weights or [1.0] * len(model_names)
    scores: list[float] = []
    for candidate in candidates:
        total = 0.0
        for w, model in zip(weights, model_names):
            total += w * float(await logprob_fn(model, candidate))
        scores.append(total)
    best = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best], {"scores": scores, "best_index": best, "models": list(model_names)}
