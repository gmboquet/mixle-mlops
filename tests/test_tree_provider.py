"""TreeLogitProvider (trie KV cache) + the mixle enumeration bridge, on an in-process tiny GPT-2.

The claims under test: KV-prefix-cached forwards return the SAME logits as full re-encodes (parity within
float noise) while computing O(1) new positions per tree node instead of O(depth) (counted directly on the
wrapped model); the LRU bound preserves correctness; ``prefix_logprobs`` teacher-forcing rows equal the
per-prefix next-token distributions; and ``autoregressive_enumerable`` wires a real causal LM into mixle's
SeekIndex / envelope / rescoring stack with exact log-densities end to end.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from mixle_mlops.engines import HFLogitProvider, TreeLogitProvider, autoregressive_enumerable  # noqa: E402


def _tiny_model():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=32, n_positions=64, n_embd=16, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=1)
    return GPT2LMHeadModel(cfg)


class _CountingModel:
    """Wraps the HF model counting how many token POSITIONS each forward computes."""

    def __init__(self, inner):
        self.inner = inner
        self.positions = 0

    def __call__(self, tensor, **kw):
        self.positions += int(tensor.shape[1])
        return self.inner(tensor, **kw)


def test_kv_cached_logits_match_full_reencode():
    # like-for-like: the tree provider roots at BOS, so compare against a full re-encode of [bos] + p
    model = _tiny_model()
    tree = TreeLogitProvider(model=model)
    full = HFLogitProvider(model=model)
    prefixes = [(), (3,), (3, 7), (3, 7, 2), (3, 9), (5,), (5, 5, 5, 5), (3, 7, 2, 11, 4)]
    for p in prefixes:
        a = tree.next_logits(list(p))
        b = full.next_logits([0, *p])  # bos_token_id = 0 in the tiny config
        np.testing.assert_allclose(a, b, atol=1e-4, err_msg=str(p))


def test_incremental_positions_vs_quadratic_reencode():
    model = _tiny_model()
    depth = 10
    chain = tuple(range(2, 2 + depth))

    tree = TreeLogitProvider(model=model)
    tree.model = counting_tree = _CountingModel(tree.model)
    for d in range(depth + 1):
        tree.next_logits(list(chain[:d]))

    full = HFLogitProvider(model=model)
    full.model = counting_full = _CountingModel(full.model)
    for d in range(depth + 1):
        full.next_logits(list(chain[:d]))

    # tree: 1 root position + 1 per extension; full re-encode: sum of prefix lengths (quadratic)
    assert counting_tree.positions <= 2 * (depth + 2)
    assert counting_full.positions >= depth * (depth + 1) // 2
    assert counting_tree.positions * 3 < counting_full.positions


def test_lru_bound_preserves_correctness():
    model = _tiny_model()
    tree = TreeLogitProvider(model=model, max_cached_nodes=2)  # aggressive eviction
    full = HFLogitProvider(model=model)
    for p in [(), (4,), (4, 6), (7,), (7, 8, 9), (4, 6, 1)]:
        np.testing.assert_allclose(tree.next_logits(list(p)), full.next_logits([0, *p]), atol=1e-4)
    assert len(tree._nodes) <= 2


def test_prefix_logprobs_rows_match_per_prefix_distributions():
    model = _tiny_model()
    tree = TreeLogitProvider(model=model)
    seq = [3, 7, 2, 11]
    rows = tree.prefix_logprobs(seq)
    assert rows.shape == (4, 32)
    for d in range(4):
        logits = tree.next_logits(seq[:d])
        expect = logits - (np.max(logits) + np.log(np.sum(np.exp(logits - np.max(logits)))))
        np.testing.assert_allclose(rows[d], expect, atol=1e-4)
    assert tree.prefix_logprobs([]).shape == (1, 32)  # empty sequence: the root row


def test_bridge_end_to_end_with_mixle_stack():
    from mixle.enumeration import AREnvelopeIndex, SeekIndex

    model = _tiny_model()
    provider = TreeLogitProvider(model=model)
    ar = autoregressive_enumerable(provider, max_len=3, branch_cap=8, oversample=8)

    # exact log-density == teacher-forcing row gather == batched scorer
    seq = (3, 7, 2)
    rows = provider.prefix_logprobs(list(seq))
    manual = float(sum(rows[d, t] for d, t in enumerate(seq)))
    assert abs(ar.log_density(seq) - manual) < 1e-5
    np.testing.assert_allclose(ar.score_sequences([seq, (1, 1, 1)]), [manual, ar.log_density((1, 1, 1))], atol=1e-5)

    # the persistent index unranks real sequences with exact scores over the capped sub-support
    si = SeekIndex(ar)
    seqs = si.slice(0, 5)
    assert len(seqs) == 5
    lps = [lp for _s, lp in seqs]
    assert lps == sorted(lps, reverse=True)
    for s, lp in seqs:
        assert abs(lp - ar.log_density(s)) < 1e-9

    # corpus-calibrated envelope over the real model (harvest = one forward per calibration sequence)
    env = AREnvelopeIndex(ar, calibration_sequences=[(2, 3, 4), (5, 6, 7), (0, 2, 9)])
    s, lp = env.unrank(100)
    assert len(s) == 3
    assert abs(lp - ar.log_density(s)) < 1e-9


def test_bridge_defaults_to_provider_eos():
    provider = TreeLogitProvider(model=_tiny_model())
    ar = autoregressive_enumerable(provider)  # no max_len: terminating on the provider's eos
    assert ar.eos == 1
