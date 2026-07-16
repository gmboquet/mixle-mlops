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


def test_sibling_branches_do_not_mutate_the_shared_ancestor_cache():
    """Regression test for the removed ``transformers`` ``DynamicCache.to_legacy_cache``/``from_legacy_cache``
    round-trip that ``TreeLogitProvider._fresh_past`` used to rely on for a "fresh" per-branch copy.

    Once those helpers were removed upstream, ``_fresh_past``'s ``except Exception`` silently fell back to
    handing out the SAME live (mutable) ``Cache`` object every time an ancestor was resumed, instead of an
    independent copy: exploring one child of a node grew that node's OWN stored cache in place, so every later
    sibling silently resumed from the *first* sibling's already-extended state rather than the ancestor's.
    A branch_cap-wide search (see ``test_bridge_end_to_end_with_mixle_stack``) routes through exactly this
    path for every node, so the accumulated sequence length climbed past the model's ``n_positions`` with
    enough siblings explored -- ``IndexError: index out of range in self`` out of the position-embedding
    table (``wpe``), or silently wrong logits for prefixes that hadn't yet overflowed.

    Two children of the SAME ancestor must therefore leave that ancestor's own cached KV state exactly as
    long as it was before either child was explored.
    """
    model = _tiny_model()
    tree = TreeLogitProvider(model=model)
    tree.next_logits([3])  # caches ancestor prefix (3,)
    ancestor_len_before = tree._nodes[(3,)][0].get_seq_length()

    tree.next_logits([3, 7])  # explore one child of (3,)
    tree.next_logits([3, 9])  # explore a SIBLING child of (3,), resuming from the same ancestor
    tree.next_logits([3, 20])  # and a third sibling, for good measure

    ancestor_len_after = tree._nodes[(3,)][0].get_seq_length()
    assert ancestor_len_after == ancestor_len_before, (
        f"exploring children of (3,) must not grow the ancestor's own stored KV cache "
        f"(was {ancestor_len_before}, now {ancestor_len_after})"
    )


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
    # Ordering is exact BETWEEN quantized fine buckets, not within one (see SeekIndex.slice's docstring):
    # two candidates less than one bucket-width apart may surface in either relative order. Compare by fine
    # bucket -- the same granularity the index itself sorts on -- not raw log-density, which is finer than
    # the index's own ordering guarantee and can legitimately flip between near-tied candidates.
    buckets = [ar.structural_fine_bucket(s, si.quantizer) for s, _lp in seqs]
    assert buckets == sorted(buckets)
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
