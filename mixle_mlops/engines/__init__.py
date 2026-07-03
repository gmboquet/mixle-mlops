"""Logit-level local inference: incremental decoding with token-level Product-of-Experts fusion + grammar masking
— the serving integration the OpenAI chat API can't provide (it has no forced-token continuation / logit access)."""

from .decode import LogitProvider, decode, decode_iter, fuse_logprobs, speculative_decode
from .grammar import TokenFSA
from .enumerate_bridge import autoregressive_enumerable
from .providers import HFLogitProvider, NgramProvider, TreeLogitProvider

__all__ = [
    "decode",
    "decode_iter",
    "speculative_decode",
    "fuse_logprobs",
    "LogitProvider",
    "TokenFSA",
    "NgramProvider",
    "HFLogitProvider",
    "TreeLogitProvider",
    "autoregressive_enumerable",
]
