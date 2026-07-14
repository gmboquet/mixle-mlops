"""IC-6 — the `Verifier` protocol + `Verdict` (frozen; work-plan §5). Fills the empty ``verification/`` package.

A verifier judges a claim against context and returns a `Verdict{passed, score, reasons, kind}`. Frozen ``kind``
values: ``physical`` (bounds/mass-balance), ``calibration`` (PIT/coverage), ``exact``, ``llm_judge``. E10 lands the
physical + calibration verifiers under this protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

VERDICT_KINDS = ("physical", "calibration", "exact", "llm_judge")


@dataclass(frozen=True)
class Verdict:
    """The outcome of a verification: pass/fail, a continuous score in ``[0, 1]``, human-readable reasons, and kind."""

    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    kind: str = "exact"

    def __post_init__(self) -> None:
        if self.kind not in VERDICT_KINDS:
            raise ValueError(f"unknown verdict kind {self.kind!r}; expected one of {VERDICT_KINDS}")


@runtime_checkable
class Verifier(Protocol):
    """Judge ``claim`` against ``context`` → `Verdict`. Frozen signature; E10 fills real physical/calibration ones."""

    def verify(self, claim: dict, context: dict) -> Verdict:
        ...
