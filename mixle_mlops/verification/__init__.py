"""Verification: the `Verifier`/`Verdict` protocol (IC-6) plus the physics/calibration checks that fill it (E10)."""

from .base import Verdict, Verifier, VERDICT_KINDS
from .physics import CalibrationVerifier, PhysicalVerifier

__all__ = ["Verdict", "Verifier", "VERDICT_KINDS", "PhysicalVerifier", "CalibrationVerifier"]
