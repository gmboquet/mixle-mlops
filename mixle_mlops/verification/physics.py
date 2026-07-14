"""E10 — physics and calibration verifiers (IC-6 `Verifier` implementations).

`PhysicalVerifier` checks a claimed subsurface field for basic physical admissibility: no negative
porosity/density, mass balance, and rock-physics-consistent moduli/velocities. `CalibrationVerifier`
checks a fitted posterior (IC-1) for statistical calibration: does its credible interval actually
cover held-out truth at the rate it claims to?

Both verifiers operate on plain dict claims/contexts. Neither hard-imports `mixle.reason.posterior_protocol`
at module load time — a posterior is duck-typed (it just needs `credible_interval(level)`), so this module
works whether or not that IC-1 stub has landed in the core `mixle` package yet.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Verdict

# Physically-admissible ranges for common rock-physics quantities. Generous enough to admit real
# reservoir/crustal rock across lithologies, tight enough to catch sign errors and unit mistakes.
_BOUNDS: dict[str, tuple[float, float]] = {
    "porosity": (0.0, 1.0),  # fraction of bulk volume
    "density": (800.0, 8000.0),  # kg/m^3: light sediment .. dense sulfide/metal ore
    "bulk_modulus": (0.0, 4.0e11),  # Pa
    "shear_modulus": (0.0, 4.0e11),  # Pa
    "vp": (300.0, 9000.0),  # m/s: air-filled soil .. fastest crustal rock
    "vs": (0.0, 6000.0),  # m/s (0 admits fluids/melts with no shear strength)
}


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _check_bounds(name: str, value: Any, lo: float, hi: float) -> tuple[bool, float, str | None]:
    """Bound-check one field; returns (ok, fraction_in_bounds, reason-if-any)."""
    arr = _as_array(value)
    if arr.size == 0:
        return True, 1.0, None
    bad = (arr < lo) | (arr > hi)
    if not bad.any():
        return True, 1.0, None
    frac_ok = 1.0 - float(bad.mean())
    offender = float(arr[bad].flat[0])
    reason = f"{name} out of physical bounds [{lo:g}, {hi:g}]: e.g. {offender:g} ({int(bad.sum())}/{arr.size} cells)"
    return False, frac_ok, reason


def _check_mass_balance(mass_balance: dict, tol: float) -> tuple[bool, float, str | None]:
    """inflow - outflow - storage_change should be ~0 within ``tol`` (relative to the flow scale)."""
    inflow = float(np.sum(_as_array(mass_balance.get("inflow", 0.0))))
    outflow = float(np.sum(_as_array(mass_balance.get("outflow", 0.0))))
    storage_change = float(np.sum(_as_array(mass_balance.get("storage_change", 0.0))))
    residual = inflow - outflow - storage_change
    scale = max(abs(inflow), abs(outflow), abs(storage_change), 1.0)
    if abs(residual) <= tol * scale:
        return True, 1.0, None
    reason = (
        f"mass balance violated: inflow {inflow:g} - outflow {outflow:g} - storage_change "
        f"{storage_change:g} = residual {residual:g} (tol {tol * scale:g})"
    )
    return False, max(0.0, 1.0 - abs(residual) / scale), reason


def _check_rock_physics(claim: dict) -> tuple[bool, float, str | None] | None:
    """Elementary rock-physics admissibility: shear velocity may not exceed compressional velocity."""
    if "vp" not in claim or "vs" not in claim:
        return None
    vp, vs = _as_array(claim["vp"]), _as_array(claim["vs"])
    if vp.shape != vs.shape:
        return False, 0.0, f"vp/vs shape mismatch: {vp.shape} vs {vs.shape}"
    bad = vs > vp
    if not bad.any():
        return True, 1.0, None
    frac_ok = 1.0 - float(bad.mean())
    return (
        False,
        frac_ok,
        f"rock-physics violation: vs exceeds vp at {int(bad.sum())}/{vp.size} cells (vs must be <= vp)",
    )


class PhysicalVerifier:
    """IC-6 `Verifier`: bounds, mass balance, and rock-physics admissibility for a claimed field.

    ``claim`` may carry any of ``porosity``, ``density``, ``bulk_modulus``, ``shear_modulus``, ``vp``,
    ``vs`` (arrays or scalars) and an optional ``mass_balance`` dict with ``inflow``/``outflow``/
    ``storage_change``. ``context`` may override any bound via ``context["bounds"][field] = (lo, hi)``
    and set ``context["mass_balance_tol"]`` (default ``1e-3``, relative to the flow scale). Only the
    fields actually present in ``claim`` are checked — an empty claim trivially passes.
    """

    def verify(self, claim: dict, context: dict) -> Verdict:
        context = context or {}
        overrides = context.get("bounds") or {}
        fractions: list[float] = []
        reasons: list[str] = []
        passed = True

        for field_name, default_bound in _BOUNDS.items():
            if field_name not in claim:
                continue
            lo, hi = overrides.get(field_name, default_bound)
            ok, frac_ok, reason = _check_bounds(field_name, claim[field_name], lo, hi)
            fractions.append(frac_ok)
            if not ok:
                passed = False
                reasons.append(reason)

        mass_balance = claim.get("mass_balance")
        if mass_balance is not None:
            tol = float(context.get("mass_balance_tol", 1e-3))
            ok, frac_ok, reason = _check_mass_balance(mass_balance, tol)
            fractions.append(frac_ok)
            if not ok:
                passed = False
                reasons.append(reason)

        rock_physics = _check_rock_physics(claim)
        if rock_physics is not None:
            ok, frac_ok, reason = rock_physics
            fractions.append(frac_ok)
            if not ok:
                passed = False
                reasons.append(reason)

        score = float(np.mean(fractions)) if fractions else 1.0
        return Verdict(passed=passed, score=score, reasons=reasons, kind="physical")


class CalibrationVerifier:
    """IC-6 `Verifier`: empirical coverage of an IC-1 `Posterior`'s credible interval against held-out truth.

    ``claim`` carries ``posterior`` (anything satisfying the IC-1 `Posterior` protocol — duck-typed, not
    imported, so this works whether or not the core `mixle.reason.posterior_protocol` module is present)
    and, optionally, ``nominal`` (the claimed coverage level, default ``0.9``). ``context`` carries
    ``truth`` (an array of held-out true values, one per posterior component) and, optionally, ``tol``
    (the allowed shortfall below nominal before the verdict fails, default ``0.05``).

    Coverage is the fraction of ``truth`` components that fall inside the posterior's per-component
    ``credible_interval(nominal)``. This is the standard PIT-adjacent calibration check: a posterior
    that is overconfident (its stated uncertainty is narrower than its actual error against truth)
    covers truth less often than its nominal level promises.
    """

    def verify(self, claim: dict, context: dict) -> Verdict:
        context = context or {}
        posterior = claim.get("posterior")
        if posterior is None or not hasattr(posterior, "credible_interval"):
            return Verdict(
                passed=False,
                score=0.0,
                reasons=["claim['posterior'] is missing or does not implement credible_interval(level)"],
                kind="calibration",
            )

        nominal = float(claim.get("nominal", context.get("nominal", 0.9)))
        tol = float(claim.get("tol", context.get("tol", 0.05)))
        if "truth" not in context:
            return Verdict(
                passed=False,
                score=0.0,
                reasons=["context['truth'] (held-out truth) is required"],
                kind="calibration",
            )
        truth = _as_array(context["truth"])

        lo, hi = posterior.credible_interval(nominal)
        lo, hi = _as_array(lo), _as_array(hi)
        covered = (truth >= lo) & (truth <= hi)
        coverage = float(np.mean(covered)) if covered.size else 0.0

        passed = coverage >= nominal - tol
        reasons: list[str] = []
        if not passed:
            reasons.append(
                f"empirical coverage {coverage:.3f} at nominal {nominal:.2f} (tol {tol:.2f}) is below the "
                f"{nominal - tol:.3f} floor -- the posterior is overconfident relative to held-out truth"
            )
        return Verdict(passed=passed, score=coverage, reasons=reasons, kind="calibration")
