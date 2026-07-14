"""F4 -- surface calibrated UQ + decision receipts in every served answer (work-plan §5).

The gateway's agentic loop (``agent_loop.run_agent_loop`` over a :class:`~.tool_registry.ToolRegistry`)
may dispatch a physics/decision tool -- E4's ``query_posterior`` (IC-3), backed by A5's decision-quantity
API (IC-8) -- whose result already carries a distribution summary (``value``/``ci``/``level``), the
honesty flag ``prior_dominated`` (IC-1/IC-8; IC-3 requires it ALWAYS be present on a ``query_posterior``
result), and a receipt/knowledge-lineage handle (E7/IC-5, M1c/IC-13). This module's only job is to lift
that already-computed result into a :class:`~mixle_mlops.core.adapters.DecisionQuantity` and gate the
served natural-language content through it -- it never computes a posterior/CI (A5) or mints a receipt
(E7); it only surfaces what they already produced.

Two invariants this module enforces on every turn that dispatched a decision-quantity tool call:

  1. a driller-facing scalar never leaves the route without its credible interval and its
     receipt/knowledge-item lineage (:func:`decision_quantity_from_result` raises
     :class:`DecisionSurfacingError` rather than silently dropping them);
  2. a prior-dominated (or explicitly un-conformal) decision never surfaces as a bare point estimate in
     ``message.content`` -- :func:`surface_decisions` replaces the served text with a calibrated abstain,
     reusing E5's :class:`~mixle.reason.language_bridge.Claim` to still name the honest range.
"""

from __future__ import annotations

from typing import Any

from mixle.reason.language_bridge import Claim

from ..core.adapters import ChatChoice, DecisionQuantity

# A tool result is a decision quantity exactly when it carries this flag -- per IC-3, `query_posterior`
# ALWAYS returns `prior_dominated`, so keying off its presence (rather than a fixed tool-name allowlist)
# lets any future decision-shaped tool opt into surfacing just by returning the same honesty flag.
_DECISION_MARKER = "prior_dominated"


class DecisionSurfacingError(ValueError):
    """A tool result declared itself a decision quantity (it carries ``prior_dominated``) but is
    missing the ``ci``/receipt/knowledge-item lineage a driller-facing scalar must carry before it is
    allowed to leave the route. Raised rather than silently degrading to a bare, uncalibrated number."""


def _is_decision_result(result: Any) -> bool:
    return isinstance(result, dict) and _DECISION_MARKER in result


def _as_ci(raw: Any) -> tuple[float, float]:
    lo, hi = raw
    return float(lo), float(hi)


def decision_quantity_from_result(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> DecisionQuantity:
    """Lift one decision-shaped tool result into a :class:`DecisionQuantity`.

    Raises :class:`DecisionSurfacingError` when the result lacks ``ci`` or the receipt/knowledge-item
    lineage -- this never invents a placeholder interval or drops the honesty flag to keep a turn
    superficially clean; a decision that cannot carry its own provenance does not get served.
    """
    if "ci" not in result:
        raise DecisionSurfacingError(f"{tool_name!r} result declares prior_dominated but no ci: {result!r}")
    receipt_ref = result.get("receipt_ref")
    knowledge_item_id = result.get("knowledge_item_id")
    if receipt_ref is None or knowledge_item_id is None:
        raise DecisionSurfacingError(
            f"{tool_name!r} result is missing receipt_ref/knowledge_item_id lineage: {result!r}"
        )
    name = str(result.get("name") or args.get("query") or tool_name)
    return DecisionQuantity(
        name=name,
        value=float(result.get("value", 0.0)),
        ci=_as_ci(result["ci"]),
        level=float(result.get("level", 0.9)),
        prior_dominated=bool(result[_DECISION_MARKER]),
        units=str(result.get("units", "")),
        receipt_ref=receipt_ref,
        knowledge_item_id=knowledge_item_id,
        bundle_id=result.get("bundle_id"),
        bundle_revision=result.get("bundle_revision"),
    )


def extract_decisions(trace_steps: list[Any]) -> list[DecisionQuantity]:
    """Every decision-quantity tool result this turn's trace steps carry, lifted into
    :class:`DecisionQuantity`\\ s in call order (a turn may dispatch more than one)."""
    return [
        decision_quantity_from_result(step.tool, step.args, step.result)
        for step in trace_steps
        if _is_decision_result(step.result)
    ]


def _abstain_text(dq: DecisionQuantity) -> str:
    """The calibrated abstain phrasing for one decision -- reuses E5's ``Claim`` to render the honest
    interval, so an abstain still names the range instead of withholding all information."""
    claim = Claim(field=dq.name, lo=dq.ci[0], hi=dq.ci[1])
    units = f" {dq.units}" if dq.units else ""
    return (
        f"I can't state a calibrated point value for {dq.name}: the posterior is prior-dominated "
        f"(the data hasn't constrained it beyond the prior yet), so a single number would be false "
        f"precision. The honest range at the {dq.level:.0%} level is {claim.lo:.4g} to {claim.hi:.4g}"
        f"{units} -- treat it as provisional pending more evidence."
    )


def _claim_fails(dq: DecisionQuantity, result: dict[str, Any]) -> bool:
    """Whether the source tool result itself flagged its declared claim as un-conformal
    (``claim_ok: False``). This never recomputes calibration (A5/E5's job) -- it only relays an
    honesty signal the upstream decision-quantity call already computed, mirroring E5's
    ``PosteriorDescriber``/``ABSTAIN`` contract (describe() -> None when no candidate claim clears the
    conformal threshold) without redoing that scoring here."""
    return result.get("claim_ok") is False


def surface_decisions(choice: ChatChoice, trace_steps: list[Any]) -> list[DecisionQuantity]:
    """The single entry point ``chat.py`` calls after an agentic turn: lift every decision-quantity tool
    result the turn dispatched into ``choice.decisions``, and gate ``choice.message.content`` through
    them. A prior-dominated (or explicitly un-conformal) decision replaces the served text with a
    calibrated abstain instead of a bare point estimate; ``choice.decisions`` still carries the full
    interval + receipt + knowledge-item lineage either way, so a calibrated caller can always hydrate
    the canonical distribution/UQ rather than treat the displayed text as the record.

    Returns the surfaced decisions (already attached to ``choice.decisions``).
    """
    decisions: list[DecisionQuantity] = []
    abstain_texts: list[str] = []
    for step in trace_steps:
        result = step.result
        if not _is_decision_result(result):
            continue
        dq = decision_quantity_from_result(step.tool, step.args, result)
        decisions.append(dq)
        if dq.prior_dominated or _claim_fails(dq, result):
            dq.prior_dominated = True  # never a bare point estimate leaves without the honesty flag set
            abstain_texts.append(_abstain_text(dq))
    choice.decisions = decisions
    if abstain_texts:
        choice.message.content = "\n".join(abstain_texts)
    return decisions
