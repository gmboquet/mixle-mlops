"""General, domain-agnostic verification for agent answers -- the binding-constraint layer.

A tool-composition agent is only as trustworthy as what checks it, and the autonomous-agent runs
surfaced the exact failure this exists to catch: lacking the right tool, the agent computed a number
*in its own head* and stated it confidently, and a weak per-demo rule-based verifier passed it. This
module is a *general* gate -- it works for any agent task, not one hand-written per demo:

  * :func:`numeric_grounding` -- a cheap, deterministic first pass: does every number in the answer
    trace to a tool result or input? (Heuristic; tolerant of rounding/restated inputs.)
  * :func:`adversarial_critique` -- the authoritative check: a *separate* model, prompted
    adversarially to FIND unsupported quantitative claims (default to flagging when unsure), returns a
    structured verdict via :func:`~mixle_mlops.core.decisions.structured_decision` (a forced tool call,
    so the critic can't waffle its way to a pass). This is what catches "reasoned instead of verified."
  * :class:`AgentAnswerVerifier` -- combines them into an (async) ``(answer, steps) -> verdict``
    callable you pass straight to :func:`~mixle_mlops.agent.tool_agent.run_tool_agent` as its
    ``verifier``; a rejected answer gets the loop's one correction round, so verification *shapes* the
    result rather than just labeling it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mixle_mlops.agent.tool_agent import AgentStep
from mixle_mlops.core.decisions import structured_decision

__all__ = ["numeric_grounding", "adversarial_critique", "AgentAnswerVerifier"]

_NUM = re.compile(r"-?\d+\.?\d*")


def _collect_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                out.append(float(m))
            except ValueError:
                pass


def numeric_grounding(answer: str, steps: list[AgentStep], *, rel_tol: float = 0.02) -> dict[str, Any]:
    """Cheap, deterministic check: is every number in ``answer`` matched (within ``rel_tol``) by a
    number in some executed tool's args or result?

    Grounded set = all numbers appearing in the args and results of tools that ran. A number in the
    answer that matches none of them is 'ungrounded'. Heuristic by nature (an answer legitimately
    computing ``mean/2`` would flag), so it is a *signal*, not the authority -- :func:`adversarial_critique`
    is. Returns ``{"grounded": bool, "ungrounded_numbers": [...]}``.
    """
    grounded: list[float] = []
    for s in steps:
        if s.ok:
            _collect_numbers(s.args, grounded)
            _collect_numbers(s.result, grounded)
    grounded_set = grounded

    ungrounded: list[float] = []
    for m in _NUM.findall(answer):
        try:
            val = float(m)
        except ValueError:
            continue
        if any(abs(val - g) <= (rel_tol * abs(g) + 1e-9) or (abs(g) < 1e-9 and abs(val) < 1e-9) for g in grounded_set):
            continue
        # allow small integers / trivially-common values (list indices, "two labs", years-as-labels)
        if val == int(val) and abs(val) <= 12:
            continue
        ungrounded.append(val)
    return {"grounded": not ungrounded, "ungrounded_numbers": ungrounded}


_CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean", "description": "true if and only if EVERY quantitative claim in the answer is directly supported by a tool result in the trace"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}, "description": "each quantitative claim in the answer that is NOT backed by a tool result (empty list if fully supported)"},
    },
    "required": ["supported", "unsupported_claims"],
}


async def adversarial_critique(answer: str, steps: list[AgentStep], adapter: Any) -> dict[str, Any]:
    """Ask a separate model, adversarially, whether every quantitative claim in ``answer`` is supported
    by the executed-tool trace. Returns ``{"supported": bool, "unsupported_claims": [...]}`` via a forced
    tool call, so the critic commits to a structured verdict instead of a hedge."""
    trace = [{"tool": s.tool, "args": s.args, "result": s.result} for s in steps if s.ok]
    prompt = (
        "You are an ADVERSARIAL verifier. Find every quantitative claim in the ANSWER that is NOT "
        "directly supported by a tool result in the TRACE. A number the answer computed itself, "
        "restated incorrectly, or invented is UNSUPPORTED. Only a number that literally appears in a "
        "tool's result (or is that exact result trivially restated) counts as supported. When unsure, "
        "flag it. Be strict.\n\n"
        f"TRACE (tools that actually ran, with args and results):\n{json.dumps(trace, default=str)}\n\n"
        f"ANSWER to check:\n{answer}"
    )
    return await structured_decision(adapter, prompt, _CRITIQUE_SCHEMA, model=getattr(adapter, "name", "default"), temperature=0.0)


class AgentAnswerVerifier:
    """A general, model-backed agent-answer verifier -- pass an instance as ``run_tool_agent``'s
    ``verifier``. It requires a minimum number of tools to have run and then runs the adversarial
    critique; an answer with any unsupported claim fails (and gets the loop's correction round)."""

    def __init__(self, critic_adapter: Any, *, require_min_tools: int = 1, use_numeric_precheck: bool = True) -> None:
        self.critic_adapter = critic_adapter
        self.require_min_tools = int(require_min_tools)
        self.use_numeric_precheck = use_numeric_precheck

    async def __call__(self, answer: str, steps: list[AgentStep]) -> dict[str, Any]:
        ran = [s for s in steps if s.ok]
        if len(ran) < self.require_min_tools:
            return {"passed": False, "reasons": [f"answer is grounded in only {len(ran)} executed tool(s); need >= {self.require_min_tools}"]}

        critique = await adversarial_critique(answer, steps, self.critic_adapter)
        if not critique.get("supported", False):
            claims = critique.get("unsupported_claims") or ["unspecified unsupported claim"]
            return {"passed": False, "reasons": ["adversarial critic flagged unsupported claims: " + "; ".join(map(str, claims))]}

        precheck = numeric_grounding(answer, steps) if self.use_numeric_precheck else {"grounded": True, "ungrounded_numbers": []}
        return {
            "passed": True,
            "reasons": ["all quantitative claims are tool-grounded (adversarial critic passed)"],
            "numeric_precheck": precheck,
        }
