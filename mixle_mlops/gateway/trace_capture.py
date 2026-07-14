"""Turn on `ExecutionTrace`/`Receipt` capture in the gateway (work-plan M4a).

Every agentic turn the gateway serves -- whether driven by `mixle.task.orchestrate.orchestrate` against
a `World`, or by the gateway's own OpenAI-style model-tool-calling loop (`agent_loop.run_agent_loop` over
a :class:`~mixle_mlops.gateway.tool_registry.ToolRegistry`) -- should land in the same durable feed the
M4 dataset foundry mines. This module is the one place that binds a
:class:`~mixle.task.replay.ExecutionTrace` into a :class:`~mixle.inference.receipt.Receipt` and emits the
IC-5 frozen JSON envelope: ``{prompt, steps: [{tool, args, result, model, verdict}], outcome, provenance}``.

`capture_turn` is the thin wrapper over `orchestrate` the work order specifies. `capture_steps` is the
sibling entry point for the gateway's tool-calling loop, which has no `World`/`orchestrate` involved at
all -- `ToolRegistry` now records a `TraceStep` per dispatched call (`record_tool_call`, wired in by this
same task), and `chat.py`'s agentic-turn handler binds that step list into a `Receipt` the same way. Both
paths share `_stamp_trace` (the `model`/`verdict` stamping) and `_persist` (the substrate write), so a
trace mined by M4 never cares which loop produced it.

`TraceStep` (`mixle.task.replay.ExecutionTrace`, `replay.py:23`) carries only ``tool``/``args``/``seed``/
``result`` -- no `model`/`verdict` fields, and it is frozen (non-goals: no schema redesign). So `model`
and an optional IC-6 `Verdict` (as a plain dict) are stashed under the reserved
``args["_model"]``/``args["_verdict"]`` keys and pulled back out (never left behind) when `to_foundry_row`
lifts a step into the IC-5 shape.
"""

from __future__ import annotations

from typing import Any

from mixle.inference.receipt import Receipt
from mixle.task.orchestrate import World, orchestrate
from mixle.task.replay import ExecutionTrace, TraceStep

# Reserved keys stashed onto TraceStep.args to carry the calling model + (optional) verdict; TraceStep
# itself has no such fields (frozen -- replay.py:23), so `to_foundry_row` pops them back out of args
# rather than leaving them mixed in with the tool's real arguments.
_MODEL_KEY = "_model"
_VERDICT_KEY = "_verdict"


def _verdict_dict(
    verifier: Any, tool: str, args: dict[str, Any], result: Any, question: str
) -> dict[str, Any] | None:
    """Run an IC-6 `Verifier` over one step and return its `Verdict` as a plain dict, or ``None`` when no
    verifier is configured (a step never claims a verdict it wasn't actually given)."""
    if verifier is None:
        return None
    verdict = verifier.verify({"tool": tool, "args": args, "result": result}, {"question": question})
    return {"passed": verdict.passed, "score": verdict.score, "reasons": list(verdict.reasons), "kind": verdict.kind}


def record_tool_call(
    tool: str, args: dict[str, Any], result: Any, *, model_id: str, verifier: Any = None, question: str = ""
) -> TraceStep:
    """Build one `TraceStep` for a single dispatched tool call, with `model_id` and an optional `Verdict`
    stamped under ``args["_model"]``/``args["_verdict"]``. `ToolRegistry.dispatch` calls this for every
    registered tool it runs, so the gateway's model-tool-calling loop accumulates the same trace shape
    `capture_turn` builds from `orchestrate`."""
    stamped = dict(args)
    stamped[_MODEL_KEY] = model_id
    stamped[_VERDICT_KEY] = _verdict_dict(verifier, tool, args, result, question)
    return TraceStep(tool=tool, args=stamped, result=result)


def _stamp_trace(trace: ExecutionTrace, *, model_id: str, verifier: Any = None) -> None:
    """Stamp `model`/`verdict` onto every step of `trace` in place (idempotent: a step `record_tool_call`
    already stamped is left as-is)."""
    for step in trace.steps:
        step.args.setdefault(_MODEL_KEY, model_id)
        if _VERDICT_KEY not in step.args:
            step.args[_VERDICT_KEY] = _verdict_dict(verifier, step.tool, step.args, step.result, trace.request)


def _persist(receipt: Receipt, sink: Any) -> str | None:
    """Persist `receipt` into `sink` as a ``kind="trace"`` substrate item -- the durable M4 foundry feed
    (`mixle.substrate.ingest` shapes every harvested trace this way; there is no single-record ingest
    helper today, so the equivalent `SubstrateItem` is built directly). ``None`` sink is a no-op: capture
    without a foundry to write to is still useful (the DoD path), just not durable."""
    if sink is None:
        return None
    from mixle.substrate.core import SubstrateItem

    row = to_foundry_row(receipt)
    item = SubstrateItem(
        kind="trace", text=str(row.get("prompt", "")), payload=row, provenance=dict(receipt.provenance)
    )
    return sink.put(item)


def capture_turn(
    question: str,
    plan_model,
    world: World,
    *,
    budget: int,
    model_id: str,
    verifier: Any = None,
    sink: Any = None,
    bundle_id: str | None = None,
    base_revision: int | None = None,
) -> Receipt:
    """Plan+execute one orchestrated turn (`mixle.task.orchestrate.orchestrate`), bind the resulting
    `ExecutionTrace` into a `Receipt`, and (when `sink` is given) persist it as a foundry trace item.

    ``provenance`` carries ``bundle_id``/``base_revision`` -- named to mirror IC-13's
    `KnowledgeDelta.base_bundle_id`/`base_revision` -- so a later task (M1c) can attach output bundle/delta
    ids onto the same provenance dict without touching IC-5's frozen top-level keys.
    """
    result = orchestrate(question, plan_model, world, budget=budget)
    _stamp_trace(result.trace, model_id=model_id, verifier=verifier)
    provenance = {"bundle_id": bundle_id, "base_revision": base_revision, "stopped_reason": result.stopped_reason}
    receipt = Receipt(answer=result.answer, produced_by=model_id, trace=result.trace, provenance=provenance)
    _persist(receipt, sink)
    return receipt


def capture_steps(
    question: str,
    steps: list[TraceStep],
    answer: Any,
    *,
    model_id: str,
    verifier: Any = None,
    sink: Any = None,
    bundle_id: str | None = None,
    base_revision: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> Receipt:
    """The `capture_turn` sibling for a turn that never went through `orchestrate`/`World` at all: the
    gateway's OpenAI-style model-tool-calling loop (`agent_loop.run_agent_loop`). `steps` is whatever
    `ToolRegistry.trace_steps` accumulated this turn (each already stamped by `record_tool_call`); this
    wraps them into an `ExecutionTrace` and binds the same `Receipt` + foundry row `capture_turn` would.
    """
    trace = ExecutionTrace(request=question, steps=list(steps))
    _stamp_trace(trace, model_id=model_id, verifier=verifier)
    merged_provenance = {"bundle_id": bundle_id, "base_revision": base_revision}
    if provenance:
        merged_provenance.update(provenance)
    receipt = Receipt(answer=answer, produced_by=model_id, trace=trace, provenance=merged_provenance)
    _persist(receipt, sink)
    return receipt


def to_foundry_row(receipt: Receipt) -> dict[str, Any]:
    """Emit the IC-5 shape: ``{prompt, steps: [{tool, args, result, model, verdict}], outcome, provenance}``.

    `trace.request` -> ``prompt``, `receipt.answer` -> ``outcome``, `receipt.provenance` -> ``provenance``;
    each step's stashed ``_model``/``_verdict`` are popped back out of ``args`` into their own ``model``/
    ``verdict`` keys, so ``args`` in the emitted row is exactly the tool's real call arguments.
    """
    trace = receipt.trace
    steps: list[dict[str, Any]] = []
    for step in trace.steps if trace is not None else []:
        args = dict(step.args)
        model = args.pop(_MODEL_KEY, None)
        verdict = args.pop(_VERDICT_KEY, None)
        steps.append({"tool": step.tool, "args": args, "result": step.result, "model": model, "verdict": verdict})
    return {
        "prompt": trace.request if trace is not None else "",
        "steps": steps,
        "outcome": receipt.answer,
        "provenance": dict(receipt.provenance),
    }
