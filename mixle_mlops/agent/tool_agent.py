"""An agentic tool-composition loop -- the piece that moves the *composition intelligence* out of a
human and into the framework.

Until now every pipeline in this codebase was hand-wired: a person (or an outer agent) chose which
tools to call, in what order, with what arguments, and wrote a bespoke catalog + orchestration each
time. That means the framework's "autonomy" was really the human's, and the scope of problems it could
address was exactly the set someone pre-wired. This closes that gap: give :func:`run_tool_agent` a
natural-language goal and a set of REAL Python functions, and an LLM (via any
:class:`~mixle_mlops.core.adapters.ModelAdapter`, e.g. the capability-routed text model) discovers
which tools apply, sequences them, calls them for real, observes the results, and iterates to an
answer -- composing a pipeline it was never pre-wired for.

Two design commitments, both learned the hard way in this project:

  * **Tools are introspected, not hand-specified.** :func:`introspect_tool` turns a function's
    signature + type hints + docstring into the tool schema the model sees, so registering a real
    capability is ``{fn.__name__: fn}`` -- no hand-written JSON schema to drift out of sync.
  * **Verification is a first-class step, not an afterthought.** An optional ``verifier`` gates the
    final answer (and can force the loop to keep working); a tool-composition agent with no verifier
    is a plausible-garbage generator, which is the failure mode this whole line of work exists to
    avoid.
"""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

from mixle_mlops.core.adapters import ChatMessage, ChatRequest, FunctionDef, ModelAdapter, ToolDef

__all__ = ["introspect_tool", "AgentStep", "AgentResult", "run_tool_agent"]

_PY_TO_JSON = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}

_DEFAULT_SYSTEM = (
    "You solve the user's goal by calling the provided tools. Call one or more tools, observe their "
    "results, and call more as needed. When you have enough to answer, stop calling tools and reply "
    "with the final answer in plain text, citing the concrete numbers the tools returned. Do not "
    "invent tool results; only use what the tools actually returned."
)


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    """Best-effort JSON-schema for one parameter's type hint. Unknown/unannotated -> string (the model
    can still pass a value; this never blocks registering a real function)."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    origin = typing.get_origin(annotation)
    if origin in (list, tuple):
        args = typing.get_args(annotation)
        item = _type_to_schema(args[0]) if args else {"type": "number"}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object"}
    return {"type": _PY_TO_JSON.get(annotation, "string")}


def introspect_tool(fn: Callable[..., Any], *, name: str | None = None) -> ToolDef:
    """Turn a real function into a :class:`~mixle_mlops.core.adapters.ToolDef` from its signature +
    type hints + docstring. The description is the docstring's first paragraph; required params are
    those with no default. ``self`` is skipped so bound methods introspect cleanly."""
    sig = inspect.signature(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, p in sig.parameters.items():
        if pname == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[pname] = _type_to_schema(p.annotation)
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    doc = (inspect.getdoc(fn) or "").strip()
    description = doc.split("\n\n", 1)[0].replace("\n", " ") if doc else (name or fn.__name__)
    return ToolDef(function=FunctionDef(
        name=name or fn.__name__, description=description,
        parameters={"type": "object", "properties": props, "required": required},
    ))


@dataclass
class AgentStep:
    """One executed tool call in the agent's trace: what it called, with what args, and what came back."""

    tool: str
    args: dict[str, Any]
    result: Any
    ok: bool


@dataclass
class AgentResult:
    """The outcome of an agent run: the final answer, the full tool-execution trace, the verifier's
    verdict (if any), and why it stopped."""

    answer: str | None
    steps: list[AgentStep] = field(default_factory=list)
    verdict: dict[str, Any] | None = None
    stopped_reason: str = ""  # "answered" | "max_steps" | "verifier_rejected"
    n_model_calls: int = 0


async def run_tool_agent(
    goal: str,
    tools: dict[str, Callable[..., Any]],
    adapter: ModelAdapter,
    *,
    max_steps: int = 8,
    system: str | None = None,
    verifier: Callable[[str, list[AgentStep]], dict[str, Any]] | None = None,
    temperature: float = 0.0,
) -> AgentResult:
    """Drive an LLM tool-use loop over real ``tools`` toward ``goal``.

    Each round: the model is given the goal + tool schemas (:func:`introspect_tool`) and either calls a
    tool -- which is executed for real and its result fed back as a ``role="tool"`` message -- or emits
    a final text answer. ``max_steps`` bounds tool-call rounds. If a ``verifier`` is supplied, the final
    answer must pass it; a rejected answer is fed back once with the verifier's reasons so the agent can
    correct, then re-checked (so verification actually shapes the result, not just labels it).

    Args:
        goal: the natural-language problem to solve.
        tools: ``{name: function}`` -- the real callables the agent may compose. Names must match the
            tool schema names (``introspect_tool`` uses ``fn.__name__`` unless overridden).
        adapter: any chat ``ModelAdapter`` whose backend supports OpenAI tool-calling (verified for
            DeepSeek). Get one from the capability layer: ``resolve_from_settings("text")``.
        max_steps: max tool-call rounds before giving up (``stopped_reason="max_steps"``).
        system: override the default system prompt.
        verifier: optional ``(answer, steps) -> {"passed": bool, "reasons": [...]}`` gate on the answer.
            May be sync or async (awaited if it returns a coroutine) -- so a verifier that itself calls
            a model, e.g. :class:`mixle_mlops.agent.verification.AgentAnswerVerifier`, works directly.
        temperature: decode temperature (0.0 = deterministic planning).

    Returns:
        An :class:`AgentResult` with the answer, the executed-tool trace, the verdict, and stop reason.
    """
    tool_defs = [introspect_tool(fn, name=name) for name, fn in tools.items()]
    messages = [
        ChatMessage(role="system", content=system or _DEFAULT_SYSTEM),
        ChatMessage(role="user", content=goal),
    ]
    steps: list[AgentStep] = []
    n_calls = 0
    verifier_retry_used = False

    for _ in range(max_steps + 1):
        req = ChatRequest(model=adapter.name, messages=messages, tools=tool_defs, tool_choice="auto", temperature=temperature)
        completion = await adapter.chat(req)
        n_calls += 1
        msg = completion.choices[0].message

        if msg.tool_calls:
            messages.append(ChatMessage(role="assistant", content=msg.content or "", tool_calls=msg.tool_calls))
            for tc in msg.tool_calls:
                tname = tc.function.name
                try:
                    targs = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    targs = {}
                fn = tools.get(tname)
                if fn is None:
                    result: Any = f"error: no such tool {tname!r}"
                    ok = False
                else:
                    try:
                        result = fn(**targs)
                        ok = True
                    except Exception as exc:  # noqa: BLE001 -- a tool error is an observation the agent can react to
                        result = f"error: {type(exc).__name__}: {exc}"
                        ok = False
                steps.append(AgentStep(tool=tname, args=targs, result=result, ok=ok))
                messages.append(ChatMessage(role="tool", tool_call_id=tc.id, content=json.dumps(result, default=str)))
            continue

        # a final text answer
        answer = msg.text()
        if verifier is not None:
            verdict = verifier(answer, steps)
            if inspect.isawaitable(verdict):  # a verifier may itself call a model (e.g. an adversarial critic)
                verdict = await verdict
            if not verdict.get("passed", False) and not verifier_retry_used:
                verifier_retry_used = True
                messages.append(ChatMessage(role="assistant", content=answer))
                messages.append(ChatMessage(
                    role="user",
                    content=f"That answer did not pass verification: {verdict.get('reasons')}. "
                            f"Use the tools to correct it, then answer again.",
                ))
                continue
            return AgentResult(
                answer=answer, steps=steps, verdict=verdict, n_model_calls=n_calls,
                stopped_reason="answered" if verdict.get("passed") else "verifier_rejected",
            )
        return AgentResult(answer=answer, steps=steps, verdict=None, n_model_calls=n_calls, stopped_reason="answered")

    return AgentResult(answer=None, steps=steps, verdict=None, n_model_calls=n_calls, stopped_reason="max_steps")
