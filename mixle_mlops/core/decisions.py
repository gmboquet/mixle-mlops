"""Force a single, atomic, schema-validated decision out of an LLM -- the fix for a real observed
failure mode in an LLM-as-loop-controller pattern: a model asked to reason and answer in the same
free-text response can talk itself into a wrong answer, catch its own mistake, and write a corrected
answer later in the *same* response. A regex/string parser that takes the first matching token then
returns the model's discarded first draft instead of its considered final one -- exactly what
happened in practice (see ``experiments/adaptive-gravity-survey-design`` in the top-level repo:
DeepSeek wrote "DECISION: CONTINUE ... I will correct: DECISION: STOP", and a first-match parser
returned CONTINUE).

A forced tool call sidesteps this class of bug entirely: the backend commits the model to exactly one
atomic, schema-shaped answer per call. There is no "first draft" to accidentally parse, because the
model cannot emit two tool calls expressing two different decisions the way it can write two
paragraphs of free text.
"""

from __future__ import annotations

import json
from typing import Any

from .adapters import ChatMessage, ChatRequest, FunctionDef, ModelAdapter, ToolDef

__all__ = ["StructuredDecisionError", "structured_decision"]


class StructuredDecisionError(Exception):
    """Raised when the backend doesn't return the forced tool call (ignores tool_choice, calls the
    wrong tool, or returns malformed arguments) -- surfaced explicitly rather than silently falling
    back to a guessed default, which would just trade one silent-failure mode for another."""


async def structured_decision(
    adapter: ModelAdapter,
    prompt: str,
    schema: dict[str, Any],
    *,
    model: str = "default",
    tool_name: str = "record_decision",
    tool_description: str = "Record your decision. This is the only way to answer -- there is no free-text response.",
    **request_kwargs: Any,
) -> dict[str, Any]:
    """Ask ``adapter`` a question and force it to answer via one schema-validated tool call.

    ``schema`` is a JSON-Schema ``object`` (the tool's ``parameters``) describing exactly the fields
    the decision needs -- e.g. ``{"type": "object", "properties": {"decision": {"type": "string",
    "enum": ["STOP", "CONTINUE"]}, "reason": {"type": "string"}}, "required": ["decision", "reason"]}``.
    ``request_kwargs`` are forwarded to :class:`~mixle_mlops.core.adapters.ChatRequest` (e.g.
    ``temperature=0.0``).

    Returns the parsed ``arguments`` dict from the model's tool call. Raises
    :class:`StructuredDecisionError` if the backend doesn't honor the forced tool choice or returns
    something unparseable -- callers should let this propagate (or catch it and retry/escalate)
    rather than silently defaulting, which would reintroduce the exact failure mode this exists to fix.
    """
    tool = ToolDef(function=FunctionDef(name=tool_name, description=tool_description, parameters=schema))
    req = ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        **request_kwargs,
    )
    completion = await adapter.chat(req)
    message = completion.choices[0].message
    if not message.tool_calls:
        raise StructuredDecisionError(
            f"expected a forced call to {tool_name!r} but got free text instead "
            f"(backend may not support tool_choice): {message.content!r}"
        )
    call = message.tool_calls[0]
    if call.function.name != tool_name:
        raise StructuredDecisionError(f"model called {call.function.name!r}, expected the forced {tool_name!r}")
    try:
        return json.loads(call.function.arguments)
    except json.JSONDecodeError as exc:
        raise StructuredDecisionError(f"tool-call arguments were not valid JSON: {call.function.arguments!r}") from exc
