"""structured_decision: a forced tool call instead of free-text regex scraping for an LLM-as-loop-
controller decision. Regression coverage for a real bug: a model that reasons out loud, reaches a
wrong conclusion, catches itself, and writes a corrected conclusion later in the same response --
which a first-match text parser would silently prefer over the model's own final answer.
"""
import asyncio
import json

import httpx
import pytest
from mixle_mlops.core.decisions import StructuredDecisionError, structured_decision
from mixle_mlops.models.openai_compat import OpenAICompatAdapter

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["STOP", "CONTINUE"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
}


def _transport(handler):
    def wrapped(request: httpx.Request) -> httpx.Response:
        return handler(request)
    return httpx.MockTransport(wrapped)


def _capturing_transport(captured: dict, handler):
    def wrapped(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content) if request.content else {}
        return handler(request)
    return httpx.MockTransport(wrapped)


def test_forces_tool_choice_in_the_outgoing_request():
    """The exact fix, verified at the wire level: tool_choice must force this specific tool, not
    leave the model free to answer in prose it could later contradict."""
    captured: dict = {}

    def handler(_request):
        return httpx.Response(200, json={
            "id": "x", "model": "m", "choices": [{
                "message": {"role": "assistant", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "record_decision", "arguments": json.dumps({"decision": "STOP", "reason": "confident"})},
                }]},
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        })

    adapter = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_capturing_transport(captured, handler))
    result = asyncio.run(structured_decision(adapter, "should we stop?", DECISION_SCHEMA))

    assert captured["body"]["tool_choice"] == {"type": "function", "function": {"name": "record_decision"}}
    assert captured["body"]["tools"][0]["function"]["name"] == "record_decision"
    assert result == {"decision": "STOP", "reason": "confident"}


def test_the_original_bug_cannot_recur_with_a_forced_tool_call():
    """The old free-text failure mode: DeepSeek wrote "DECISION: CONTINUE ... I will correct:
    DECISION: STOP" in one response, and a first-match regex returned CONTINUE. A forced tool call
    has no equivalent "first draft" -- the backend returns exactly one function call. Simulated here
    by returning only the corrected decision as the (only) tool call, which is what a real forced
    tool_choice response looks like regardless of how much the model reasoned before emitting it."""
    def handler(_request):
        return httpx.Response(200, json={
            "id": "x", "model": "m", "choices": [{
                "message": {"role": "assistant", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "record_decision", "arguments": json.dumps({"decision": "STOP", "reason": "width 5.97 < threshold 8.0"})},
                }]},
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        })

    adapter = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_transport(handler))
    result = asyncio.run(structured_decision(adapter, "stop or continue?", DECISION_SCHEMA))
    assert result["decision"] == "STOP"  # not CONTINUE -- there is no discarded first draft to mis-parse


def test_raises_when_backend_ignores_tool_choice_and_returns_free_text():
    def handler(_request):
        return httpx.Response(200, json={
            "id": "x", "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "I think we should stop."}, "finish_reason": "stop"}],
            "usage": {},
        })

    adapter = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_transport(handler))
    with pytest.raises(StructuredDecisionError, match="free text"):
        asyncio.run(structured_decision(adapter, "stop or continue?", DECISION_SCHEMA))


def test_raises_on_malformed_tool_call_arguments():
    def handler(_request):
        return httpx.Response(200, json={
            "id": "x", "model": "m", "choices": [{
                "message": {"role": "assistant", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "record_decision", "arguments": "{not valid json"},
                }]},
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        })

    adapter = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_transport(handler))
    with pytest.raises(StructuredDecisionError, match="not valid JSON"):
        asyncio.run(structured_decision(adapter, "stop or continue?", DECISION_SCHEMA))
