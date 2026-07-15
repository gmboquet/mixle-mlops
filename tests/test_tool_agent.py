"""Agentic tool-composition loop: introspection turns real functions into tool schemas, and the loop
executes real tools / threads results / iterates to an answer. Loop mechanics are tested offline with
scripted tool-call responses (httpx.MockTransport); the live "DeepSeek composes a novel pipeline"
proof is run separately, not in the test suite (needs network + a key)."""

import asyncio
import json

import httpx
import pytest

from mixle_mlops.agent.tool_agent import AgentResult, introspect_tool, run_tool_agent
from mixle_mlops.models.openai_compat import OpenAICompatAdapter


# --- real tools the agent composes ---
def mean_and_std(values: list[float]) -> dict:
    """Return the mean and standard deviation of a list of numbers."""
    import statistics
    return {"mean": statistics.fmean(values), "std": statistics.pstdev(values)}


def probability_above(values: list[float], threshold: float) -> float:
    """Fraction of values strictly above a threshold."""
    return sum(1 for v in values if v > threshold) / len(values)


def test_introspect_tool_builds_schema_from_signature_and_docstring():
    tool = introspect_tool(probability_above)
    assert tool.function.name == "probability_above"
    assert "above a threshold" in tool.function.description
    props = tool.function.parameters["properties"]
    assert props["values"] == {"type": "array", "items": {"type": "number"}}
    assert props["threshold"] == {"type": "number"}
    assert set(tool.function.parameters["required"]) == {"values", "threshold"}


def _scripted_transport(responses: list[dict]):
    """A MockTransport that returns the given OpenAI-shaped completion bodies in order."""
    state = {"i": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        body = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _tool_call_response(call_id, name, arguments):
    return {"id": "x", "model": "m", "choices": [{
        "message": {"role": "assistant", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        "finish_reason": "tool_calls"}], "usage": {}}


def _final_response(text):
    return {"id": "x", "model": "m", "choices": [
        {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": {}}


def test_loop_executes_real_tools_threads_results_and_returns_final_answer():
    responses = [
        _tool_call_response("c1", "mean_and_std", {"values": [1, 2, 3, 4]}),
        _tool_call_response("c2", "probability_above", {"values": [1, 2, 3, 4], "threshold": 2.5}),
        _final_response("The mean is 2.5 and half the values exceed 2.5."),
    ]
    adapter = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_scripted_transport(responses))
    result = asyncio.run(run_tool_agent(
        "Summarize [1,2,3,4] and say what fraction exceed 2.5.",
        {"mean_and_std": mean_and_std, "probability_above": probability_above}, adapter,
    ))
    assert isinstance(result, AgentResult)
    assert result.stopped_reason == "answered"
    # the loop executed BOTH real tools, in order, with real results:
    assert [s.tool for s in result.steps] == ["mean_and_std", "probability_above"]
    assert result.steps[0].result == {"mean": 2.5, "std": pytest.approx(1.1180, abs=1e-3)}
    assert result.steps[1].result == 0.5  # real computation, not the model's claim
    assert result.steps[0].ok and result.steps[1].ok
    assert "2.5" in result.answer


def test_a_tool_error_is_captured_as_an_observation_not_a_crash():
    def broken(x: int) -> int:
        """Always raises."""
        raise ValueError("boom")

    responses = [_tool_call_response("c1", "broken", {"x": 1}), _final_response("The tool failed.")]
    adapter = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_scripted_transport(responses))
    result = asyncio.run(run_tool_agent("do it", {"broken": broken}, adapter))
    assert result.steps[0].ok is False
    assert "boom" in str(result.steps[0].result)
    assert result.stopped_reason == "answered"  # loop kept going after the tool error


def test_verifier_rejects_then_the_agent_gets_one_correction_round():
    # first answer is unverifiable; the mock's next answer passes.
    responses = [
        _final_response("I think it's probably fine."),
        _final_response("Verified: the mean is 2.5."),
    ]
    adapter = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_scripted_transport(responses))

    def verifier(answer, steps):
        ok = "2.5" in answer
        return {"passed": ok, "reasons": ["answer must cite the computed mean 2.5"]}

    result = asyncio.run(run_tool_agent("what is the mean of [1,2,3,4]?", {"mean_and_std": mean_and_std}, adapter, verifier=verifier))
    assert result.verdict["passed"] is True
    assert result.stopped_reason == "answered"
    assert result.n_model_calls == 2  # it used its one correction round


def test_max_steps_bounds_a_tool_calling_loop_that_never_finishes():
    # the model keeps calling a tool forever; the loop must terminate.
    loop_response = _tool_call_response("c", "noop", {})
    adapter = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_scripted_transport([loop_response]))
    result = asyncio.run(run_tool_agent("spin", {"noop": lambda: "ok"}, adapter, max_steps=3))
    assert result.stopped_reason == "max_steps"
    assert result.answer is None
    assert len(result.steps) >= 3
