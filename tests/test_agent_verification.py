"""General agent-answer verification: numeric grounding (deterministic) + an adversarial model critic
that flags claims not supported by the executed-tool trace. Critic is exercised offline with a scripted
structured-decision (forced tool-call) response; the live 'catches a hand-computed number' proof runs
separately."""

import asyncio
import json

import httpx
import pytest

from mixle_mlops.agent.tool_agent import AgentStep
from mixle_mlops.agent.verification import AgentAnswerVerifier, numeric_grounding
from mixle_mlops.models.openai_compat import OpenAICompatAdapter


def _steps():
    return [
        AgentStep(tool="mine_npv", args={"grade_pct": 1.8}, result={"npv_mean_million": 127.6, "prob_npv_positive": 0.971}, ok=True),
        AgentStep(tool="fuse_estimates", args={"values": [1.7, 1.9]}, result={"fused_estimate": 1.767, "fused_std": 0.183}, ok=True),
    ]


def test_numeric_grounding_passes_when_every_number_traces_to_a_tool():
    answer = "The NPV is 127.6 million with 0.971 probability positive; the fused estimate is 1.767 (std 0.183)."
    g = numeric_grounding(answer, _steps())
    assert g["grounded"] is True
    assert g["ungrounded_numbers"] == []


def test_numeric_grounding_flags_a_number_the_tools_never_produced():
    # 999.9 appears nowhere in any tool result/args -> ungrounded
    answer = "The NPV is 127.6 million, but I also estimate a hidden upside of 999.9 million."
    g = numeric_grounding(answer, _steps())
    assert g["grounded"] is False
    assert 999.9 in g["ungrounded_numbers"]


def test_numeric_grounding_tolerates_inputs_and_small_integers():
    answer = "Using the 2 lab estimates 1.7 and 1.9, and step 1's grade of 1.8, the result holds."
    g = numeric_grounding(answer, _steps())
    assert g["grounded"] is True  # 2 (small int), 1.7/1.9 (args), 1.8 (arg) all allowed


def _critic_transport(supported: bool, unsupported: list):
    """MockTransport returning a structured_decision forced tool-call with the critic's verdict."""
    def handler(_request):
        return httpx.Response(200, json={"id": "x", "model": "m", "choices": [{
            "message": {"role": "assistant", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "record_decision", "arguments": json.dumps({"supported": supported, "unsupported_claims": unsupported})}}]},
            "finish_reason": "tool_calls"}], "usage": {}})
    return httpx.MockTransport(handler)


def test_verifier_passes_when_the_critic_finds_everything_supported():
    critic = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_critic_transport(True, []))
    verifier = AgentAnswerVerifier(critic)
    verdict = asyncio.run(verifier("NPV 127.6M, fused 1.767.", _steps()))
    assert verdict["passed"] is True


def test_verifier_fails_when_the_critic_flags_an_unsupported_claim():
    critic = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_critic_transport(False, ["fused estimate 1.9 was computed by hand, not returned by fuse_estimates"]))
    verifier = AgentAnswerVerifier(critic)
    verdict = asyncio.run(verifier("The fused estimate is 1.9 (I averaged them).", _steps()))
    assert verdict["passed"] is False
    assert "computed by hand" in verdict["reasons"][0]


def test_verifier_fails_when_too_few_tools_ran():
    critic = OpenAICompatAdapter("m", base_url="http://x/v1", transport=_critic_transport(True, []))
    verifier = AgentAnswerVerifier(critic, require_min_tools=3)
    verdict = asyncio.run(verifier("answer", _steps()))  # only 2 tools ran
    assert verdict["passed"] is False
    assert "need >= 3" in verdict["reasons"][0]
