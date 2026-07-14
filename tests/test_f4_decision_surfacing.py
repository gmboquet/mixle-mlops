"""F4 -- surface calibrated UQ + decision receipts in every served answer.

Covers the two layers `decision_surfacing.py` adds:

  * unit-level: lifting a decision-quantity tool result (as `query_posterior`/IC-3 would return) into a
    `DecisionQuantity`, the missing-lineage guard, and the prior-dominated -> abstain gate;
  * route-level: a stub physics tool dispatched through the gateway's agentic loop (`extra.agent`) ends
    up on `choices[0].decisions` with its full interval + receipt + knowledge-item + bundle lineage, and
    a prior-dominated decision replaces the served text with a calibrated abstain -- never a bare
    tonnage number -- while the surfaced `knowledge_item_id` still lets a caller hydrate the full
    distribution/UQ out of band.
"""

from __future__ import annotations

import asyncio

import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    DecisionQuantity,
    FunctionCall,
    FunctionDef,
    ModelAdapter,
    ToolCall,
    ToolDef,
)
from mixle_mlops.gateway.decision_surfacing import (
    DecisionSurfacingError,
    decision_quantity_from_result,
    extract_decisions,
    surface_decisions,
)
from mixle_mlops.gateway.tool_registry import ToolRegistry

# --- a stub physics/decision tool result, shaped like IC-3's `query_posterior` (E4) reading an IC-8
# decision quantity off a posterior: a distribution summary + the ALWAYS-present `prior_dominated` flag
# + content-hashed receipt/knowledge lineage (E7/M1c). F4 does not compute any of this -- it only lifts
# an already-produced result like this one.
_TONNAGE_RESULT = {
    "name": "tonnage_above_cutoff",
    "value": 1250.0,
    "ci": [980.0, 1520.0],
    "level": 0.9,
    "prior_dominated": True,
    "units": "tonnes",
    "receipt_ref": "sha256:abc123deadbeef",
    "knowledge_item_id": "ki-tonnage-001",
    "bundle_id": "kb-project-42",
    "bundle_revision": 3,
}

_CONFIDENT_RESULT = {**_TONNAGE_RESULT, "prior_dominated": False, "name": "net_pay"}


class _Step:
    """Minimal `mixle.task.replay.TraceStep` stand-in (tool/args/result), avoiding a torch/mixle.task
    import just for the shape this module reads."""

    def __init__(self, tool, args, result):
        self.tool = tool
        self.args = args
        self.result = result


# ---------------------------------------------------------------------------
# unit-level: lifting + gating
# ---------------------------------------------------------------------------


def test_f4_decision_surfacing_lifts_ci_receipt_and_lineage():
    dq = decision_quantity_from_result("query_posterior", {"query": "region_mass"}, _TONNAGE_RESULT)
    assert dq.ci == (980.0, 1520.0)
    assert dq.prior_dominated is True
    assert dq.receipt_ref == "sha256:abc123deadbeef"
    assert dq.knowledge_item_id == "ki-tonnage-001"
    assert dq.bundle_id == "kb-project-42"
    assert dq.bundle_revision == 3


def test_f4_decision_surfacing_extracts_only_decision_shaped_steps():
    steps = [
        _Step("rag_search", {"query": "x"}, {"results": []}),  # not a decision quantity
        _Step("query_posterior", {"query": "region_mass"}, _TONNAGE_RESULT),
    ]
    decisions = extract_decisions(steps)
    assert len(decisions) == 1
    assert decisions[0].name == "tonnage_above_cutoff"


def test_f4_decision_surfacing_rejects_missing_lineage():
    incomplete = {k: v for k, v in _TONNAGE_RESULT.items() if k != "knowledge_item_id"}
    with pytest.raises(DecisionSurfacingError):
        decision_quantity_from_result("query_posterior", {}, incomplete)


def test_f4_decision_surfacing_abstains_when_prior_dominated():
    choice = ChatChoice(message=ChatMessage(role="assistant", content="The tonnage is about 1250 tonnes."))
    steps = [_Step("query_posterior", {"query": "region_mass"}, _TONNAGE_RESULT)]
    decisions = surface_decisions(choice, steps)

    assert len(decisions) == 1 and choice.decisions == decisions
    dq = decisions[0]
    assert dq.ci == (980.0, 1520.0) and dq.prior_dominated is True
    assert dq.receipt_ref and dq.knowledge_item_id and dq.bundle_id and dq.bundle_revision == 3
    # never a bare point estimate: the raw value must not appear, and the text is a calibrated abstain
    assert "1250" not in choice.message.content
    assert "prior-dominated" in choice.message.content
    assert "980" in choice.message.content and "1520" in choice.message.content


def test_f4_decision_surfacing_passes_through_when_calibrated():
    original = "Net pay over the region is 1250 units, 90% CI 980-1520."
    choice = ChatChoice(message=ChatMessage(role="assistant", content=original))
    steps = [_Step("query_posterior", {"query": "net_pay"}, _CONFIDENT_RESULT)]
    decisions = surface_decisions(choice, steps)

    assert len(decisions) == 1 and decisions[0].prior_dominated is False
    assert choice.message.content == original  # a calibrated decision is not overwritten


def test_f4_decision_surfacing_hydrates_full_distribution_from_knowledge_item():
    """The DoD's hydration check: a caller holding only `knowledge_item_id`/`bundle_id`/`bundle_revision`
    (as surfaced on the DecisionQuantity) can look up the full canonical distribution/UQ -- F4 never
    makes the displayed CI the record of truth, only a pointer to it."""
    # stands in for the M1c knowledge store this ultimately backs onto; keyed exactly by the ids F4 surfaced
    knowledge_store = {
        "ki-tonnage-001": {
            "bundle_id": "kb-project-42",
            "bundle_revision": 3,
            "distribution": {"samples": [910.0, 1005.0, 1240.0, 1480.0, 1510.0], "ci": [980.0, 1520.0], "level": 0.9},
        }
    }
    choice = ChatChoice(message=ChatMessage(role="assistant", content="placeholder"))
    dq = surface_decisions(choice, [_Step("query_posterior", {}, _TONNAGE_RESULT)])[0]

    item = knowledge_store[dq.knowledge_item_id]
    assert item["bundle_id"] == dq.bundle_id
    assert item["bundle_revision"] == dq.bundle_revision
    assert item["distribution"]["ci"] == list(dq.ci)
    assert len(item["distribution"]["samples"]) == 5  # the full UQ, not just the displayed scalar


# ---------------------------------------------------------------------------
# route-level: through the gateway's agentic loop
# ---------------------------------------------------------------------------


class _QueryPosteriorAdapter(ModelAdapter):
    """A model that calls `query_posterior` once, then drafts a bare-number answer from the tool
    result -- exactly the kind of answer F4 must gate before it reaches the driller."""

    kind = "llm"
    name = "posteriorbot"

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        tool_msgs = [m for m in req.messages if m.role == "tool"]
        if tool_msgs:
            return ChatCompletion(
                model=req.model,
                choices=[
                    ChatChoice(
                        message=ChatMessage(role="assistant", content="The tonnage is about 1250 tonnes."),
                        finish_reason="stop",
                    )
                ],
            )
        return ChatCompletion(
            model=req.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(function=FunctionCall(name="query_posterior", arguments="{}"))],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

    async def stream(self, req: ChatRequest):
        completion = await self.chat(req)
        yield ChatCompletionChunk(
            model=req.model,
            choices=[
                ChatChunkChoice(
                    delta=ChoiceDelta(role="assistant", content=completion.choices[0].message.text()),
                    finish_reason="stop",
                )
            ],
        )


async def _stub_query_posterior(args):
    return dict(_TONNAGE_RESULT)


def _install_query_posterior_tool(monkeypatch):
    """Register a stub `query_posterior` tool onto every `ToolRegistry` built for the test -- standing
    in for E4's real physics-tool wiring (`mcp/physics_tools.py`, not yet landed), which is out of
    scope for F4 (surfacing only)."""
    original_build = ToolRegistry._build

    def patched_build(self, *args, **kwargs):
        # Forward whatever ToolRegistry._build's real signature currently expects (it has grown
        # positional params, e.g. D7's `include_platform`, since this stub was first written) so this
        # fixture keeps working across upstream signature changes instead of hard-coding an arity.
        original_build(self, *args, **kwargs)
        self._add(
            ToolDef(
                function=FunctionDef(
                    name="query_posterior",
                    description="stub decision-quantity tool",
                    parameters={"type": "object", "properties": {}},
                )
            ),
            _stub_query_posterior,
        )

    monkeypatch.setattr(ToolRegistry, "_build", patched_build)


def test_f4_decision_surfacing_tool_registry_dispatch_is_decision_shaped():
    """Sanity-check the stub tool dispatches to exactly the IC-3 result shape used above, independent
    of the FastAPI route plumbing."""
    from mixle_mlops.core.registry import ModelRegistry
    from mixle_mlops.models import EchoAdapter

    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    tools = ToolRegistry(reg, user_id=None)
    tools._add(ToolDef(function=FunctionDef(name="query_posterior", parameters={})), _stub_query_posterior)
    result = asyncio.run(tools.dispatch("query_posterior", {}))
    assert result["prior_dominated"] is True
    assert tools.trace_steps[-1].tool == "query_posterior"
    assert tools.trace_steps[-1].result["knowledge_item_id"] == "ki-tonnage-001"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    from mixle_mlops.config import get_settings
    from mixle_mlops.gateway.app import create_app

    _install_query_posterior_tool(monkeypatch)
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        app.state.registry.register(_QueryPosteriorAdapter())
        yield c
    get_settings.cache_clear()
    db._engine = None


def _auth_headers(client, email):
    raw = client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def test_f4_decision_surfacing_end_to_end_agent_turn_carries_decisions_and_abstains(client):
    headers = _auth_headers(client, "f4@t.com")
    r = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "posteriorbot",
            "extra": {"agent": True},
            "messages": [{"role": "user", "content": "what is the tonnage above cutoff?"}],
        },
    )
    assert r.status_code == 200
    choice = r.json()["choices"][0]

    decisions = choice["decisions"]
    assert len(decisions) == 1
    dq = decisions[0]
    assert dq["ci"] == [980.0, 1520.0]
    assert dq["level"] == 0.9
    assert dq["prior_dominated"] is True
    assert dq["receipt_ref"] == "sha256:abc123deadbeef"
    assert dq["knowledge_item_id"] == "ki-tonnage-001"
    assert dq["bundle_id"] == "kb-project-42"
    assert dq["bundle_revision"] == 3

    content = choice["message"]["content"]
    assert "1250" not in content  # never a bare point estimate
    assert "prior-dominated" in content
    assert "980" in content and "1520" in content  # the honest range is still named

    # DecisionQuantity round-trips through the wire shape unchanged
    DecisionQuantity.model_validate(dq)


def test_f4_decision_surfacing_end_to_end_streaming_agent_turn_also_abstains(client):
    headers = _auth_headers(client, "f4-stream@t.com")
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "posteriorbot",
            "stream": True,
            "extra": {"agent": True},
            "messages": [{"role": "user", "content": "tonnage?"}],
        },
    ) as s:
        lines = [ln for ln in s.iter_lines() if ln and ln.startswith("data:")]
    body = "\n".join(lines)
    assert "1250" not in body
    assert "prior-dominated" in body
