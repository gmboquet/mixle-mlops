"""M1c DoD: target-aware model-context handoff + delta application.

Scripted scenario: model A is handed a graph/table/image bundle, annotates it by opening one discovery
gap (never touching the canonical items); a tool resolves that gap with a new typed-table item; model B
then receives a *different* capability rendering of the same underlying bundle (it is vision-capable,
model A was not) while every original item's canonical payload/hash survives untouched. Separately: a
stale delta, a privilege-escalating delta, a cross-caller-private item, and a free-prose "delta" are all
rejected rather than silently accepted or coerced.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mixle_knowledge.contracts import (
    PROPERTY_GRAPH_SCHEMA,
    SPATIAL_MEDIA_SCHEMA,
    TYPED_TABLE_SCHEMA,
    KnowledgeGapStatus,
)
from mixle_knowledge.kb.merge import StaleDeltaError
from mixle_knowledge.kb.store import CallerScope, StructuredKnowledgeStore, canonical_hash

from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatCompletion,
    ChatMessage,
    FunctionCall,
    ToolCall,
)
from mixle_mlops.knowledge.handoff import HandoffError, handoff
from mixle_mlops.models.domain_adapter import ProvenancedResult


# --- fakes -----------------------------------------------------------------------------------------------


class _FakeChatAdapter:
    """A minimal chat-shaped target: ``response_fn(req) -> ChatCompletion``."""

    def __init__(self, name: str, capabilities: set[str], response_fn) -> None:
        self._name = name
        self._capabilities = set(capabilities)
        self._response_fn = response_fn

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> set[str]:
        return set(self._capabilities)

    async def chat(self, req):
        return self._response_fn(req)


class _FakeToolAdapter:
    """A minimal IC-7-shaped target: no ``.chat``, only ``.call(inputs) -> ProvenancedResult``."""

    def __init__(self, name: str, payload: dict) -> None:
        self._name = name
        self._payload = payload

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> set[str]:
        return {"call"}

    async def call(self, inputs):
        return ProvenancedResult(value=self._payload, model_id=self._name, version="v1", content_hash="0" * 64)


class _FakeRegistry:
    def __init__(self, adapters: dict) -> None:
        self._adapters = adapters

    def has(self, name: str) -> bool:
        return name in self._adapters

    def get(self, name: str):
        return self._adapters[name]


def _tool_call_response(payload: dict):
    def _build(req):
        call = ToolCall(function=FunctionCall(name="propose_knowledge_delta", arguments=json.dumps(payload)))
        msg = ChatMessage(role="assistant", content=None, tool_calls=[call])
        return ChatCompletion(model=req.model, choices=[ChatChoice(message=msg)])

    return _build


def _prose_response(req):
    msg = ChatMessage(role="assistant", content="Here is my answer, in plain prose, no tool call at all.")
    return ChatCompletion(model=req.model, choices=[ChatChoice(message=msg)])


# --- fixtures ---------------------------------------------------------------------------------------------


def _graph_table_image_bundle(store: StructuredKnowledgeStore, *, project_id: str = "proj-1"):
    from mixle_knowledge.contracts import KnowledgeItem, Modality, ResourceKind

    graph_payload = {
        "nodes": [{"id": "n1", "type": "deposit"}],
        "edges": [{"id": "e1", "source": "n1", "target": "n1", "type": "refines"}],
    }
    graph = KnowledgeItem(
        id="graph-1",
        kind=ResourceKind.ARTIFACT,
        modality=Modality.GRAPH,
        schema_uri=PROPERTY_GRAPH_SCHEMA,
        payload=graph_payload,
        content_hash=canonical_hash(
            schema_uri=PROPERTY_GRAPH_SCHEMA,
            schema_version="1.0.0",
            payload=graph_payload,
            artifact_ref=None,
            metadata={},
        ),
    )

    table_payload = {
        "primary_key": ["sample_id"],
        "columns": [{"name": "sample_id", "type": "string"}, {"name": "cu_pct", "type": "float", "unit": "%"}],
        "rows": [{"sample_id": "A-1", "cu_pct": 1.8}],
    }
    table = KnowledgeItem(
        id="table-1",
        kind=ResourceKind.TABLE,
        modality=Modality.TABLE,
        schema_uri=TYPED_TABLE_SCHEMA,
        payload=table_payload,
        content_hash=canonical_hash(
            schema_uri=TYPED_TABLE_SCHEMA,
            schema_version="1.0.0",
            payload=table_payload,
            artifact_ref=None,
            metadata={},
        ),
    )

    image_bytes = b"totally-a-tiff"
    artifact_ref = store.artifacts.put(image_bytes, media_type="image/tiff")
    image_payload = {
        "crs": "EPSG:32611",
        "extent": [500000, 4100000, 501000, 4101000],
        "pixel_to_crs": [1, 0, 500000, 0, -1, 4101000],
    }
    image = KnowledgeItem(
        id="image-1",
        kind=ResourceKind.IMAGE,
        modality=Modality.IMAGE,
        schema_uri=SPATIAL_MEDIA_SCHEMA,
        media_type="image/tiff",
        artifact_ref=artifact_ref,
        payload=image_payload,
        content_hash=canonical_hash(
            schema_uri=SPATIAL_MEDIA_SCHEMA,
            schema_version="1.0.0",
            payload=image_payload,
            artifact_ref=artifact_ref,
            metadata={},
        ),
    )

    for item in (graph, table, image):
        store.put_item(item)
    bundle = store.materialize(
        [graph.id, table.id, image.id],
        project_id=project_id,
        task="rank targets",
        target_kind="model",
        target_id="model-a",
    )
    return bundle, graph, table, image


# --- the scripted two-handoff scenario ---------------------------------------------------------------------


def test_two_handoffs_preserve_canonical_payloads_with_different_capability_renderings(tmp_path):
    store = StructuredKnowledgeStore(tmp_path / "kb")
    bundle, graph, table, image = _graph_table_image_bundle(store)

    model_a = _FakeChatAdapter(
        "model-a",
        {"chat", "tools"},
        _tool_call_response(
            {
                "add_gaps": [
                    {
                        "id": "gap-assay",
                        "question": "Find the missing Cu assay for A-2",
                        "required_schema": {"type": "number"},
                        "acceptance_criteria": ["verified lab result"],
                    }
                ],
            }
        ),
    )

    new_table_payload = {
        "primary_key": ["sample_id"],
        "columns": [{"name": "sample_id", "type": "string"}, {"name": "cu_pct", "type": "float", "unit": "%"}],
        "rows": [{"sample_id": "A-2", "cu_pct": 2.1}],
    }
    new_hash = canonical_hash(
        schema_uri=TYPED_TABLE_SCHEMA,
        schema_version="1.0.0",
        payload=new_table_payload,
        artifact_ref=None,
        metadata={},
    )
    tool_resolver = _FakeToolAdapter(
        "tool-resolver",
        {
            "add_items": [
                {
                    "id": "table-assay-a2",
                    "kind": "table",
                    "modality": "table",
                    "schema_uri": TYPED_TABLE_SCHEMA,
                    "schema_version": "1.0.0",
                    "content_hash": new_hash,
                    "payload": new_table_payload,
                }
            ],
            "gap_updates": [
                {
                    "gap_id": "gap-assay",
                    "status": "resolved",
                    "resolved_by_item_ids": ["table-assay-a2"],
                    "attempt": {
                        "actor": "tool-resolver",
                        "query": "lookup missing Cu assay A-2",
                        "status": "resolved",
                        "produced_item_ids": ["table-assay-a2"],
                    },
                }
            ],
        },
    )

    note_payload = {"text": "Model B interpretation: viable given the resolved assay."}
    note_hash = canonical_hash(
        schema_uri="mixle://schema/text-note/1",
        schema_version="1.0.0",
        payload=note_payload,
        artifact_ref=None,
        metadata={},
    )
    model_b = _FakeChatAdapter(
        "model-b",
        {"chat", "tools", "vision"},
        _tool_call_response(
            {
                "add_items": [
                    {
                        "id": "note-b1",
                        "kind": "artifact",
                        "modality": "text",
                        "schema_uri": "mixle://schema/text-note/1",
                        "schema_version": "1.0.0",
                        "content_hash": note_hash,
                        "payload": note_payload,
                        "text_surface": note_payload["text"],
                    }
                ],
            }
        ),
    )

    registry = _FakeRegistry({"model-a": model_a, "tool-resolver": tool_resolver, "model-b": model_b})

    result_a = asyncio.run(
        handoff(
            bundle.id,
            target_model="model-a",
            question="Review this bundle; flag anything missing.",
            store=store,
            registry=registry,
        )
    )
    assert [g.id for g in result_a.delta.add_gaps] == ["gap-assay"]
    assert result_a.output_bundle.id == bundle.id
    assert result_a.output_bundle.revision == bundle.revision + 1
    gap = next(g for g in result_a.output_bundle.gaps if g.id == "gap-assay")
    assert gap.status == KnowledgeGapStatus.OPEN

    # model A never touched the canonical items it was handed
    orig_by_id = {i.id: i for i in bundle.items}
    for item in result_a.output_bundle.items:
        assert item.model_dump(mode="json") == orig_by_id[item.id].model_dump(mode="json")

    # model A has no vision -- the image item renders as an artifact ref, not a media part
    image_resource_a = next(r for r in result_a.rendered.resources if r["item_id"] == "image-1")
    assert image_resource_a["rendering"] == "artifact_ref"
    # graph/table items are structured resources (model A supports "tools")
    graph_resource_a = next(r for r in result_a.rendered.resources if r["item_id"] == "graph-1")
    assert graph_resource_a["rendering"] == "json_tool"
    assert graph_resource_a["content_hash"] == graph.content_hash
    assert graph_resource_a["payload"]["edges"][0]["id"] == "e1"  # canonical payload, never flattened

    result_tool = asyncio.run(
        handoff(
            bundle.id,
            target_model="tool-resolver",
            question="resolve the missing-assay gap",
            store=store,
            registry=registry,
        )
    )
    resolved_gap = next(g for g in result_tool.output_bundle.gaps if g.id == "gap-assay")
    assert resolved_gap.status == KnowledgeGapStatus.RESOLVED
    assert resolved_gap.resolved_by_item_ids == ["table-assay-a2"]
    assert any(i.id == "table-assay-a2" for i in result_tool.output_bundle.items)

    result_b = asyncio.run(
        handoff(
            bundle.id,
            target_model="model-b",
            question="Given the resolved assay, summarize the target.",
            store=store,
            registry=registry,
        )
    )
    # model B IS vision-capable -- same underlying image item, a different rendering hint
    image_resource_b = next(r for r in result_b.rendered.resources if r["item_id"] == "image-1")
    assert image_resource_b["rendering"] == "media_part"
    assert image_resource_a["rendering"] != image_resource_b["rendering"]

    # every original graph/table/image payload+hash is untouched all the way through both handoffs
    final_by_id = {i.id: i for i in result_b.output_bundle.items}
    assert final_by_id["graph-1"].content_hash == graph.content_hash
    assert final_by_id["graph-1"].payload == graph.payload
    assert final_by_id["table-1"].content_hash == table.content_hash
    assert final_by_id["image-1"].content_hash == image.content_hash
    assert final_by_id["image-1"].artifact_ref == image.artifact_ref
    # and the newly-added evidence/annotation items are both present
    assert "table-assay-a2" in final_by_id
    assert "note-b1" in final_by_id

    # receipts carry the IC-5 shape plus M1c's bundle/delta lineage
    assert result_b.receipt.provenance["source_bundle_id"] == bundle.id
    assert result_b.receipt.provenance["delta_hash"]


# --- rejections --------------------------------------------------------------------------------------------


def test_stale_delta_is_rejected(tmp_path):
    store = StructuredKnowledgeStore(tmp_path / "kb")
    bundle, *_ = _graph_table_image_bundle(store)

    bump = _FakeChatAdapter("bumper", {"chat", "tools"}, _tool_call_response({}))
    registry = _FakeRegistry({"bumper": bump})
    asyncio.run(handoff(bundle.id, target_model="bumper", question="q", store=store, registry=registry))

    stale = _FakeChatAdapter(
        "stale-model",
        {"chat", "tools"},
        _tool_call_response({"base_revision": bundle.revision}),  # explicitly the now-superseded revision
    )
    registry2 = _FakeRegistry({"stale-model": stale})
    with pytest.raises(StaleDeltaError):
        asyncio.run(handoff(bundle.id, target_model="stale-model", question="q", store=store, registry=registry2))


def test_free_prose_delta_is_rejected(tmp_path):
    store = StructuredKnowledgeStore(tmp_path / "kb")
    bundle, *_ = _graph_table_image_bundle(store)
    prose_model = _FakeChatAdapter("prose-model", {"chat"}, _prose_response)
    registry = _FakeRegistry({"prose-model": prose_model})
    with pytest.raises(HandoffError, match="free prose"):
        asyncio.run(handoff(bundle.id, target_model="prose-model", question="q", store=store, registry=registry))


def test_invalid_content_hash_is_rejected(tmp_path):
    store = StructuredKnowledgeStore(tmp_path / "kb")
    bundle, *_ = _graph_table_image_bundle(store)
    bad_model = _FakeChatAdapter(
        "bad-model",
        {"chat", "tools"},
        _tool_call_response(
            {
                "add_items": [
                    {
                        "id": "bad-item",
                        "kind": "artifact",
                        "modality": "text",
                        "schema_uri": "mixle://schema/text-note/1",
                        "schema_version": "1.0.0",
                        "content_hash": "f" * 64,
                        "payload": {"text": "tampered"},
                    }
                ],
            }
        ),
    )
    registry = _FakeRegistry({"bad-model": bad_model})
    with pytest.raises(ValueError, match="content_hash"):
        asyncio.run(handoff(bundle.id, target_model="bad-model", question="q", store=store, registry=registry))


def test_privilege_escalation_and_private_cross_caller_item_are_rejected(tmp_path):
    store = StructuredKnowledgeStore(tmp_path / "kb")
    bundle, graph, table, image = _graph_table_image_bundle(store)

    # A private item owned by "carol", added directly (not through a caller-scoped delta).
    from mixle_knowledge.contracts import AccessPolicy, AccessScope, KnowledgeItem, Modality, ResourceKind

    private_payload = {"text": "carol's private note"}
    private_item = KnowledgeItem(
        id="private-note",
        kind=ResourceKind.ARTIFACT,
        modality=Modality.TEXT,
        schema_uri="mixle://schema/text-note/1",
        payload=private_payload,
        content_hash=canonical_hash(
            schema_uri="mixle://schema/text-note/1",
            schema_version="1.0.0",
            payload=private_payload,
            artifact_ref=None,
            metadata={},
        ),
        access=AccessPolicy(scope=AccessScope.PRIVATE, owner="carol"),
    )
    store.put_item(private_item)
    bundle = store.materialize(
        [graph.id, table.id, image.id, private_item.id],
        project_id=bundle.project_id,
        task=bundle.task,
        target_kind=bundle.target_kind,
        target_id=bundle.target_id,
    )

    benign = _FakeChatAdapter("benign-model", {"chat", "tools"}, _tool_call_response({}))
    registry = _FakeRegistry({"benign-model": benign})
    alice = CallerScope(identity="alice")

    result = asyncio.run(
        handoff(
            bundle.id,
            target_model="benign-model",
            question="q",
            store=store,
            registry=registry,
            caller_scope=alice,
        )
    )
    # carol's private item never reached alice's rendering -- omitted, not silently dropped without a trace
    assert "private-note" not in result.rendered.preserved_item_ids
    assert "private-note" in result.rendered.omitted_item_ids
    assert all(r["item_id"] != "private-note" for r in result.rendered.resources)

    # A delta that tries to claim ownership of a *new* item under someone else's identity is rejected outright.
    escalate = _FakeChatAdapter(
        "escalate-model",
        {"chat", "tools"},
        _tool_call_response(
            {
                "add_items": [
                    {
                        "id": "bobs-item",
                        "kind": "artifact",
                        "modality": "text",
                        "schema_uri": "mixle://schema/text-note/1",
                        "schema_version": "1.0.0",
                        "content_hash": canonical_hash(
                            schema_uri="mixle://schema/text-note/1",
                            schema_version="1.0.0",
                            payload={"text": "mine now"},
                            artifact_ref=None,
                            metadata={},
                        ),
                        "payload": {"text": "mine now"},
                        "access": {"scope": "private", "owner": "bob"},
                    }
                ],
            }
        ),
    )
    registry2 = _FakeRegistry({"escalate-model": escalate})
    with pytest.raises(PermissionError):
        asyncio.run(
            handoff(
                bundle.id,
                target_model="escalate-model",
                question="q",
                store=store,
                registry=registry2,
                caller_scope=alice,
            )
        )
