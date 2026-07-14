"""M1c -- target-aware model-context handoff + delta application (work order M1c; IC-5/IC-6/IC-7/IC-10/IC-13).

`handoff` is the single place a `KnowledgeBundle` moves from the store to a target model/tool and back:

  1. load the exact source bundle revision through the M1a store, gated by ``caller_scope`` (project/team/
     owner access) -- any item the caller cannot see is simply absent from ``bundle.items`` already, and the
     ids that fell away are folded into the returned rendering's ``omitted_item_ids`` (never their content);
  2. render it for whatever ``registry.get(target_model)`` actually advertises via `render.render_bundle`;
  3. call the target and *require* a structured `KnowledgeDelta` back -- a chat-shaped adapter is forced
     through a single required tool call (an IC-13 JSON-schema-typed function), an IC-7 `DomainModelAdapter`-
     shaped target is called directly and must itself return a delta-shaped mapping; free prose from either
     is rejected outright, never coerced into a delta;
  4. hand the delta to `mixle_knowledge.kb.merge.apply_delta` (M2a) -- optimistic concurrency, hash
     re-verification, access-escalation and conflict handling all live there, not duplicated here, so a stale
     ``base_revision``, a widened-access item, or an unresolved conflict claim is rejected the same way for
     every caller of `apply_delta`, `handoff` included;
  5. bind the new bundle id/revision and a hash of the applied delta into an IC-5-shaped receipt
     (`mixle_mlops.gateway.trace_capture`), and return everything the caller needs to keep chaining handoffs
     (`HandoffResult.output_bundle.id` is the next call's ``bundle_id``).

`handoff` never mutates the source bundle or its media: every write goes through `apply_delta`, which only
ever *appends* a new bundle revision (M1a's store is itself append-only), so a source revision handed to one
model stays exactly reproducible for the next.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .render import RenderedContext, render_bundle

if TYPE_CHECKING:  # pragma: no cover - typing only; the real imports are lazy (see module docstring)
    from mixle_knowledge.contracts import KnowledgeDelta

__all__ = ["HandoffResult", "HandoffError", "DELTA_TOOL_NAME", "handoff"]

DELTA_TOOL_NAME = "propose_knowledge_delta"


class HandoffError(Exception):
    """Raised for a handoff-specific failure: an unregistered target, a target that answered in free prose
    instead of a structured delta, or a delta payload that does not even parse as an IC-13 `KnowledgeDelta`.

    Deliberately distinct from `mixle_knowledge.kb.merge.apply_delta`'s own typed errors (``StaleDeltaError``,
    ``PermissionError``, ``ValueError``): those are the M2a write-path rejections (stale revision, access
    escalation, invalid hash, unresolved conflict) and are left to propagate as-is, since they are already
    specific and well-typed -- wrapping them here would only make them harder to match on.
    """


@dataclass
class HandoffResult:
    """The frozen M1c handoff outcome: what was rendered, the delta the target produced, the resulting
    bundle, and its IC-5-shaped receipt."""

    source_bundle_id: str
    rendered: RenderedContext
    delta: Any
    output_bundle: Any
    receipt: Any


def _capabilities_for(target: Any) -> set[str]:
    """Duck-typed capability set for whatever ``registry.get(target_model)`` returned: a real
    `ModelAdapter`/`DomainModelAdapter` (``.capabilities()``), or a bare IC-10 `CatalogEntry`-shaped object
    (schema/owner only, no live adapter) synthesized defensively from its ``owner`` -- so a target reached
    through the router/tool catalog (IC-10) renders sensibly even without a live adapter object."""
    caps = getattr(target, "capabilities", None)
    if callable(caps):
        return set(caps())
    owner = getattr(target, "owner", None)
    if owner in ("physics", "climate", "economic", "external"):
        return {"call"}
    if owner == "model":
        return {"chat", "tools"}
    return {"chat"}


def _delta_tool_schema() -> dict[str, Any]:
    from mixle_knowledge.contracts import KnowledgeDelta

    return {
        "type": "function",
        "function": {
            "name": DELTA_TOOL_NAME,
            "description": (
                "Propose a KnowledgeDelta patch against the exact bundle revision you were handed. This is "
                "the ONLY way to write back evidence, annotations, or discovery gaps -- never answer in "
                "free prose when this tool is available."
            ),
            "parameters": KnowledgeDelta.model_json_schema(),
        },
    }


async def _invoke_target(
    target: Any, *, rendered: RenderedContext, question: str, bundle_id: str, base_revision: int
) -> dict[str, Any]:
    """Call ``target`` and return its raw (unvalidated) delta payload as a dict, rejecting free prose.

    Two duck-typed shapes: an IC-7 `DomainModelAdapter`-like object (``.call(inputs)``, expected to itself
    emit a delta-shaped mapping -- "a tool resolves it") and an ordinary chat `ModelAdapter`
    (``.chat(ChatRequest)``), forced through one required `KnowledgeDelta` tool call.
    """
    call = getattr(target, "call", None)
    chat = getattr(target, "chat", None)

    if callable(call) and not callable(chat):
        result = await call(
            {
                "question": question,
                "bundle_id": bundle_id,
                "base_revision": base_revision,
                "messages": rendered.messages,
                "resources": rendered.resources,
            }
        )
        value = getattr(result, "value", result)
        if not isinstance(value, dict):
            raise HandoffError(
                f"target {getattr(target, 'name', target)!r} returned a free-form "
                f"{type(value).__name__}, not a structured KnowledgeDelta payload"
            )
        return value

    if not callable(chat):
        raise HandoffError(f"target {target!r} exposes neither .call(...) nor .chat(...)")

    from ..core.adapters import ChatMessage, ChatRequest

    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in rendered.messages]
    messages.append(ChatMessage(role="user", content=question))
    req = ChatRequest(
        model=getattr(target, "name", ""),
        messages=messages,
        tools=[_delta_tool_schema()],
        tool_choice={"type": "function", "function": {"name": DELTA_TOOL_NAME}},
    )
    completion = await chat(req)
    if not completion.choices:
        raise HandoffError("target model returned no choices")
    message = completion.choices[0].message
    if not message.tool_calls:
        raise HandoffError("target model answered in free prose instead of calling the required KnowledgeDelta tool")
    raw_args = message.tool_calls[0].function.arguments
    try:
        payload = json.loads(raw_args)
    except (TypeError, ValueError) as exc:
        raise HandoffError(f"target model's delta tool-call arguments were not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HandoffError("target model's delta tool-call arguments were not a JSON object")
    return payload


def _delta_hash(delta: "KnowledgeDelta") -> str:
    canonical = json.dumps(delta.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def handoff(
    bundle_id: str,
    *,
    target_model: str,
    question: str,
    store: Any,
    registry: Any,
    verifier: Any = None,
    caller_scope: Any = None,
) -> HandoffResult:
    """Hand ``bundle_id`` off to ``target_model``: render for its declared capabilities, require a
    structured `KnowledgeDelta` in reply, apply it (M2a `apply_delta`), and materialize the result.

    ``store`` is an `mixle_knowledge.kb.store.StructuredKnowledgeStore`; ``registry`` is anything exposing
    ``has(name)``/``get(name)`` over `ModelAdapter`/`DomainModelAdapter`-shaped targets (an
    `mixle_mlops.core.registry.ModelRegistry`, in practice). Raises `HandoffError` for a handoff-specific
    failure, or lets `apply_delta`'s own typed errors (stale revision, access escalation, invalid hash,
    unresolved conflict) propagate unchanged.
    """
    from mixle_knowledge.contracts import KnowledgeDelta
    from mixle_knowledge.kb.merge import apply_delta

    from ..gateway.trace_capture import capture_steps

    bundle = store.get_bundle(bundle_id, caller_scope=caller_scope)

    access_omitted: list[str] = []
    if caller_scope is not None:
        try:
            unrestricted = store.get_bundle(bundle_id, revision=bundle.revision, caller_scope=None)
            access_omitted = sorted({i.id for i in unrestricted.items} - {i.id for i in bundle.items})
        except Exception:
            access_omitted = []

    if not hasattr(registry, "has") or not registry.has(target_model):
        raise HandoffError(f"target model {target_model!r} is not registered")
    target = registry.get(target_model)
    capabilities = _capabilities_for(target)

    rendered = render_bundle(
        bundle, capabilities=capabilities, token_budget=bundle.token_budget, byte_budget=bundle.byte_budget
    )
    if access_omitted:
        rendered.omitted_item_ids = sorted(set(rendered.omitted_item_ids) | set(access_omitted))

    raw_delta = await _invoke_target(
        target, rendered=rendered, question=question, bundle_id=bundle.id, base_revision=bundle.revision
    )
    raw_delta.setdefault("base_bundle_id", bundle.id)
    raw_delta.setdefault("base_revision", bundle.revision)
    raw_delta.setdefault("produced_by", target_model)
    try:
        delta: KnowledgeDelta = KnowledgeDelta.model_validate(raw_delta)
    except Exception as exc:
        raise HandoffError(f"target {target_model!r} produced an invalid KnowledgeDelta: {exc}") from exc

    output_bundle = apply_delta(store, delta, verifier=verifier, caller_scope=caller_scope)

    receipt = capture_steps(
        question,
        [],
        delta.model_dump(mode="json"),
        model_id=target_model,
        bundle_id=output_bundle.id,
        base_revision=delta.base_revision,
        provenance={
            "source_bundle_id": bundle.id,
            "delta_hash": _delta_hash(delta),
            "preserved_item_ids": rendered.preserved_item_ids,
            "omitted_item_ids": rendered.omitted_item_ids,
        },
    )

    return HandoffResult(
        source_bundle_id=bundle.id,
        rendered=rendered,
        delta=delta,
        output_bundle=output_bundle,
        receipt=receipt,
    )
