"""M1c -- render an IC-13 `KnowledgeBundle` for one target's declared capabilities (work order M1c).

`render_bundle` is the only place a bundle's canonical items become model-facing content, and it never
flattens a graph/table/spatial-media payload into one opaque context string -- that would undo the entire
point of M1a/M2a keeping those payloads structured. Every item becomes exactly one of:

  * a budgeted, inlined `text_surface` fragment (plain text items only);
  * an ephemeral, capability-gated media resource descriptor (image items): a vision/image-capable target's
    descriptor is tagged ``rendering="media_part"`` so the caller resolves it into a real signed/data URL at
    the actual adapter boundary (the same "descriptor now, bytes only at the boundary" discipline
    :class:`mixle_mlops.multimodal.content.StructuredMediaRef` already follows for D5) -- ``render_bundle``
    itself never touches a blob store and never inlines pixels;
  * a typed JSON-schema resource for a graph/table/structured item (``rendering="json_tool"``) when the
    target can consume tool/resource channels, or a constrained inline JSON block (``rendering="json_block"``)
    when it can't -- either way the canonical payload is *also* always carried, verbatim, in `resources`, so
    it is never *only* prompt state;
  * a field/spatial array (raster/geospatial/timeseries/vector modality) -- always ``rendering="artifact_ref"``,
    regardless of capabilities: a physics field is too large, and too easy to silently truncate/corrupt, to
    ever become a bare inline blob.

`token_budget`/`byte_budget` gate what a caller can afford to *read in the prompt*: the compact per-item
index line and any inlined text/JSON fragment. They never remove an item's structured `resources` entry --
an item a target can't afford to read inline can still reach it as a tool result/attachment, so nothing a
budget can't afford silently vanishes; an item is only budget-omitted (`omitted_item_ids`) when there truly
is no room left even for its one-line index entry once at least one item has already been preserved (an
empty rendering is never returned just because the very first item is large).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only; the real import is lazy, matching this repo's convention
    from mixle_knowledge.contracts import KnowledgeBundle

__all__ = ["RenderedContext", "render_bundle"]

_TEXT_MODALITIES = {"text", "document", "code"}
_IMAGE_MODALITIES = {"image"}
_FIELD_MODALITIES = {"raster", "geospatial", "timeseries", "vector"}
_STRUCTURED_MODALITIES = {"table", "graph", "structured"}

# Capability-string vocabulary actually emitted by this repo's adapters (core/adapters.py subclasses,
# models/domain_adapter.py): "chat"/"tools" for LLM backends, "vision"/"image" for multimodal ones, "call"
# for an IC-7 `DomainModelAdapter`. Treated as an open set (unknown strings are just ignored) rather than
# a frozen enum, since new adapters may advertise new capability strings over time.
_MEDIA_CAPS = {"vision", "image"}
_AUDIO_CAPS = {"audio"}
_STRUCTURED_CHANNEL_CAPS = {"tools", "call"}


@dataclass
class RenderedContext:
    """One capability-aware rendering of a `KnowledgeBundle`. ``messages`` are plain ``{"role","content"}``
    dicts (trivially adaptable into a backend's own chat-message type); ``resources`` are JSON-serializable
    descriptors of every non-purely-textual item, carried out-of-band from the prompt."""

    messages: list[Any] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    preserved_item_ids: list[str] = field(default_factory=list)
    omitted_item_ids: list[str] = field(default_factory=list)
    bytes_used: int = 0
    tokens_used: int = 0


def _estimate_tokens(text: str) -> int:
    """A crude, backend-agnostic token estimate (~4 chars/token). Good enough to budget by; never claimed
    exact against any particular tokenizer."""
    return max(1, (len(text) + 3) // 4)


def _byte_len(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def _bucket(modality: str) -> str:
    if modality in _IMAGE_MODALITIES:
        return "image"
    if modality in _FIELD_MODALITIES:
        return "field"
    if modality in _STRUCTURED_MODALITIES:
        return "structured"
    return "text"  # conservative default: audio/video/model/simulator/unknown render as inert text refs


def _rendering_hint(bucket: str, capabilities: set[str]) -> str:
    if bucket == "image":
        if capabilities & _MEDIA_CAPS:
            return "media_part"
        return "artifact_ref"
    if bucket == "field":
        return "artifact_ref"
    if bucket == "structured":
        if capabilities & _STRUCTURED_CHANNEL_CAPS:
            return "json_tool"
        return "json_block"
    return "text_surface"


def _short_summary(item: Any, bucket: str) -> str:
    payload = item.payload
    if bucket == "structured" and isinstance(payload, dict):
        if "nodes" in payload or "edges" in payload:
            return f"{len(payload.get('nodes') or [])} nodes / {len(payload.get('edges') or [])} edges"
        if "rows" in payload or "columns" in payload:
            return f"{len(payload.get('rows') or [])} rows x {len(payload.get('columns') or [])} cols"
    if bucket == "image":
        return item.media_type or "image"
    if bucket == "field":
        crs = payload.get("crs") if isinstance(payload, dict) else None
        return f"field ({crs or 'no crs'})"
    if item.text_surface:
        snippet = item.text_surface.strip().replace("\n", " ")
        return snippet[:80] + ("..." if len(snippet) > 80 else "")
    return bucket


def _index_line(item: Any, bucket: str, rendering: str) -> str:
    return (
        f"- [{item.id}] modality={item.modality} rendering={rendering} "
        f"hash={item.content_hash[:12]} rev={item.revision}: {_short_summary(item, bucket)}"
    )


def _resource_descriptor(item: Any, bucket: str, rendering: str) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "item_id": item.id,
        "kind": item.kind,
        "modality": item.modality,
        "schema_uri": item.schema_uri,
        "schema_version": item.schema_version,
        "content_hash": item.content_hash,
        "revision": item.revision,
        "rendering": rendering,
        "media_type": item.media_type,
        "artifact_ref": item.artifact_ref,
    }
    # Prefer referencing an existing artifact rather than duplicating a (possibly large) payload; inline
    # the canonical payload only when there is nothing else to point at.
    descriptor["payload"] = item.payload if item.artifact_ref is None else None
    return descriptor


def _gap_lines(bundle: "KnowledgeBundle") -> list[str]:
    return [f"- gap[{gap.id}] status={gap.status} priority={gap.priority}: {gap.question}" for gap in bundle.gaps]


def render_bundle(
    bundle: "KnowledgeBundle",
    *,
    capabilities: set[str],
    token_budget: int | None = None,
    byte_budget: int | None = None,
) -> RenderedContext:
    """Render ``bundle`` into a `RenderedContext` for a target advertising ``capabilities``.

    Deterministic over ``bundle.items`` order. Every preserved item gets a `resources` descriptor
    (always, so the canonical payload/hash/artifact_ref is never lost); some also get an inline prompt
    fragment (a one-line index entry, plus a budgeted text/JSON body for text and no-tool-channel
    structured items). ``token_budget``/``byte_budget`` gate only that inline-prompt spend; an item is
    omitted outright (into ``omitted_item_ids``, with no content anywhere) only when even its one-line
    index entry cannot fit and at least one other item has already been preserved.
    """
    ctx = RenderedContext()
    index_lines: list[str] = []
    content_fragments: list[str] = []

    gap_lines = _gap_lines(bundle)
    header = (
        f"Structured knowledge bundle {bundle.id} rev {bundle.revision} "
        f"for task {bundle.task!r} (target={bundle.target_kind}:{bundle.target_id})."
    )
    header_cost_bytes = _byte_len(header) + sum(_byte_len(gl) for gl in gap_lines)
    header_cost_tokens = _estimate_tokens(header) + sum(_estimate_tokens(gl) for gl in gap_lines)
    ctx.bytes_used += header_cost_bytes
    ctx.tokens_used += header_cost_tokens

    for item in bundle.items:
        bucket = _bucket(item.modality)
        rendering = _rendering_hint(bucket, capabilities)
        line = _index_line(item, bucket, rendering)

        inline_body: str | None = None
        if bucket == "text":
            text = (item.text_surface or "").strip()
            if text:
                inline_body = f"[{item.id}] {text}"
        elif bucket == "structured" and rendering == "json_block":
            inline_body = (
                f"[{item.id}] JSON ({item.schema_uri}):\n{json.dumps(item.payload, sort_keys=True, default=str)}"
            )

        candidate_bytes = _byte_len(line) + (_byte_len(inline_body) if inline_body else 0)
        candidate_tokens = _estimate_tokens(line) + (_estimate_tokens(inline_body) if inline_body else 0)
        descriptor = _resource_descriptor(item, bucket, rendering)
        candidate_bytes += _byte_len(descriptor)

        would_exceed_bytes = byte_budget is not None and (ctx.bytes_used + candidate_bytes) > byte_budget
        would_exceed_tokens = token_budget is not None and (ctx.tokens_used + candidate_tokens) > token_budget
        if (would_exceed_bytes or would_exceed_tokens) and ctx.preserved_item_ids:
            ctx.omitted_item_ids.append(item.id)
            continue

        index_lines.append(line)
        if inline_body:
            content_fragments.append(inline_body)
        ctx.resources.append(descriptor)
        ctx.preserved_item_ids.append(item.id)
        ctx.bytes_used += candidate_bytes
        ctx.tokens_used += candidate_tokens

    index_body = header
    if gap_lines:
        index_body += "\n\nOpen discovery gaps:\n" + "\n".join(gap_lines)
    index_body += "\n\nEvidence index:\n" + ("\n".join(index_lines) if index_lines else "(no items)")
    ctx.messages.append({"role": "system", "content": index_body})

    if content_fragments:
        ctx.messages.append({"role": "system", "content": "\n\n".join(content_fragments)})

    return ctx
