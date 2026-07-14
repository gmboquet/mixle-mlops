"""OpenAI-compatible chat completions — the product's main inference route. Composes the platform pipeline:
rate-limit → multimodal-normalize → RAG-augment → response-cache → dispatch → persist (conversation memory).
Each stage is gated/opt-in/defensive so the default path stays simple and the extras never break a chat."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ...accounts.models import User
from ...config import get_settings
from ...core.adapters import (
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatRequest,
    ChoiceDelta,
)
from ...multimodal.content import MultimodalError, normalize_messages
from ..auth import current_user
from ..decision_surfacing import surface_decisions

router = APIRouter()


def _principal(user: User | None, request: Request) -> str:
    if user is not None:
        return user.id
    return request.client.host if request.client else "anon"


def _persist(user: User | None, req: ChatRequest, name: str, assistant_text: str) -> str | None:
    """Record the turn into the user's conversation history; return the conversation id (for threading). Best-effort."""
    if user is None:
        return None
    try:
        from sqlmodel import Session

        from ...conversations.service import persist_turn
        from ...storage.db import get_engine

        user_text = req.messages[-1].text() if req.messages else ""
        with Session(get_engine()) as session:
            conv = persist_turn(session, user.id, req.extra.get("conversation_id"), user_text, assistant_text, model=name)
            return conv.id
    except Exception:
        return None


def _record_signal(model_id: str, *, kind: str, confidence=None, escalated=None) -> None:
    """Persist a usage signal (cascade decision / best-of-N confidence) for the self-evolution loop. Best-effort."""
    try:
        from sqlmodel import Session

        from ...evolve.signals import record_signal
        from ...storage.db import get_engine

        with Session(get_engine()) as session:
            record_signal(session, model_id, kind=kind, confidence=confidence, escalated=escalated)
    except Exception:
        pass


def _persist_trace(tool_reg, question: str, name: str, assistant_text: str, *, stopped_reason: str) -> None:
    """Bind this agentic turn's dispatched tool calls (`tool_reg.trace_steps`, stamped by `ToolRegistry`)
    into a `Receipt` and land it in the durable M4 trace foundry feed (M4a). Best-effort, same as
    `_persist`/`_record_signal`: capturing a trace must never fail or slow down the chat response."""
    try:
        from mixle.substrate import Substrate

        from ..trace_capture import capture_steps

        root = Path(get_settings().registry_root) / "substrate" / "agent_traces"
        sink = Substrate(str(root))
        capture_steps(question, tool_reg.trace_steps, assistant_text, model_id=name, sink=sink,
                     provenance={"stopped_reason": stopped_reason})
        sink.save()
    except Exception:
        pass


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, response: Response,
                           user: User | None = Depends(current_user)):
    settings = get_settings()
    registry = request.app.state.registry
    name = req.model or settings.default_model
    if not registry.has(name):
        raise HTTPException(status_code=404, detail=f"model {name!r} not found")
    adapter = registry.get(name)

    # 1. rate limit (opt-in via MIXLE_RATE_LIMIT_PER_MIN), shared across replicas when Redis is configured
    if settings.rate_limit_per_min > 0:
        try:
            from ...cache import RateLimiter, get_cache

            res = RateLimiter(get_cache(), limit=settings.rate_limit_per_min, window=60).check(_principal(user, request))
            if not res.allowed:
                raise HTTPException(status_code=429, detail="rate limit exceeded", headers=res.headers())
        except HTTPException:
            raise
        except Exception:
            pass

    # 2. resolve uploaded-file refs → image_url parts for vision LLMs
    try:
        req.messages = normalize_messages(req.messages)
    except MultimodalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 3. RAG: augment with retrieved context from this user's past conversations + documents (opt-in via extra.rag)
    if user is not None and req.extra.get("rag"):
        try:
            from ...rag.augment import build_rag_messages

            req.messages = build_rag_messages(user.id, req.messages)
        except Exception:
            pass

    # 4. server-side agentic loop (opt-in via extra.agent): the gateway executes the platform's tools
    #    (MCP / RAG / mixle decide+predict) across turns. Client-declared `tools` still pass through to the
    #    backend untouched when agent mode is off (standard OpenAI client-side tool use).
    if req.extra.get("agent"):
        from ..agent_loop import run_agent_loop
        from ..tool_registry import ToolRegistry

        question = req.messages[-1].text() if req.messages else ""
        tool_reg = ToolRegistry(registry, user_id=user.id if user is not None else None,
                                names=req.extra.get("tools"), model_id=name)
        max_iters = req.max_tool_iters or 6
        if req.stream:
            async def agent_stream():
                completion = await run_agent_loop(adapter, req, tool_reg, max_iters=max_iters)
                if completion.choices:
                    # F4: surface any decision-quantity tool result this turn dispatched, gating the
                    # streamed text through it (prior-dominated -> calibrated abstain, never a bare number).
                    surface_decisions(completion.choices[0], tool_reg.trace_steps)
                text = completion.choices[0].message.text() if completion.choices else ""
                chunk = ChatCompletionChunk(model=name, choices=[ChatChunkChoice(
                    index=0, delta=ChoiceDelta(role="assistant", content=text), finish_reason="stop")])
                yield f"data: {chunk.model_dump_json()}\n\n"
                _persist_trace(tool_reg, question, name, text, stopped_reason="agent_stream")
                cid = _persist(user, req, name, text)
                if cid:
                    yield f"data: {json.dumps({'conversation_id': cid})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(agent_stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        completion = await run_agent_loop(adapter, req, tool_reg, max_iters=max_iters)
        if completion.choices:
            # F4: surface any decision-quantity tool result this turn dispatched (query_posterior /
            # IC-8 region_mass / prob_exceed / ...) into `choice.decisions`, gating the served text
            # through the honesty flag -- a prior-dominated decision never reaches the driller as a
            # bare point estimate.
            surface_decisions(completion.choices[0], tool_reg.trace_steps)
        answer_text = completion.choices[0].message.text() if completion.choices else ""
        _persist_trace(tool_reg, question, name, answer_text, stopped_reason="agent")
        cid = _persist(user, req, name, answer_text)
        if cid:
            response.headers["X-Conversation-Id"] = cid
        return completion

    # 4a. cascade router (opt-in via extra.cascade={frontier, threshold, n}): answer locally when the local
    #     model is self-consistent enough, else escalate to a frontier model — the FrugalGPT quality/cost dial.
    casc = req.extra.get("cascade")
    if isinstance(casc, dict) and not req.stream:
        frontier_id = casc.get("frontier")
        if frontier_id and registry.has(frontier_id):
            from ..cascade import cascade

            completion, info = await cascade(adapter, registry.get(frontier_id), req,
                                             threshold=float(casc.get("threshold", 0.6)),
                                             n=int(casc.get("n", 5)))
            response.headers["X-Cascade-Escalated"] = "1" if info["escalated"] else "0"
            if info.get("local_confidence") is not None:
                response.headers["X-Self-Consistency"] = f"{info['local_confidence']:.3f}"
            _record_signal(name, kind="cascade", confidence=info.get("local_confidence"),
                           escalated=info["escalated"])
            cid = _persist(user, req, name, completion.choices[0].message.text() if completion.choices else "")
            if cid:
                response.headers["X-Conversation-Id"] = cid
            return completion

    # 4b. best-of-N self-consistency (opt-in via extra.best_of_n): sample N, return the majority answer +
    #     a calibrated confidence (the X-Self-Consistency header), the test-time-compute quality lever.
    bon = req.extra.get("best_of_n")
    if bon and not req.stream:
        try:
            n = int(bon)
        except (TypeError, ValueError):
            n = 0
        if n > 1:
            temp = req.temperature if req.temperature is not None else 0.8
            # with a real verifier (exact-match / computed-reference / LLM-judge) selection beats majority voting
            verifier_spec = req.extra.get("verifier")
            if isinstance(verifier_spec, dict):
                from ..verifiers import best_of_n_verified, build_verifier

                verifier = build_verifier(verifier_spec, registry)
                if verifier is not None:
                    completion, info = await best_of_n_verified(adapter, req, n=n, verifier=verifier, temperature=temp)
                    response.headers["X-Verifier-Score"] = f"{info.get('best_score', 0):.3f}"
                    cid = _persist(user, req, name, completion.choices[0].message.text() if completion.choices else "")
                    if cid:
                        response.headers["X-Conversation-Id"] = cid
                    return completion

            from ..bestofn import best_of_n

            completion, info = await best_of_n(adapter, req, n=n,
                                               selector=str(req.extra.get("select", "self_consistency")),
                                               temperature=temp)
            conf = info.get("confidence")
            if conf is not None:
                response.headers["X-Self-Consistency"] = f"{conf:.3f}"
                _record_signal(name, kind="best_of_n", confidence=conf)
            cid = _persist(user, req, name, completion.choices[0].message.text() if completion.choices else "")
            if cid:
                response.headers["X-Conversation-Id"] = cid
            return completion

    # 4c. Mixture-of-Agents (opt-in via extra.moa={proposers:[...], aggregator, layers}): several models propose,
    #     an aggregator synthesizes — the ensemble lever for a heterogeneous local fleet.
    moa = req.extra.get("moa")
    if isinstance(moa, dict) and not req.stream:
        proposer_ids = [m for m in (moa.get("proposers") or []) if registry.has(m)]
        aggregator_id = moa.get("aggregator") or name
        if proposer_ids and registry.has(aggregator_id):
            from ..moa import mixture_of_agents

            completion, info = await mixture_of_agents(
                [registry.get(m) for m in proposer_ids], registry.get(aggregator_id), req,
                layers=int(moa.get("layers", 1)), select_k=moa.get("select_k"))
            cid = _persist(user, req, name, completion.choices[0].message.text() if completion.choices else "")
            if cid:
                response.headers["X-Conversation-Id"] = cid
            return completion

    # 4d. constrained / structured decoding (opt-in via extra.constrained = the schema/grammar spec): pass the
    #     constraint to the backend's guided decoder + validate + repair-retry until the output conforms.
    cspec = req.extra.get("constrained")
    if isinstance(cspec, dict) and not req.stream:
        # TRUE in-decode grammar masking for local-engine models with logit access (regex/choices -> TokenFSA);
        # proxied models fall back to the backend's guided decoder + validate/repair.
        from ...models.local_engine import LocalEngineAdapter

        if isinstance(adapter, LocalEngineAdapter) and (cspec.get("regex") or cspec.get("choices")
                                                        or cspec.get("json_schema")):
            try:
                from ...engines.regex_fsa import (
                    build_choice_fsa,
                    json_schema_to_token_fsa,
                    regex_to_token_fsa,
                )

                vocab = adapter.vocab()
                eos = getattr(adapter._primary, "eos", None)
                if cspec.get("choices"):
                    fsa = build_choice_fsa(list(cspec["choices"]), vocab, eos_id=eos)
                elif cspec.get("json_schema"):
                    fsa = json_schema_to_token_fsa(cspec["json_schema"], vocab, eos_id=eos)
                else:
                    fsa = regex_to_token_fsa(cspec["regex"], vocab, eos_id=eos)
                completion = await adapter.chat(req.model_copy(update={"extra": {**req.extra, "_grammar": fsa}}))
                response.headers["X-Constrained-Valid"] = "1"      # masking makes the output well-formed by construction
                cid = _persist(user, req, name, completion.choices[0].message.text() if completion.choices else "")
                if cid:
                    response.headers["X-Conversation-Id"] = cid
                return completion
            except Exception:
                pass                                               # fall through to the backend path on any failure

        from ..constrained import constrained_complete

        completion, info = await constrained_complete(adapter, req, cspec)
        response.headers["X-Constrained-Valid"] = "1" if info["valid"] else "0"
        if info.get("repairs"):
            response.headers["X-Constrained-Repairs"] = str(info["repairs"])
        cid = _persist(user, req, name, completion.choices[0].message.text() if completion.choices else "")
        if cid:
            response.headers["X-Conversation-Id"] = cid
        return completion

    # 5. response cache (opt-in via MIXLE_ENABLE_RESPONSE_CACHE), exact-match, non-streaming, non-tool only
    rc = None
    if settings.enable_response_cache and not req.stream and not req.tools:
        try:
            from ...cache import ResponseCache, get_cache

            rc = ResponseCache(get_cache())
            hit = rc.get(req)
            if hit is not None:
                return ChatCompletion.model_validate(hit)
        except Exception:
            rc = None

    if req.stream:
        async def event_stream():
            buf: list[str] = []
            try:
                async for chunk in adapter.stream(req):
                    for ch in chunk.choices:
                        if ch.delta.content:
                            buf.append(ch.delta.content)
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
            cid = _persist(user, req, name, "".join(buf))
            if cid:
                yield f"data: {json.dumps({'conversation_id': cid})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    completion = await adapter.chat(req)
    if rc is not None:
        try:
            rc.set(req, completion.model_dump())
        except Exception:
            pass
    cid = _persist(user, req, name, completion.choices[0].message.text() if completion.choices else "")
    if cid:
        response.headers["X-Conversation-Id"] = cid
    return completion
