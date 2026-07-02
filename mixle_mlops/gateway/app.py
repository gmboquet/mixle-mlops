"""The platform gateway: a FastAPI app exposing the OpenAI-compatible API + the platform API, with a model
registry built from config. Runnable end-to-end against the echo model or any OpenAI-compatible LLM backend."""
from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import Settings, get_settings
from ..core.registry import ModelRegistry
from ..image_gen import register_demo_image_model
from ..models import EchoAdapter, OpenAICompatAdapter, make_adapter
from ..models.mixle_model import register_demo_mixle_model
from ..models.task_cascade import register_demo_task_model
from ..storage.db import init_db
from .routes import (
    accounts,
    cache,
    chat,
    cloud,
    conversations,
    datasets,
    evolve,
    feedback,
    files,
    fine_tunes,
    images,
    mcp,
    mixle,
    models,
    rag,
)


def build_registry(settings: Settings) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(EchoAdapter("echo"))                      # always available; zero backends needed
    for model_id in settings.llm_models:                        # configured LLM backend models (shared backend)
        registry.register(OpenAICompatAdapter(model_id, base_url=settings.llm_base_url,
                                              api_key=settings.llm_api_key))
    for model_id, backend in (settings.llm_backends or {}).items():   # per-model backends (local + cloud in one registry)
        try:
            # provider= selects a native Anthropic/Gemini adapter; else OpenAI-compatible. One config, any provider.
            registry.register(make_adapter(
                model_id, backend,
                default_base_url=settings.llm_base_url, default_api_key=settings.llm_api_key))
        except Exception:
            pass
    if settings.enable_demo_models:                             # demo models for /v1/mixle and /v1/images
        for _register in (register_demo_mixle_model, register_demo_image_model, register_demo_task_model):
            try:
                _register(registry)                             # task demo needs torch; tolerated if absent
            except Exception:                                   # never let a demo registration break startup
                pass
    if settings.image_model and settings.image_base_url:        # a real image backend, if configured
        try:
            from ..image_gen import ImageGenAdapter
            registry.register(ImageGenAdapter(settings.image_model, backend="openai",
                                              base_url=settings.image_base_url, api_key=settings.image_api_key))
        except Exception:
            pass
    if settings.local_model or len(settings.local_poe_models) >= 2:   # local logit-level engine (PoE + grammar)
        try:
            from ..models.local_engine import load_local_engine, load_speculative_engine
            if settings.local_model:
                registry.register(load_local_engine("local", [settings.local_model],
                                                    max_new_tokens=settings.local_max_tokens))
            if len(settings.local_poe_models) >= 2:
                registry.register(load_local_engine("local-poe", settings.local_poe_models,
                                                    max_new_tokens=settings.local_max_tokens))
            if settings.local_draft_model and settings.local_model:   # speculative-decoding fast model
                registry.register(load_speculative_engine("local-fast", settings.local_draft_model,
                                                          settings.local_model, k=settings.local_spec_k,
                                                          max_new_tokens=settings.local_max_tokens))
        except Exception:                                            # never let model loading break startup
            pass
    if settings.diffusion_model:                                     # local Stable Diffusion via diffusers
        try:
            from ..image_gen.local_diffusion import load_local_diffusion
            registry.register(load_local_diffusion(
                "sd-local", settings.diffusion_model,
                steps=settings.diffusion_steps,
                guidance_scale=settings.diffusion_guidance,
                width=settings.diffusion_width,
                height=settings.diffusion_height,
            ))
        except Exception:
            pass
    return registry


async def _evolution_loop(app: FastAPI, interval: int) -> None:
    """Autonomous self-improvement: every ``interval`` seconds, run one verify-gated improve pass over all
    hosted mixle models. Best-effort and anti-regression — a pass can only improve or no-op a model."""
    from sqlmodel import Session

    from ..evolve.scheduler import EvolutionScheduler
    from ..storage.db import get_engine

    while True:
        await asyncio.sleep(interval)
        try:
            with Session(get_engine()) as session:
                EvolutionScheduler(app.state.registry).tick(session)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    app.state.settings = settings
    app.state.registry = build_registry(settings)
    evolve_task = None
    if settings.evolve_interval_seconds > 0:                     # opt-in autonomous self-evolution loop
        evolve_task = asyncio.create_task(_evolution_loop(app, settings.evolve_interval_seconds))
    try:
        yield
    finally:
        if evolve_task is not None:
            evolve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await evolve_task


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="mixle-mlops", version="0.1.0",
                  description="All-in-one AI platform: host mixle + open LLMs, OpenAI-compatible.",
                  lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
        allow_headers=["*"], allow_credentials=True,
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "mixle-mlops"}

    app.include_router(accounts.router, tags=["accounts"])
    app.include_router(models.router, prefix="/v1", tags=["models"])
    app.include_router(chat.router, prefix="/v1", tags=["chat"])
    app.include_router(mixle.router, prefix="/v1", tags=["mixle"])        # /v1/mixle/predict|score|decide|latent|capabilities
    app.include_router(feedback.router, prefix="/v1", tags=["feedback"])  # /v1/feedback, /v1/rlhf/*
    app.include_router(files.router, prefix="/v1", tags=["files"])        # /v1/files (multimodal uploads)
    app.include_router(mcp.router, tags=["mcp"])                          # /mcp (JSON-RPC over HTTP)
    app.include_router(rag.router, prefix="/v1", tags=["rag"])            # /v1/documents, /v1/rag/search
    app.include_router(cache.router, prefix="/v1", tags=["cache"])        # /v1/cache/stats
    app.include_router(conversations.router, prefix="/v1", tags=["conversations"])  # /v1/conversations (+ export)
    app.include_router(images.router, prefix="/v1", tags=["images"])      # /v1/images/generations
    app.include_router(datasets.router, prefix="/v1", tags=["datasets"])  # /v1/datasets/generate
    app.include_router(fine_tunes.router, prefix="/v1", tags=["fine_tunes"])  # /v1/fine_tunes (train + serve)
    app.include_router(cloud.router, prefix="/v1", tags=["cloud"])        # /v1/cloud/objectstore
    app.include_router(evolve.router, prefix="/v1", tags=["evolve"])      # /v1/evolve/* (self-evolution)
    return app


app = create_app()
