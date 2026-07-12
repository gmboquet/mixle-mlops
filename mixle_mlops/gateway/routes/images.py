"""OpenAI-compatible image generation: ``POST /v1/images/generations``.

Accepts ``{model, prompt, n, size}`` (the OpenAI Images API shape), resolves the model from the registry — an
:class:`~mixle_mlops.image_gen.adapter.ImageGenAdapter` — generates the images (stored in the platform blob
store), and returns ``{created, data:[{url, b64_json?}]}``. Behind ``Depends(require_user)``.

Wiring (integrator): ``app.include_router(images.router, prefix="/v1", tags=["images"])`` in ``gateway/app.py``;
optionally register a local demo image model at startup via ``register_demo_image_model(registry)``."""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...accounts.models import User
from ..auth import require_user

router = APIRouter()


class ImageGenerationRequest(BaseModel):
    model: str = ""                       # empty → first image-capable model in the registry
    prompt: str
    n: int = 1
    size: str | None = None
    response_format: str = "url"          # "url" (default) or "b64_json"
    user: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def _resolve_image_model(registry, requested: str):
    """Pick the adapter to serve this request: the requested model if given, else the first image-capable one."""
    if requested:
        if not registry.has(requested):
            raise HTTPException(status_code=404, detail=f"model {requested!r} not found")
        adapter = registry.get(requested)
        if "image_generation" not in adapter.capabilities():
            raise HTTPException(status_code=422,
                                detail=f"model {requested!r} does not support image_generation")
        return adapter
    for name in registry.names():
        adapter = registry.get(name)
        if "image_generation" in adapter.capabilities():
            return adapter
    raise HTTPException(status_code=404, detail="no image-generation model is registered")


@router.post("/images/generations")
async def create_image(req: ImageGenerationRequest, request: Request,
                       user: User = Depends(require_user)):
    registry = request.app.state.registry
    adapter = _resolve_image_model(registry, req.model)
    try:
        images = await adapter.generate(req.prompt, n=req.n, size=req.size, **req.extra)
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # backend failure → 502
        raise HTTPException(status_code=502, detail=f"image backend error: {exc}")

    data = []
    for img in images:
        entry: dict[str, Any] = {"url": img["url"]}
        if req.response_format == "b64_json":
            entry = {"b64_json": img["b64_json"]}
        data.append(entry)
    return {"created": int(time.time()), "data": data}
