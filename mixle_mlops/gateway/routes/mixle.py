"""The mixle distribution/decision routes -- the platform's differentiator over an LLM proxy.

``POST /predict /score /latent /decide`` and ``GET /capabilities/{model_id}`` each pull the model from
the registry, require an authenticated user, and map a model's :class:`CapabilityError` to HTTP 422 so a
client learns precisely which capability a given model lacks.

The integrator includes this router on the app (see this module's wiring note in the build report).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...accounts.models import User
from ...core.adapters import CapabilityError
from ..auth import require_user

router = APIRouter(prefix="/mixle", tags=["mixle"])


class PredictBody(BaseModel):
    model: str
    records: list[Any] = Field(default_factory=list)
    quantiles: list[float] | None = None
    interval_level: float | None = None
    n_ensemble: int | None = None
    seed: int | None = None


class ScoreBody(BaseModel):
    model: str
    records: list[Any] = Field(default_factory=list)


class LatentBody(BaseModel):
    model: str
    records: list[Any] = Field(default_factory=list)


class DecideBody(BaseModel):
    """A decision request. ``loss`` names a built-in loss; ``actions`` is the candidate set.

    Built-in losses operate on a scalar outcome/draw ``y`` and a scalar action ``a``:

    * ``squared`` evaluates ``(a - y)**2`` for point estimation under squared error.
    * ``absolute`` evaluates ``abs(a - y)`` for median-optimal actions.
    * ``linex`` evaluates ``exp(c*(a-y)) - c*(a-y) - 1`` using ``loss_params['c']``.
    * ``newsvendor`` evaluates underage/overage costs from ``loss_params['cu']`` and
      ``loss_params['co']``.

    Custom Python callables are intentionally NOT accepted over HTTP; use the in-process adapter for that.
    """

    model: str
    records: list[Any] = Field(default_factory=list)
    actions: list[float]
    loss: str = "squared"
    loss_params: dict[str, float] = Field(default_factory=dict)
    over: str = "predictive"
    n: int = 2000
    seed: int = 0
    cvar_alpha: float = 0.1


def _resolve(request: Request, model_id: str):
    registry = request.app.state.registry
    if not registry.has(model_id):
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
    return registry.get(model_id)


def _build_loss(name: str, params: dict[str, float]):
    """Map a named loss to a numpy-vectorized ``loss(action, draws) -> array``."""
    import numpy as np

    if name == "squared":
        return lambda a, y: (float(a) - np.asarray(y, dtype=float)) ** 2
    if name == "absolute":
        return lambda a, y: np.abs(float(a) - np.asarray(y, dtype=float))
    if name == "linex":
        c = float(params.get("c", 1.0))
        return lambda a, y: np.exp(c * (float(a) - np.asarray(y, dtype=float))) - c * (
            float(a) - np.asarray(y, dtype=float)
        ) - 1.0
    if name == "newsvendor":
        cu = float(params.get("cu", 1.0))   # underage cost (a < y)
        co = float(params.get("co", 1.0))   # overage cost  (a > y)
        def _nv(a, y):
            y = np.asarray(y, dtype=float)
            short = np.maximum(y - float(a), 0.0)
            over = np.maximum(float(a) - y, 0.0)
            return cu * short + co * over
        return _nv
    raise HTTPException(status_code=422, detail=f"unknown loss {name!r}")


@router.post("/predict")
async def predict(body: PredictBody, request: Request, user: User = Depends(require_user)):
    adapter = _resolve(request, body.model)
    opts = body.model_dump(exclude={"model", "records"}, exclude_none=True)
    try:
        return await adapter.predict(body.records, **opts)
    except CapabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/score")
async def score(body: ScoreBody, request: Request, user: User = Depends(require_user)):
    adapter = _resolve(request, body.model)
    try:
        return await adapter.score(body.records)
    except CapabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/latent")
async def latent(body: LatentBody, request: Request, user: User = Depends(require_user)):
    adapter = _resolve(request, body.model)
    try:
        return await adapter.latent(body.records)
    except CapabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/decide")
async def decide(body: DecideBody, request: Request, user: User = Depends(require_user)):
    adapter = _resolve(request, body.model)
    loss = _build_loss(body.loss, body.loss_params)
    try:
        return await adapter.decide(
            body.records,
            loss=loss,
            actions=body.actions,
            over=body.over,
            n=body.n,
            seed=body.seed,
            cvar_alpha=body.cvar_alpha,
        )
    except CapabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/capabilities/{model_id}")
async def capabilities(model_id: str, request: Request, user: User = Depends(require_user)):
    adapter = _resolve(request, model_id)
    return {
        "model": model_id,
        "kind": getattr(adapter, "kind", "llm"),
        "capabilities": sorted(adapter.capabilities()),
    }
