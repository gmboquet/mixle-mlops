"""``POST /v1/interpret`` -- a calibrated natural-language claim about a physics posterior, or an honest
abstain, wiring ``mixle.reason.language_bridge.PosteriorDescriber`` (imported by nothing until this
route, via ``mixle_pde.reasoning.describe_posterior``) onto an IC-2 content-hashed posterior artifact.

  * ``POST /v1/interpret`` -- ``{posterior_ref, field, tol, level}`` -> ``{claim, abstained,
    provenance}``. ``level`` is the caller-facing credible-interval coverage (the same convention as
    IC-1's ``Posterior.credible_interval(level)``); it is converted to ``PosteriorDescriber``'s native
    ``alpha = 1 - level`` before calling :func:`mixle_pde.reasoning.describe_posterior`.

Posterior resolution is a swappable module-level hook (:func:`resolve_posterior`) rather than a hard
import-time dependency on ``mixle_pde.io.artifacts.load_posterior`` (IC-2): that module is owned by a
sibling task and is lazily imported inside the default hook, so this route is wired for the real
artifact loader the moment it lands, without this gateway failing to import (or a demo/dev deployment
without ``mixle_pde``'s IO extra failing to boot) in the meantime.

Authenticated like the platform's other ``/v1`` routes (``Depends(require_user)``). E7 fills the full
hashed provenance/lineage edge into the response's ``provenance`` field; here it is a stub carrying the
request parameters that produced the claim.

Wiring (integrator): ``app.include_router(interpret.router, prefix="/v1", tags=["interpret"])`` in
``gateway/app.py``.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from ...accounts.models import User
from ..auth import require_user

router = APIRouter()


class InterpretRequest(BaseModel):
    posterior_ref: str
    field: str
    tol: float
    level: float = 0.9


def _default_resolve_posterior(posterior_ref: str) -> Any:
    """The real IC-2 path: ``mixle_pde.io.artifacts.load_posterior`` (lazy import -- E2 owns that
    module; this route must not hard-fail at import time on a sibling package/extra it doesn't own)."""
    try:
        from mixle_pde.io.artifacts import load_posterior
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="posterior artifact storage unavailable: install/land mixle_pde's io.artifacts (IC-2)",
        ) from exc
    return load_posterior(posterior_ref)


# Swappable so callers/tests can inject an in-memory posterior resolver ahead of IC-2 landing; the
# default always tries the real artifact loader first.
resolve_posterior: Callable[[str], Any] = _default_resolve_posterior


@router.post("/interpret")
async def interpret(body: InterpretRequest = Body(...), user: User = Depends(require_user)) -> dict:
    """Resolve ``posterior_ref`` (IC-2), describe ``field`` at absolute precision ``tol``, and return a
    calibrated claim or an honest abstain -- never a driller-facing number without the honesty check."""
    if not 0.0 < body.level < 1.0:
        raise HTTPException(status_code=400, detail="level must be in (0, 1)")

    try:
        posterior = resolve_posterior(body.posterior_ref)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"could not resolve posterior_ref: {exc}") from exc

    from mixle_pde.reasoning import describe_posterior

    alpha = 1.0 - body.level
    claim = describe_posterior(posterior, body.field, tol=body.tol, alpha=alpha)
    return {
        "object": "interpret.claim",
        "claim": "" if claim is None else claim.text(),
        "abstained": claim is None,
        "provenance": {
            "posterior_ref": body.posterior_ref,
            "field": body.field,
            "tol": body.tol,
            "level": body.level,
        },  # E7 fills the full hashed lineage edge (data -> inversion -> interpretation -> decision)
    }
