"""M1c -- target-aware model-context handoff + delta application.

Public surface: `render_bundle`/`RenderedContext` (capability-aware bundle rendering, `render.py`) and
`handoff`/`HandoffResult` (render + require-a-structured-delta + M2a-apply + receipt, `handoff.py`). See
each module's docstring for the algorithm; `mixle_mlops.gateway.routes.knowledge` exposes both over HTTP.
"""

from __future__ import annotations

from .handoff import DELTA_TOOL_NAME, HandoffError, HandoffResult, handoff
from .render import RenderedContext, render_bundle

__all__ = [
    "RenderedContext",
    "render_bundle",
    "HandoffResult",
    "HandoffError",
    "DELTA_TOOL_NAME",
    "handoff",
]
