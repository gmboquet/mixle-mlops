"""A searchable tool registry for the agent -- the piece that expands the *scope* of problems the
agent can address, and keeps it robust while doing so.

The agentic loop (:func:`mixle_mlops.agent.tool_agent.run_tool_agent`) is only as broad as the tools
it's handed. Handing it every registered capability at once bloats the prompt and degrades tool
selection (a well-known failure of large flat tool lists). This registry instead embeds each tool's
description once, and :meth:`AgentToolRegistry.retrieve` returns the handful most relevant to a given
goal -- so a registry of dozens (eventually hundreds) of real functions stays usable: the agent sees
a focused, on-topic subset per problem. Broader scope AND a tighter, more reliable working set.

Uses the existing local text embedder (:class:`mixle_mlops.rag.embeddings.Embedder` in its
deterministic hashing-fallback mode), so retrieval needs no external service and is reproducible.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mixle_mlops.agent.tool_agent import introspect_tool
from mixle_mlops.rag.embeddings import Embedder

__all__ = ["AgentToolRegistry"]


class AgentToolRegistry:
    """Register real functions as agent tools and retrieve the ones relevant to a goal.

    ``tools`` is ``{name: function}``. Each tool's searchable text is its :func:`introspect_tool`
    description (the function's docstring first paragraph) plus its name -- so a well-documented
    function is well-retrievable with no extra annotation work.
    """

    def __init__(self, tools: dict[str, Callable[..., Any]], *, embedder: Embedder | None = None) -> None:
        if not tools:
            raise ValueError("AgentToolRegistry needs at least one tool")
        self.tools = dict(tools)
        self._embedder = embedder or Embedder(allow_remote=False)
        self._text = {name: f"{name}: {introspect_tool(fn, name=name).function.description}" for name, fn in tools.items()}
        names = list(self._text)
        vecs = self._embedder.embed([self._text[n] for n in names])
        self._vecs = {n: np.asarray(v, dtype=float) for n, v in zip(names, vecs)}

    def scores(self, goal: str) -> list[tuple[str, float]]:
        """Every tool with its cosine similarity to ``goal``, most relevant first (for inspection/debugging)."""
        q = np.asarray(self._embedder.embed([goal])[0], dtype=float)
        scored = [(name, float(np.dot(q, v))) for name, v in self._vecs.items()]
        return sorted(scored, key=lambda kv: kv[1], reverse=True)

    def retrieve(self, goal: str, *, k: int = 5) -> dict[str, Callable[..., Any]]:
        """Return the ``k`` tools most relevant to ``goal`` as a ``{name: function}`` dict ready to pass
        straight to :func:`~mixle_mlops.agent.tool_agent.run_tool_agent`."""
        top = [name for name, _ in self.scores(goal)[: max(1, int(k))]]
        return {name: self.tools[name] for name in top}
