"""MCP (Model Context Protocol) integration for the platform.

Two halves:

* ``server`` — exposes the platform's hosted models as MCP **tools** (a ``chat`` tool per model plus a
  ``list_models`` tool) over a :class:`~mixle_mlops.core.registry.ModelRegistry`, so any MCP-speaking client
  (a desktop client, an agent, or another model) can discover and invoke the models.
* ``client`` — a small MCP **client** so hosted models can call *external* MCP tools (connect, list, call) over
  stdio or HTTP.

The implementation prefers the official ``mcp`` python package when importable, otherwise it speaks the
JSON-RPC 2.0 MCP subset directly (``initialize`` / ``tools/list`` / ``tools/call``) so it works dependency-light.
"""
from __future__ import annotations

from .client import MCPClient, MCPClientError, StdioTransport, HTTPTransport
from .server import MCPServer, run_mcp_server, build_model_tools, HAVE_OFFICIAL_MCP

__all__ = [
    "MCPServer",
    "run_mcp_server",
    "build_model_tools",
    "HAVE_OFFICIAL_MCP",
    "MCPClient",
    "MCPClientError",
    "StdioTransport",
    "HTTPTransport",
]
