"""Path-safety helper shared by every route that keys a filesystem path off a client-supplied name
(substrate shards, tasks, solutions, routes, agentic toolcallers/planners, fine-tune artifacts).

pathlib's ``/`` operator does not resolve ``..`` segments, and joining a base path with a string that
LOOKS absolute (e.g. ``"/etc"``) replaces the base entirely instead of concatenating to it -- so a
client-supplied ``name`` used directly as ``root / name`` can escape ``root`` unless validated first.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def validate_path_segment(name: str) -> str:
    """Reject a name that isn't safe to use as a single path segment under some root: empty, containing
    a path separator, or containing ``..``. Returns ``name`` unchanged so this composes with call sites
    that build something other than a bare ``root / name`` join (e.g. appending a file suffix)."""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=422, detail=f"invalid name {name!r}: must be a single path segment")
    return name


def safe_join(root: Path, name: str) -> Path:
    """``root / name``, validated: rejects a name that isn't a single safe segment (see
    :func:`validate_path_segment`), and -- as a second, independent check -- rejects a result that
    doesn't actually resolve under ``root`` (defense in depth against anything the substring check
    misses)."""
    validate_path_segment(name)
    candidate = (root / name).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise HTTPException(status_code=422, detail=f"invalid name {name!r}: escapes its root")
    return candidate
