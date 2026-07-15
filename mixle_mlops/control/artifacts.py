"""Owner-scoped content-addressed local artifact storage."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .contracts import ArtifactRef, OperationalError, OwnerScope


class LocalArtifactStore:
    def __init__(self, root: str | Path, *, max_bytes: int = 256 * 1024 * 1024):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    @staticmethod
    def _safe(value: str) -> str:
        if not value or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value
        ):
            raise OperationalError("owner identifiers contain unsafe path characters")
        return value

    def _path(self, owner: OwnerScope, digest: str) -> Path:
        return self.root / self._safe(owner.organization_id) / self._safe(owner.project_id) / digest[:2] / digest

    def put(self, owner: OwnerScope, data: bytes, *, media_type: str, semantic_type: str) -> ArtifactRef:
        if len(data) > self.max_bytes:
            raise OperationalError("artifact exceeds configured size bound")
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(owner, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        return ArtifactRef(owner, digest, len(data), media_type, f"cas://{owner.key}/{digest}", semantic_type)

    def get(self, owner: OwnerScope, artifact: ArtifactRef) -> bytes:
        if artifact.owner != owner:
            raise PermissionError("artifact owner scope does not match caller")
        path = self._path(owner, artifact.sha256)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(artifact.uri)
        data = path.read_bytes()
        if len(data) != artifact.size_bytes or hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise OperationalError("artifact failed size or digest verification")
        return data


__all__ = ["LocalArtifactStore"]
