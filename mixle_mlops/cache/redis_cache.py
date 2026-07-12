"""Redis-backed ``Cache``: shared across gateway replicas so the response cache and rate limiter are
consistent platform-wide. ``redis`` is lazy-imported (optional ``scale`` extra) — importing this module
costs nothing until a ``RedisCache`` is constructed.

Values are JSON-encoded on the wire. ``incr`` maps to Redis' atomic ``INCRBY`` (with ``EXPIRE`` only on the
first write of the window), which is what makes the fixed-window rate limiter correct across replicas."""
from __future__ import annotations

import json
from typing import Any

from .base import Cache


class RedisCache(Cache):
    """A ``Cache`` over a Redis server.

    Construct with a ``redis://`` URL, or pass a preconfigured client in tests.
    Counters use native atomic operations so multiple replicas agree.
    """

    def __init__(self, url: str | None = None, *, client: Any = None) -> None:
        if client is not None:
            self._r = client
        else:
            try:
                import redis  # lazy: optional 'scale' extra
            except ImportError as exc:  # pragma: no cover - exercised only without redis installed
                raise RuntimeError(
                    "RedisCache requires the 'redis' package (install the 'scale' extra)."
                ) from exc
            self._r = redis.Redis.from_url(url, decode_responses=True)

    @staticmethod
    def _ttl_ms(ttl: float | None) -> int | None:
        return int(ttl * 1000) if ttl is not None else None

    def get(self, key: str) -> Any | None:
        raw = self._r.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        raw = json.dumps(value, default=str)
        px = self._ttl_ms(ttl)
        if px is not None:
            self._r.set(key, raw, px=px)
        else:
            self._r.set(key, raw)

    def incr(self, key: str, amount: int = 1, *, ttl: float | None = None) -> int:
        new = int(self._r.incrby(key, amount))
        if ttl is not None and new == amount:  # first write of this window → arm the expiry
            self._r.pexpire(key, self._ttl_ms(ttl))
        return new

    def expire(self, key: str, ttl: float) -> bool:
        return bool(self._r.pexpire(key, self._ttl_ms(ttl)))

    def delete(self, key: str) -> bool:
        return bool(self._r.delete(key))

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": "redis"}
        try:
            info = self._r.info()
            out["used_memory"] = info.get("used_memory")
            out["connected_clients"] = info.get("connected_clients")
            out["keys"] = self._r.dbsize()
        except Exception:  # pragma: no cover - server may not expose INFO
            pass
        return out
