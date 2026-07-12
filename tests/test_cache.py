"""Build caching + concurrency primitives.

Covers: memory cache get/set/ttl + atomic incr; exact response-cache hit/miss; semantic cache with a deterministic
embedder (NumPy vectors) hitting a near-duplicate; the rate limiter blocking past the budget (both fixed
window and token bucket); and the ``/v1/cache/stats`` route end-to-end.

Self-contained: builds the app via ``create_app()``, includes the cache router, signs up for a key. The
Redis path is exercised only if ``redis`` and an in-memory Redis test client are importable; it is skipped
otherwise to stay dependency-light.
"""
from __future__ import annotations


import numpy as np
import pytest
from fastapi.testclient import TestClient

import mixle_mlops.cache as cache_pkg
import mixle_mlops.storage.db as db
from mixle_mlops.cache import MemoryCache, RateLimiter, ResponseCache, SemanticCache, get_cache
from mixle_mlops.cache.base import cache_key, chat_request_key
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import cache as cache_routes


# ----------------------------- memory cache -----------------------------
def test_memory_get_set_missing():
    c = MemoryCache()
    assert c.get("nope") is None
    c.set("a", {"x": 1})
    assert c.get("a") == {"x": 1}
    assert c.delete("a") is True
    assert c.get("a") is None


def test_memory_ttl_expiry(monkeypatch):
    c = MemoryCache()
    t = {"now": 1000.0}
    monkeypatch.setattr("mixle_mlops.cache.memory.time.monotonic", lambda: t["now"])
    c.set("k", "v", ttl=5.0)
    assert c.get("k") == "v"
    t["now"] += 6.0
    assert c.get("k") is None          # expired


def test_memory_incr_atomic_and_window():
    c = MemoryCache()
    assert c.incr("ctr") == 1
    assert c.incr("ctr", 2) == 3
    assert c.expire("ctr", 10.0) is True
    assert c.expire("absent", 10.0) is False


# ----------------------------- key helpers ------------------------------
def test_chat_request_key_stable_and_param_sensitive():
    msgs = [{"role": "user", "content": "hello"}]
    k1 = chat_request_key("m", msgs, temperature=0.0)
    k2 = chat_request_key("m", msgs, temperature=0.0)
    k3 = chat_request_key("m", msgs, temperature=0.7)
    assert k1 == k2                    # deterministic
    assert k1 != k3                    # sampling params matter
    # dict-order independence
    assert cache_key({"a": 1, "b": 2}) == cache_key({"b": 2, "a": 1})


# --------------------------- exact response cache -----------------------
def test_exact_response_cache_hit_miss():
    rc = ResponseCache(MemoryCache(), ttl=None)
    req = {"model": "echo", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    assert rc.get(req) is None         # miss
    rc.set(req, {"answer": "hello there"})
    assert rc.get(req) == {"answer": "hello there"}   # hit
    other = {"model": "echo", "messages": [{"role": "user", "content": "different"}]}
    assert rc.get(other) is None       # different request → miss


# ----------------------------- semantic cache ---------------------------
def _stub_embedder():
    """Deterministic toy embedder: near-duplicate phrasings map to nearby vectors."""
    table = {
        "what is the capital of france": np.array([1.0, 0.0, 0.0]),
        "tell me france's capital city": np.array([0.98, 0.02, 0.0]),   # near-duplicate of the above
        "how do i bake bread": np.array([0.0, 1.0, 0.0]),               # unrelated
    }

    def embed(text: str):
        return table.get(text.strip().lower(), np.array([0.0, 0.0, 1.0]))

    return embed


def test_semantic_cache_hits_near_duplicate():
    sc = SemanticCache(MemoryCache(), embedder=_stub_embedder(), threshold=0.9, ttl=None)
    stored_req = {"model": "echo", "messages": [{"role": "user", "content": "What is the capital of France"}]}
    assert sc.store(stored_req, {"answer": "Paris"}) is True

    near = {"model": "echo", "messages": [{"role": "user", "content": "Tell me France's capital city"}]}
    resp, sim = sc.lookup(near)
    assert resp == {"answer": "Paris"}
    assert sim >= 0.9

    far = {"model": "echo", "messages": [{"role": "user", "content": "How do I bake bread"}]}
    resp2, sim2 = sc.lookup(far)
    assert resp2 is None
    assert sim2 < 0.9


def test_semantic_cache_no_embedder_is_miss():
    sc = SemanticCache(MemoryCache(), embedder=None)
    req = {"model": "echo", "messages": [{"role": "user", "content": "anything"}]}
    assert sc.store(req, {"answer": "x"}) is False
    assert sc.lookup(req) == (None, -1.0)


# ------------------------------ rate limiter ----------------------------
def test_rate_limiter_fixed_window_blocks_past_budget():
    rl = RateLimiter(MemoryCache(), limit=3, window=60.0)
    results = [rl.check("user-1") for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert results[-1].retry_after > 0
    # separate identity has its own budget
    assert rl.check("user-2").allowed is True


def test_rate_limiter_token_bucket_refills(monkeypatch):
    cache = MemoryCache()
    rl = RateLimiter(cache, limit=60, window=60.0, burst=2)   # capacity 2, refill 1 tok/sec
    t = {"now": 1000.0}
    monkeypatch.setattr("mixle_mlops.cache.ratelimit.time.time", lambda: t["now"])
    assert rl.check("u").allowed is True
    assert rl.check("u").allowed is True
    blocked = rl.check("u")            # bucket drained
    assert blocked.allowed is False
    assert blocked.retry_after > 0
    t["now"] += 1.5                    # ~1.5 tokens refilled
    assert rl.check("u").allowed is True


# --------------------------- get_cache selection ------------------------
def test_get_cache_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("MIXLE_REDIS_URL", raising=False)
    cache_pkg.reset_cache()
    c = get_cache()
    assert isinstance(c, MemoryCache)
    cache_pkg.reset_cache()


# ------------------------------ route test ------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MIXLE_REDIS_URL", raising=False)
    get_settings.cache_clear()
    db._engine = None
    cache_pkg.reset_cache()
    app = create_app()
    app.include_router(cache_routes.router, prefix="/v1", tags=["cache"])
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None
    cache_pkg.reset_cache()


def _signup(client, email="cache@t.com"):
    raw = client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def test_cache_stats_route_requires_auth(client):
    assert client.get("/v1/cache/stats").status_code == 401
    headers = _signup(client)
    r = client.get("/v1/cache/stats", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "cache.stats"
    assert body["stats"]["backend"] == "memory"


# ------------------------------ redis path ------------------------------
@pytest.mark.skipif(
    pytest.importorskip("fakeredis", reason="redis path needs fakeredis to test without a server") is None,
    reason="no fakeredis",
)
def test_redis_cache_roundtrip_with_fakeredis():
    fakeredis = pytest.importorskip("fakeredis")
    from mixle_mlops.cache.redis_cache import RedisCache

    client = fakeredis.FakeStrictRedis(decode_responses=True)
    rc = RedisCache(client=client)
    rc.set("k", {"v": 1}, ttl=None)
    assert rc.get("k") == {"v": 1}
    assert rc.incr("ctr") == 1
    assert rc.incr("ctr", 2) == 3
    assert rc.delete("k") is True
    assert rc.get("k") is None
    # rate limiter works across the redis-backed cache too
    rl = RateLimiter(rc, limit=2, window=60.0)
    assert [rl.check("x").allowed for _ in range(3)] == [True, True, False]
    assert rc.stats()["backend"] == "redis"
