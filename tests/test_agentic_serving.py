"""Serving the agentic artifacts: /v1/toolcallers + /v1/planners, local-or-escalate over HTTP."""

from __future__ import annotations

import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app

torch = pytest.importorskip("torch")
mixle_task = pytest.importorskip("mixle.task")


def _call_teacher(request):
    m = re.search(r"weather (?:in|for) (\w+)", request)
    if m:
        return {"tool": "get_weather", "args": {"city": m.group(1)}}
    m = re.search(r"search for (.+)$", request)
    if m:
        return {"tool": "search", "args": {"query": m.group(1)}}
    return {"tool": None, "args": {}}


def _plan_teacher(request):
    m = re.search(r"refund order (\d+) for (\w+)", request)
    if m:
        return [
            {"tool": "lookup_order", "args": {"order_id": m.group(1)}},
            {"tool": "notify", "args": {"user": m.group(2)}},
        ]
    m = re.search(r"check status of order (\d+)", request)
    if m:
        return [{"tool": "lookup_order", "args": {"order_id": m.group(1)}}]
    return []


def _call_requests(n, seed=0):
    rng = np.random.RandomState(seed)
    cities = ["paris", "tokyo", "denver", "oslo"]
    out = []
    for _ in range(n):
        r = rng.rand()
        if r < 0.45:
            out.append(f"please tell me the weather in {cities[rng.randint(0, 4)]} today")
        elif r < 0.8:
            out.append(f"can you search for item {rng.randint(1000, 9999)}")
        else:
            out.append(f"thanks so much, note {rng.randint(0, 99)}")
    return out


def _plan_requests(n, seed=0):
    rng = np.random.RandomState(seed)
    users = ["bob", "ana", "kim"]
    out = []
    for _ in range(n):
        oid, user = rng.randint(1000, 9999), users[rng.randint(0, 3)]
        if rng.rand() < 0.5:
            out.append(f"please refund order {oid} for {user} as discussed")
        else:
            out.append(f"can you check status of order {oid} right away")
    return out


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MIXLE_REGISTRY_ROOT", str(tmp_path / "registry"))
    get_settings.cache_clear()
    db._engine = None
    import mixle_mlops.gateway.routes.agentic as agentic_routes

    agentic_routes._CACHE.clear()
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _headers(client):
    key = client.post("/auth/signup", json={"email": "a@t.com", "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {key}"}


def test_toolcaller_serves_local_or_escalate(client):
    from mixle.task import ToolSpec, distill_tool_caller

    tools = [ToolSpec("get_weather", ["city"]), ToolSpec("search", ["query"])]
    tc = distill_tool_caller(
        _call_teacher, _call_requests(250), tools, seed=0,
        selector_kw={"ood": None, "epochs": 200}, extractor_kw={"epochs": 30},
    )
    tc.save(str(get_settings().registry_root / "toolcallers" / "assistant"))
    h = _headers(client)

    assert client.get("/v1/toolcallers", headers=h).json() == {"toolcallers": ["assistant"]}

    # served decisions match the local artifact exactly on fresh traffic
    for r in _call_requests(60, seed=7):
        got = client.post("/v1/toolcallers/assistant/call", json={"input": r}, headers=h).json()
        want = tc.try_local(r)
        if want is None:
            assert got == {"tool": None, "args": {}, "escalate": True}
        else:
            assert got == {**want, "escalate": False}

    fb = client.post(
        "/v1/toolcallers/assistant/feedback",
        json={"input": "weird request", "call": {"tool": "search", "args": {"query": "x"}}},
        headers=h,
    ).json()
    assert fb["harvested"] == 1
    ver = client.get("/v1/toolcallers/assistant/verification", headers=h).json()
    assert ver["selection_agreement"] > 0.8
    assert "get_weather" in ver["tools"]


def test_planner_serves_plans(client):
    from mixle.task import ToolSpec, distill_planner

    tools = [ToolSpec("lookup_order", ["order_id"]), ToolSpec("notify", ["user"])]
    planner = distill_planner(
        _plan_teacher, _plan_requests(250), tools, seed=0,
        selector_kw={"ood": None, "epochs": 200}, extractor_kw={"epochs": 30},
    )
    planner.save(str(get_settings().registry_root / "planners" / "ops"))
    h = _headers(client)

    got = client.post(
        "/v1/planners/ops/plan", json={"input": "please refund order 4242 for kim as discussed"}, headers=h
    ).json()
    if not got["escalate"]:
        assert [s["tool"] for s in got["plan"]] == ["lookup_order", "notify"]
        assert got["plan"][0]["args"]["order_id"] == "4242"

    ver = client.get("/v1/planners/ops/verification", headers=h).json()
    assert 0.0 <= ver["plan_agreement"] <= 1.0

    assert client.post("/v1/planners/nope/plan", json={"input": "x"}, headers=h).status_code == 404
