"""Serving solve() artifacts: /v1/tasks decide-or-escalate, feedback harvesting, verification record."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app

torch = pytest.importorskip("torch")
mixle_task = pytest.importorskip("mixle.task")


def _route(ticket):
    if ticket["amount"] > 500 and ticket["kind"] == "refund":
        return "finance-escalation"
    if ticket["kind"] in ("refund", "billing"):
        return "billing"
    return "support"


def _tickets(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["refund", "billing", "question", "bug"]
    return [
        {"kind": kinds[rng.randint(0, 4)], "amount": float(rng.gamma(2.0, 150.0)), "region": ["us", "eu"][rng.randint(0, 2)]}
        for _ in range(n)
    ]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MIXLE_REGISTRY_ROOT", str(tmp_path / "registry"))
    get_settings.cache_clear()
    db._engine = None
    import mixle_mlops.gateway.routes.tasks as tasks_routes

    tasks_routes._CACHE.clear()
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _headers(client):
    key = client.post("/auth/signup", json={"email": "t@t.com", "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {key}"}


def _deploy(client, name="router"):
    sol = mixle_task.solve(_route, _tickets(300), alpha=0.15, ood=0.05, seed=0, epochs=200)
    path = get_settings().registry_root / "tasks" / name
    sol.save(str(path))
    return sol


def test_decide_matches_the_local_model_and_escalates_aliens(client):
    sol = _deploy(client)
    h = _headers(client)

    assert client.get("/v1/tasks", headers=h).json() == {"tasks": ["router"]}

    for t in _tickets(40, seed=3):
        got = client.post("/v1/tasks/router/decide", json={"input": t}, headers=h).json()
        local = sol.cascade.model.decide(t)
        assert got["escalate"] == (local is None)
        if local is not None:
            assert got["label"] == local

    alien = {"kind": "zzz-never-seen", "amount": 1.0e9, "region": "??", "extra": "x" * 200}
    got = client.post("/v1/tasks/router/decide", json={"input": alien}, headers=h).json()
    assert got["escalate"] is True and got["label"] is None


def test_feedback_accumulates_harvested_pairs(client):
    _deploy(client)
    h = _headers(client)
    t = {"kind": "refund", "amount": 900.0, "region": "us"}
    r1 = client.post("/v1/tasks/router/feedback", json={"input": t, "label": _route(t)}, headers=h).json()
    r2 = client.post("/v1/tasks/router/feedback", json={"input": t, "label": _route(t)}, headers=h).json()
    assert (r1["harvested"], r2["harvested"]) == (1, 2)
    assert (get_settings().registry_root / "tasks" / "router" / "harvested.jsonl").exists()


def test_verification_record_is_served(client):
    sol = _deploy(client)
    h = _headers(client)
    body = client.get("/v1/tasks/router/verification", headers=h).json()
    assert body["kind"] == "record"
    ver = body["verification"]
    assert abs(ver["holdout_agreement"] - sol.holdout_agreement) < 1e-9
    assert ver["promoted"] is True


def test_unknown_task_is_404(client):
    h = _headers(client)
    assert client.post("/v1/tasks/nope/decide", json={"input": "x"}, headers=h).status_code == 404
