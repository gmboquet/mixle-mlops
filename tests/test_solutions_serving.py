"""Serving every solve shape through /v1/solutions — kind sniffing, decide contracts, harvest."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app

torch = pytest.importorskip("torch")
mixle_task = pytest.importorskip("mixle.task")


def _records(n, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {
            "kind": ["refund", "billing", "question", "bug"][rng.randint(0, 4)],
            "amount": float(rng.gamma(2.0, 150.0)),
            "region": ["us", "eu"][rng.randint(0, 2)],
        }
        for _ in range(n)
    ]


def _price(rec):  # clean linear signal -> tight qhat, deterministic answers_locally
    return 20.0 + 0.5 * rec["amount"] + (30.0 if rec["region"] == "eu" else 0.0)


def _flags(rec):
    out = []
    if rec["amount"] > 400:
        out.append("high-value")
    if rec["kind"] in ("refund", "billing"):
        out.append("money")
    if rec["region"] == "eu":
        out.append("eu-rules")
    return out


def _enrich(rec):
    return {"team": "billing" if rec["kind"] in ("refund", "billing") else "support", "price": _price(rec)}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MIXLE_REGISTRY_ROOT", str(tmp_path / "registry"))
    get_settings.cache_clear()
    db._engine = None
    import mixle_mlops.gateway.routes.solutions as sol_routes

    sol_routes._CACHE.clear()
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _headers(client):
    key = client.post("/auth/signup", json={"email": "t@t.com", "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {key}"}


def _deploy_regression(name="pricer"):
    sol = mixle_task.solve_regression(_price, _records(240), tol=1e6, alpha=0.1, seed=0, epochs=300)
    sol.save(str(get_settings().registry_root / "solutions" / name))
    return sol


def _deploy_multilabel(name="flagger"):
    sol = mixle_task.solve_multilabel(_flags, _records(240), alpha=0.1, seed=0, epochs=300)
    sol.save(str(get_settings().registry_root / "solutions" / name))
    return sol


def _deploy_structured(name="enricher"):
    sol = mixle_task.solve_structured(_enrich, _records(240), tol=1e6, alpha=0.1, seed=0, epochs=300)
    sol.save(str(get_settings().registry_root / "solutions" / name))
    return sol


class TestSolutionsServing:
    def test_list_sniffs_kinds(self, client):
        h = _headers(client)
        _deploy_regression()
        _deploy_multilabel()
        _deploy_structured()
        got = client.get("/v1/solutions", headers=h)
        assert got.status_code == 200
        assert got.json()["solutions"] == {"enricher": "structured", "flagger": "multilabel", "pricer": "regression"}

    def test_regression_decide_matches_local_artifact(self, client):
        h = _headers(client)
        sol = _deploy_regression()
        assert sol.answers_locally  # tol is generous by construction
        rec = {"kind": "billing", "amount": 320.0, "region": "eu"}
        got = client.post("/v1/solutions/pricer/decide", json={"input": rec}, headers=h)
        assert got.status_code == 200
        body = got.json()
        assert body["kind"] == "regression" and body["escalate"] is False
        assert body["value"] == pytest.approx(float(sol._predict([rec])[0]), abs=1e-6)
        assert body["qhat"] == pytest.approx(float(sol.qhat), abs=1e-9)

    def test_multilabel_decide_contract(self, client):
        h = _headers(client)
        sol = _deploy_multilabel()
        rec = {"kind": "refund", "amount": 900.0, "region": "eu"}
        got = client.post("/v1/solutions/flagger/decide", json={"input": rec}, headers=h)
        assert got.status_code == 200
        body = got.json()
        assert body["kind"] == "multilabel"
        local = sol.try_local(rec)
        if local is None:
            assert body["escalate"] is True and body["labels"] is None
        else:
            assert body["escalate"] is False and body["labels"] == local

    def test_structured_decide_contract(self, client):
        h = _headers(client)
        sol = _deploy_structured()
        rec = {"kind": "question", "amount": 120.0, "region": "us"}
        got = client.post("/v1/solutions/enricher/decide", json={"input": rec}, headers=h)
        assert got.status_code == 200
        body = got.json()
        assert body["kind"] == "structured"
        local = sol.try_local(rec)
        if local is None:
            assert body["escalate"] is True and body["output"] is None
        else:
            assert body["escalate"] is False
            assert set(body["output"]) == {"team", "price"}
            assert body["output"]["team"] == local["team"]
            assert body["output"]["price"] == pytest.approx(local["price"], abs=1e-6)

    def test_feedback_harvests(self, client):
        h = _headers(client)
        _deploy_regression()
        rec = {"kind": "bug", "amount": 5000.0, "region": "us"}
        got = client.post("/v1/solutions/pricer/feedback", json={"input": rec, "answer": 2520.0}, headers=h)
        assert got.status_code == 200 and got.json()["harvested"] == 1
        harvested = get_settings().registry_root / "solutions" / "pricer" / "harvested.jsonl"
        assert harvested.exists() and '"answer": 2520.0' in harvested.read_text()

    def test_verification_surfaces_each_shapes_trust(self, client):
        h = _headers(client)
        sol_r = _deploy_regression()
        _deploy_multilabel()
        _deploy_structured()

        v = client.get("/v1/solutions/pricer/verification", headers=h).json()
        assert v["kind"] == "regression" and v["answers_locally"] is True
        assert v["qhat"] == pytest.approx(float(sol_r.qhat), abs=1e-9)

        v = client.get("/v1/solutions/flagger/verification", headers=h).json()
        assert v["kind"] == "multilabel" and set(v["labels"]) == {"high-value", "money", "eu-rules"}
        assert 0.0 <= v["holdout_set_agreement"] <= 1.0

        v = client.get("/v1/solutions/enricher/verification", headers=h).json()
        assert v["kind"] == "structured" and set(v["fields"]) == {"team", "price"}
        assert v["fields"]["team"]["kind"] == "categorical" and "holdout_agreement" in v["fields"]["team"]
        assert v["fields"]["price"]["kind"] == "numeric" and v["fields"]["price"]["answers_locally"] is True

    def test_unknown_solution_404s(self, client):
        h = _headers(client)
        assert client.post("/v1/solutions/nope/decide", json={"input": "x"}, headers=h).status_code == 404
        assert client.get("/v1/solutions", headers=h).json()["solutions"] == {}

    def test_path_traversal_solution_name_is_rejected(self, client):
        # `name` is a URL path segment, so forward-slash traversal gets resolved/blocked by client or
        # routing before it reaches the handler -- backslash survives routing intact and is what
        # actually exercises _load()'s own validation, ahead of even the "does this exist" check.
        h = _headers(client)
        r = client.post(r"/v1/solutions/..\..\etc/decide", json={"input": "x"}, headers=h)
        assert r.status_code == 422
        assert "invalid name" in r.json()["detail"]
