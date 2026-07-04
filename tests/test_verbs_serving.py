"""Verb serving twins (I1): /v1/create, /v1/uq, /v1/simulate, /v1/synthesize, /v1/skills."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app

mixle_inference = pytest.importorskip("mixle.inference")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MIXLE_REGISTRY_ROOT", str(tmp_path / "registry"))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _headers(client, email="v@t.com"):
    key = client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {key}"}


def _scalars(n=200):
    import numpy as np

    return [float(x) for x in np.random.RandomState(0).normal(5, 2, n)]


def _ramp(n=200):
    # deterministic near-uniform ramp: the auto-detector picks a family uq() cannot Laplace-flatten
    return [float(5 + 2 * ((i * 7919) % 100) / 100) for i in range(n)]


def _create(client, h, **kw):
    return client.post("/v1/create", json={"data": _scalars(), **kw}, headers=h).json()


class TestCreate:
    def test_create_returns_a_certified_stored_artifact(self, client):
        h = _headers(client)
        out = _create(client, h)
        assert out["model_id"]
        assert out["guarantee"] in {"GLOBAL", "GLOBAL_UNIQUE", "STATIONARY"}
        assert "gradient descent" in out["why"].lower()

    def test_create_with_calibration_renders_a_verdict(self, client):
        h = _headers(client)
        out = _create(client, h, calibrate=0.3)
        assert out["is_calibrated"] in (True, False)


class TestUqAndSimulate:
    def test_uq_gives_a_credible_interval_for_a_stored_model(self, client):
        h = _headers(client)
        model_id = _create(client, h)["model_id"]
        r = client.post("/v1/uq", json={"model_id": model_id, "data": _scalars()}, headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "parameter_posterior"
        lo, hi = body["interval"]
        assert lo < hi  # a real interval on the predictive mean
        assert lo < 5.5 and hi > 4.5  # brackets the true mean ~5

    def test_uq_degrades_honestly_when_unflattenable(self, client):
        h = _headers(client)
        out = client.post("/v1/create", json={"data": _ramp()}, headers=h).json()
        r = client.post("/v1/uq", json={"model_id": out["model_id"], "data": _ramp()}, headers=h)
        assert r.status_code == 200
        assert r.json()["kind"] is None  # honest "not quantifiable", not a fabricated interval
        assert "not quantifiable" in r.json()["note"]

    def test_uq_unknown_model_404s(self, client):
        h = _headers(client)
        assert client.post("/v1/uq", json={"model_id": "nope", "data": _scalars()}, headers=h).status_code == 404

    def test_simulate_draws_from_the_stored_model(self, client):
        h = _headers(client)
        model_id = _create(client, h)["model_id"]
        r = client.post("/v1/simulate", json={"model_id": model_id, "n": 25, "seed": 1}, headers=h)
        assert r.status_code == 200
        assert r.json()["n"] == 25

    def test_simulate_interventions_on_a_scalar_model_422(self, client):
        h = _headers(client)
        model_id = _create(client, h)["model_id"]
        r = client.post("/v1/simulate", json={"model_id": model_id, "n": 5, "interventions": {"0": 1.0}}, headers=h)
        assert r.status_code == 422  # a scalar model has no do-operator; honest error


class TestSynthesize:
    def test_synthesize_with_a_declarative_constraint(self, client):
        h = _headers(client)
        r = client.post(
            "/v1/synthesize",
            json={"data": _scalars(), "n": 20, "constraint": {"min": 6.0}, "seed": 0},
            headers=h,
        )
        assert r.status_code == 200
        body = r.json()
        assert all(v >= 6.0 for v in body["rows"])  # only verified rows shipped
        assert body["acceptance_rate"] <= 1.0
        assert body["constraint"] == {"min": 6.0}  # the verifier travels with the data

    def test_synthesize_from_a_stored_model(self, client):
        h = _headers(client)
        model_id = _create(client, h)["model_id"]
        r = client.post("/v1/synthesize", json={"model_id": model_id, "n": 10}, headers=h)
        assert r.status_code == 200 and r.json()["n"] == 10


class TestSkills:
    def test_register_and_find_a_skill(self, client):
        h = _headers(client)
        model_id = _create(client, h)["model_id"]
        r = client.post(
            "/v1/skills",
            json={
                "name": "spend_model",
                "model_id": model_id,
                "description": "sample customer spend",
                "tags": ["spend"],
            },
            headers=h,
        )
        assert r.status_code == 200
        assert r.json()["guarantee"]  # the skill inherits the artifact's certificate
        found = client.get("/v1/skills", params={"query": "customer spend"}, headers=h).json()["skills"]
        assert found and found[0]["name"] == "spend_model"
        # a non-matching query finds nothing
        assert client.get("/v1/skills", params={"query": "quantum"}, headers=h).json()["skills"] == []

    def test_register_requires_name_and_model(self, client):
        h = _headers(client)
        assert client.post("/v1/skills", json={"name": "x"}, headers=h).status_code == 422


class TestLineage:
    def test_lineage_walks_data_to_model_to_skills(self, client):
        h = _headers(client)
        out = _create(client, h)
        model_id = out["model_id"]
        assert out["data_fingerprint"]  # the lineage edge back to the exact training data
        client.post(
            "/v1/skills",
            json={"name": "spend_skill", "model_id": model_id, "description": "spend"},
            headers=h,
        )
        r = client.get(f"/v1/lineage/{model_id}", headers=h)
        assert r.status_code == 200
        lin = r.json()
        assert lin["data_fingerprint"] == out["data_fingerprint"]  # data -> model
        assert lin["skills"] == ["spend_skill"]  # model -> skill
        kinds = {e["kind"] for e in lin["edges"]}
        assert kinds == {"fit", "exposes"}  # the full chain as queryable edges

    def test_lineage_unknown_model_404s(self, client):
        h = _headers(client)
        assert client.get("/v1/lineage/nope", headers=h).status_code == 404
