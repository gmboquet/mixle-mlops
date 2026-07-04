"""Serving the telemetry sink: /v1/telemetry push / stats / training-rows for learned orchestration."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app

pytest.importorskip("mixle.telemetry")


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


def _headers(client):
    key = client.post("/auth/signup", json={"email": "t@t.com", "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {key}"}


class TestTelemetryServing:
    def test_push_and_stats(self, client):
        h = _headers(client)
        r = client.post(
            "/v1/telemetry",
            json={
                "events": [
                    {"kind": "placement", "features": {"tflop": 8.2}, "choice": "pool", "outcome": {"cost": 0.41}},
                    {"kind": "escalation", "features": {"conf": 0.6}, "choice": "escalate", "outcome": {"correct": True}},
                    {"kind": "placement", "features": {"tflop": 0.1}, "choice": "local", "outcome": {"cost": 0.0}},
                ]
            },
            headers=h,
        )
        assert r.status_code == 200 and r.json()["accepted"] == 3
        s = client.get("/v1/telemetry/stats", headers=h).json()
        assert s["n_events"] == 3 and s["kinds"]["placement"] == 2

    def test_training_rows_are_feature_choice_outcome(self, client):
        h = _headers(client)
        client.post(
            "/v1/telemetry",
            json={"events": [{"kind": "placement", "features": {"tflop": 8.2}, "choice": "pool", "outcome": {"cost": 0.41}}]},
            headers=h,
        )
        r = client.get("/v1/telemetry/training/placement", headers=h)
        rows = r.json()["rows"]
        assert len(rows) == 1
        assert rows[0] == {"features": {"tflop": 8.2}, "choice": "pool", "outcome": {"cost": 0.41}}

    def test_unknown_kind_is_rejected_not_fatal(self, client):
        h = _headers(client)
        r = client.post(
            "/v1/telemetry",
            json={"events": [{"kind": "telepathy", "features": {}}, {"kind": "route", "choice": "tier0"}]},
            headers=h,
        )
        body = r.json()
        assert body["accepted"] == 1 and body["rejected"] == 1  # the good event still landed

    def test_bad_body_422(self, client):
        h = _headers(client)
        assert client.post("/v1/telemetry", json={"nope": 1}, headers=h).status_code == 422

    def test_persistence_across_reload(self, client):
        h = _headers(client)
        client.post("/v1/telemetry", json={"events": [{"kind": "fit", "choice": "em"}]}, headers=h)
        # a fresh recorder reads the sink from disk (no in-process cache)
        assert client.get("/v1/telemetry/stats", headers=h).json()["n_events"] == 1

    def test_scoped_per_user(self, client):
        h1 = _headers(client)
        key2 = client.post("/auth/signup", json={"email": "u2@t.com", "password": "pw12345"}).json()["api_key"]
        h2 = {"Authorization": f"Bearer {key2}"}
        client.post("/v1/telemetry", json={"events": [{"kind": "fit", "choice": "em"}]}, headers=h1)
        assert client.get("/v1/telemetry/stats", headers=h2).json()["n_events"] == 0  # other user's sink is empty
