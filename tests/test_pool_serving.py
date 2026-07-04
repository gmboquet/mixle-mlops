"""The pool plane over HTTP: job gateway rails (H2), bit-exact round-trip (H3), quotas/spend (H4)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app

mixle_pool = pytest.importorskip("mixle.pool")


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


def _headers(client, email="p@t.com"):
    key = client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {key}"}


def _fit_spec(n=60):
    data = [float(20 + (i % 2) * 80) for i in range(n)]
    return {"op": "fit", "data": data, "seed": 0, "max_its": 10}


class TestPoolRails:
    def test_free_fit_job_runs_and_certifies(self, client):
        h = _headers(client)
        r = client.post("/v1/pool/jobs", json={"kind": "verb", "reason": "test fit", "spec": _fit_spec()}, headers=h)
        assert r.status_code == 200
        job = r.json()
        assert job["status"] == "done"
        assert "guarantee" in job["summary"]  # the artifact is certified, not just fitted

    def test_over_budget_is_rejected_before_running(self, client):
        h = _headers(client)
        r = client.post(
            "/v1/pool/jobs",
            json={"est_cost": 5.0, "budget": 1.0, "confirm": True, "spec": _fit_spec()},
            headers=h,
        )
        job = r.json()
        assert job["status"] == "rejected"
        assert "exceeds budget" in job["reason_out"]

    def test_priced_job_requires_confirm(self, client):
        h = _headers(client)
        r = client.post("/v1/pool/jobs", json={"est_cost": 0.5, "spec": _fit_spec()}, headers=h)
        assert r.json()["status"] == "rejected"
        assert "never implicit" in r.json()["reason_out"]

    def test_quota_caps_cumulative_spend(self, client):
        h = _headers(client)
        # a job whose cost alone exceeds the default quota is rejected
        r = client.post(
            "/v1/pool/jobs",
            json={"est_cost": 1e6, "budget": 2e6, "confirm": True, "spec": _fit_spec()},
            headers=h,
        )
        assert r.json()["status"] == "rejected"
        assert "quota" in r.json()["reason_out"]

    def test_unknown_op_is_a_recorded_rejection(self, client):
        h = _headers(client)
        r = client.post("/v1/pool/jobs", json={"spec": {"op": "mine_bitcoin"}}, headers=h)
        assert r.json()["status"] == "rejected"


class TestRoundTrip:
    def test_artifact_round_trips_bit_exact(self, client):
        h = _headers(client)
        job = client.post(
            "/v1/pool/jobs", json={"reason": "fit for round-trip", "spec": _fit_spec()}, headers=h
        ).json()
        art = client.get(f"/v1/pool/jobs/{job['id']}/artifact", headers=h)
        assert art.status_code == 200
        payload = art.json()
        # reload the artifact locally and verify the fingerprint matches: bit-exact round-trip (H3)
        from mixle.inference import param_fingerprint
        from mixle.utils.serialization import from_json

        model = from_json(payload["model_json"])
        assert param_fingerprint(model) == payload["fingerprint"]
        assert payload["provenance"]["job_id"] == job["id"]  # provenance intact

    def test_artifact_of_rejected_job_409s(self, client):
        h = _headers(client)
        job = client.post("/v1/pool/jobs", json={"est_cost": 1.0, "spec": _fit_spec()}, headers=h).json()
        assert job["status"] == "rejected"
        assert client.get(f"/v1/pool/jobs/{job['id']}/artifact", headers=h).status_code == 409


class TestSpendLedger:
    def test_spend_accumulates_only_for_done_jobs(self, client):
        h = _headers(client)
        client.post("/v1/pool/jobs", json={"est_cost": 2.0, "confirm": True, "spec": _fit_spec()}, headers=h)
        client.post("/v1/pool/jobs", json={"est_cost": 1.0, "spec": _fit_spec()}, headers=h)  # rejected: no confirm
        s = client.get("/v1/pool/spend", headers=h).json()
        assert s["spent"] == 2.0  # the rejected job cost nothing
        assert s["n_done"] == 1 and s["n_rejected"] == 1
        assert s["remaining"] == s["quota"] - 2.0

    def test_queue_lists_this_users_jobs(self, client):
        h = _headers(client)
        client.post("/v1/pool/jobs", json={"spec": _fit_spec()}, headers=h)
        jobs = client.get("/v1/pool/jobs", headers=h).json()["jobs"]
        assert len(jobs) == 1
        # another user's queue is empty (per-user shard)
        h2 = _headers(client, email="other@t.com")
        assert client.get("/v1/pool/jobs", headers=h2).json()["jobs"] == []
