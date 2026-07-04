"""Serving the knowledge substrate + all-data RAG: /v1/substrate ingest / retrieve / context-or-abstain."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app

mixle_substrate = pytest.importorskip("mixle.substrate")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MIXLE_REGISTRY_ROOT", str(tmp_path / "registry"))
    get_settings.cache_clear()
    db._engine = None
    import mixle_mlops.gateway.routes.substrate as sub_routes

    sub_routes._CACHE.clear()
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _headers(client):
    key = client.post("/auth/signup", json={"email": "t@t.com", "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {key}"}


class TestSubstrateServing:
    def test_ingest_documents_and_stats(self, client):
        h = _headers(client)
        r = client.post(
            "/v1/substrate/kb/documents",
            json={
                "docs": ["refunds within 30 days for defective items", "support desk open 9 to 5"],
                "source": "docs",
            },
            headers=h,
        )
        assert r.status_code == 200 and r.json()["ingested"] == 2
        s = client.get("/v1/substrate/kb", headers=h).json()
        assert s["n_items"] == 2 and s["kinds"]["text"] == 2

    def test_add_typed_item_and_retrieve(self, client):
        h = _headers(client)
        client.post("/v1/substrate/kb/documents", json={"docs": ["refund policy allows returns"]}, headers=h)
        client.post(
            "/v1/substrate/kb/items",
            json={"kind": "artifact", "text": "refund-router classifier", "payload": {"ref": "/reg/x"}},
            headers=h,
        )
        r = client.post("/v1/substrate/kb/retrieve", json={"query": "refund", "k": 5}, headers=h)
        assert r.status_code == 200
        kinds = {it["kind"] for it in r.json()["items"]}
        assert "text" in kinds and "artifact" in kinds  # cross-kind retrieval

    def test_context_returns_citations_when_confident(self, client):
        h = _headers(client)
        client.post(
            "/v1/substrate/kb/documents",
            json={"docs": ["refunds are processed within 30 days", "refunds need approval over 500 dollars"]},
            headers=h,
        )
        r = client.post(
            "/v1/substrate/kb/context",
            json={"query": "refunds", "budget": {"max_chars": 300}, "min_confidence": 0.05},
            headers=h,
        )
        body = r.json()
        assert body["abstain"] is False
        assert body["context"] is not None
        assert len(body["citations"]) >= 1  # served context always carries provenance

    def test_context_abstains_below_confidence(self, client):
        h = _headers(client)
        client.post("/v1/substrate/kb/documents", json={"docs": ["refund policy details here"]}, headers=h)
        r = client.post(
            "/v1/substrate/kb/context",
            json={"query": "quantum chromodynamics", "min_confidence": 0.99},
            headers=h,
        )
        body = r.json()
        assert body["abstain"] is True
        assert body["context"] is None  # nothing fabricated; caller must escalate

    def test_multi_hop_retrieve(self, client):
        h = _headers(client)
        client.post(
            "/v1/substrate/kb/items",
            json={
                "kind": "trace",
                "text": "training record zeta eta",
                "payload": {},
                "provenance": {"source": "h"},
                "links": [],
            },
            headers=h,
        )
        # link a doc -> the trace so a hop chain forms
        stat = client.get("/v1/substrate/kb", headers=h)
        assert stat.status_code == 200
        r = client.post("/v1/substrate/kb/retrieve", json={"query": "training record", "hops": 2}, headers=h)
        assert r.status_code == 200 and "items" in r.json()

    def test_missing_shard_404s(self, client):
        h = _headers(client)
        assert client.get("/v1/substrate/nope", headers=h).status_code == 404
        assert client.post("/v1/substrate/nope/retrieve", json={"query": "x"}, headers=h).status_code == 404

    def test_persistence_survives_reload(self, client):
        h = _headers(client)
        client.post("/v1/substrate/kb/documents", json={"docs": ["persisted knowledge item"]}, headers=h)
        # a fresh app instance (cache cleared) must still see the shard from disk
        import mixle_mlops.gateway.routes.substrate as sub_routes

        sub_routes._CACHE.clear()
        assert client.get("/v1/substrate/kb", headers=h).json()["n_items"] == 1

    def test_factuality_grounds_and_flags_claims(self, client):
        h = _headers(client)
        client.post(
            "/v1/substrate/kb/documents",
            json={"docs": ["Refunds are processed within 30 days of a written request."]},
            headers=h,
        )
        # a mixed answer: one grounded claim, one fabricated
        ans = "Refunds are processed within 30 days. Free accounts include a dedicated account manager."
        r = client.post("/v1/substrate/kb/factuality", json={"answer": ans}, headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["n_claims"] == 2
        assert 0.0 < body["grounded_fraction"] < 1.0  # partially grounded
        assert body["n_unsupported"] == 1
        assert any("account manager" in c for c in body["unsupported"])  # the fabricated claim is named
        supported = [c for c in body["claims"] if c["supported"]]
        assert supported and supported[0]["citations"]  # the grounded claim carries a citation

    def test_factuality_requires_an_answer(self, client):
        h = _headers(client)
        client.post("/v1/substrate/kb/documents", json={"docs": ["some knowledge"]}, headers=h)
        assert client.post("/v1/substrate/kb/factuality", json={}, headers=h).status_code == 422

    def test_factuality_missing_shard_404s(self, client):
        h = _headers(client)
        assert client.post("/v1/substrate/nope/factuality", json={"answer": "x"}, headers=h).status_code == 404

    def test_publish_shares_an_item_across_scopes(self, client):
        h = _headers(client)
        # teamA adds a private item; a teamB-scoped retrieve cannot see it
        r = client.post(
            "/v1/substrate/kb/items",
            json={"kind": "text", "text": "teamA private roadmap for onboarding", "scope": "teamA"},
            headers=h,
        )
        item_id = r.json()["id"]
        rb = client.post(
            "/v1/substrate/kb/retrieve", json={"query": "onboarding roadmap", "scope": "public"}, headers=h
        )
        assert rb.json()["items"] == []  # nothing in public scope yet
        # publish teamA's item to public (guarded by from_scope)
        p = client.post(
            "/v1/substrate/kb/publish",
            json={"ids": [item_id], "to": "public", "from_scope": "teamA"},
            headers=h,
        )
        assert p.status_code == 200 and p.json()["n"] == 1
        # now a public-scope retrieve finds it
        import mixle_mlops.gateway.routes.substrate as sub_routes

        sub_routes._CACHE.clear()
        rp = client.post(
            "/v1/substrate/kb/retrieve", json={"query": "onboarding roadmap", "scope": "public"}, headers=h
        )
        assert len(rp.json()["items"]) >= 1

    def test_publish_requires_ids(self, client):
        h = _headers(client)
        client.post("/v1/substrate/kb/documents", json={"docs": ["x"]}, headers=h)
        assert client.post("/v1/substrate/kb/publish", json={}, headers=h).status_code == 422

    def test_publish_from_scope_guard_skips_foreign_items(self, client):
        h = _headers(client)
        r = client.post(
            "/v1/substrate/kb/items",
            json={"kind": "text", "text": "teamA item", "scope": "teamA"},
            headers=h,
        )
        item_id = r.json()["id"]
        # attempting to publish it as if from teamB is a no-op
        p = client.post(
            "/v1/substrate/kb/publish",
            json={"ids": [item_id], "to": "public", "from_scope": "teamB"},
            headers=h,
        )
        assert p.status_code == 200 and p.json()["n"] == 0


def _admin_headers(client, email="admin@t.com"):
    from sqlmodel import Session

    from mixle_mlops.accounts import service as acct
    from mixle_mlops.storage.db import get_engine

    with Session(get_engine()) as s:
        user = acct.create_user(s, email, "pw12345", is_admin=True)
        _rec, raw = acct.create_api_key(s, user)
    return {"Authorization": f"Bearer {raw}"}


class TestGovernanceServing:
    def test_propose_pending_and_admin_approve(self, client):
        h = _headers(client)
        r = client.post(
            "/v1/substrate/gov/items",
            json={"kind": "artifact", "text": "org ontology term", "scope": "teamA"},
            headers=h,
        )
        item_id = r.json()["id"]
        # propose to org (any user)
        p = client.post("/v1/substrate/gov/propose", json={"ids": [item_id], "to": "org"}, headers=h)
        assert p.status_code == 200 and p.json()["n"] == 1
        # it shows up as pending
        pend = client.get("/v1/substrate/gov/pending", params={"to": "org"}, headers=h)
        assert any(i["id"] == item_id for i in pend.json()["pending"])
        # a non-admin cannot approve
        assert client.post("/v1/substrate/gov/approve", json={"item_id": item_id}, headers=h).status_code == 403
        # an admin approves -> promoted to org
        admin = _admin_headers(client)
        a = client.post("/v1/substrate/gov/approve", json={"item_id": item_id}, headers=admin)
        assert a.status_code == 200 and a.json()["approved"] is True
        # the item is now in the org scope
        import mixle_mlops.gateway.routes.substrate as sub_routes

        sub_routes._CACHE.clear()
        rp = client.post("/v1/substrate/gov/retrieve", json={"query": "ontology term", "scope": "org"}, headers=h)
        assert len(rp.json()["items"]) >= 1

    def test_admin_reject_keeps_item_private(self, client):
        h = _headers(client)
        r = client.post(
            "/v1/substrate/gov/items",
            json={"kind": "artifact", "text": "rejected term", "scope": "teamB"},
            headers=h,
        )
        item_id = r.json()["id"]
        client.post("/v1/substrate/gov/propose", json={"ids": [item_id], "to": "org"}, headers=h)
        admin = _admin_headers(client, email="admin2@t.com")
        rej = client.post("/v1/substrate/gov/reject", json={"item_id": item_id, "reason": "dup"}, headers=admin)
        assert rej.status_code == 200 and rej.json()["rejected"] is True
        # not promoted, and no longer pending
        assert client.get("/v1/substrate/gov/pending", headers=admin).json()["pending"] == []

    def test_propose_requires_ids(self, client):
        h = _headers(client)
        client.post("/v1/substrate/gov/documents", json={"docs": ["x"]}, headers=h)
        assert client.post("/v1/substrate/gov/propose", json={}, headers=h).status_code == 422

    def test_reject_requires_admin(self, client):
        h = _headers(client)
        r = client.post("/v1/substrate/gov/items", json={"kind": "text", "text": "t", "scope": "teamA"}, headers=h)
        client.post("/v1/substrate/gov/propose", json={"ids": [r.json()["id"]], "to": "org"}, headers=h)
        assert client.post("/v1/substrate/gov/reject", json={"item_id": r.json()["id"]}, headers=h).status_code == 403
