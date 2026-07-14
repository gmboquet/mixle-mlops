"""E5 -- ``POST /v1/interpret`` wires ``mixle.reason.language_bridge`` onto a physics posterior.

Self-contained: builds the app via ``create_app()`` (which already includes ``interpret.router`` per
``gateway/app.py``), signs up for an API key, and injects a small IC-1-conforming fixture posterior by
swapping the route's ``resolve_posterior`` hook -- standing in for ``mixle_pde.io.artifacts.load_posterior``
(IC-2) until that sibling module lands. A thin/diffuse posterior (spread >> ``tol``) must abstain; a
sharp posterior (spread << ``tol``) must yield a non-empty calibrated claim. Both assertions are made
in the SAME test, per the work order's Definition of Done.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import interpret as interpret_route


class _FakeFieldPosterior:
    """A minimal IC-1 (``mixle.reason.posterior_protocol.Posterior``) conforming fixture: one scalar
    field replicated over a small grid, with a caller-controlled mean/spread -- everything
    ``describe_posterior`` needs (``samples(n, rng)``), nothing ``PosteriorField3D``-specific."""

    def __init__(self, mean: float, sd: float, d: int = 4):
        self._mean = float(mean)
        self._sd = float(sd)
        self.d = d

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self._mean, self._sd, size=(n, self.d))

    @property
    def mean(self) -> np.ndarray:
        return np.full(self.d, self._mean)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None

    store: dict[str, object] = {}
    monkeypatch.setattr(interpret_route, "resolve_posterior", lambda ref: store[ref])

    app = create_app()
    with TestClient(app) as c:
        yield c, store
    get_settings.cache_clear()
    db._engine = None


def _auth_headers(client) -> dict:
    body = client.post("/auth/signup", json={"email": "e5@test.com", "password": "pw12345"}).json()
    return {"Authorization": f"Bearer {body['api_key']}"}


def test_sharp_posterior_claims_and_diffuse_posterior_abstains(client):
    c, store = client
    headers = _auth_headers(c)

    # sharp: total (summed over 4 cells) has mean ~400, sd ~0.1 -- far tighter than tol=0.5
    store["sharp"] = _FakeFieldPosterior(mean=100.0, sd=0.05)
    # diffuse: same mean, spread ~10000x tol -- no candidate claim width can cover it honestly
    store["diffuse"] = _FakeFieldPosterior(mean=100.0, sd=500.0)

    sharp_resp = c.post(
        "/v1/interpret",
        headers=headers,
        json={"posterior_ref": "sharp", "field": "total", "tol": 0.5, "level": 0.9},
    )
    assert sharp_resp.status_code == 200, sharp_resp.text
    sharp_body = sharp_resp.json()
    assert sharp_body["abstained"] is False
    assert sharp_body["claim"]  # non-empty calibrated claim string
    assert "total" in sharp_body["claim"]

    diffuse_resp = c.post(
        "/v1/interpret",
        headers=headers,
        json={"posterior_ref": "diffuse", "field": "total", "tol": 0.5, "level": 0.9},
    )
    assert diffuse_resp.status_code == 200, diffuse_resp.text
    diffuse_body = diffuse_resp.json()
    assert diffuse_body["abstained"] is True
    assert diffuse_body["claim"] == ""


def test_auth_required(client):
    c, store = client
    store["sharp"] = _FakeFieldPosterior(mean=1.0, sd=0.01)
    r = c.post("/v1/interpret", json={"posterior_ref": "sharp", "field": "total", "tol": 0.5, "level": 0.9})
    assert r.status_code == 401


def test_unresolvable_ref_is_404(client):
    c, _store = client
    headers = _auth_headers(c)
    r = c.post(
        "/v1/interpret",
        headers=headers,
        json={"posterior_ref": "does-not-exist", "field": "total", "tol": 0.5, "level": 0.9},
    )
    assert r.status_code == 404


def test_level_out_of_range_is_400(client):
    c, store = client
    headers = _auth_headers(c)
    store["sharp"] = _FakeFieldPosterior(mean=1.0, sd=0.01)
    r = c.post(
        "/v1/interpret",
        headers=headers,
        json={"posterior_ref": "sharp", "field": "total", "tol": 0.5, "level": 1.5},
    )
    assert r.status_code == 400
