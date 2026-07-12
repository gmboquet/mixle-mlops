"""End-to-end test of the mixle differentiator: register the demo mixle model on a freshly-built app,
then exercise predict / score / latent / decide / capabilities through the gateway with an auth'd key.

Self-contained: it builds the app via ``create_app()``, includes the mixle router itself, and registers
the demo model on ``app.state.registry`` inside the TestClient context -- no dependence on app.py edits.
"""

import mixle_mlops.storage.db as db
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.config import get_settings
from mixle_mlops.core.decision import bayes_action
from mixle_mlops.core.predictive import predictive_batch
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import mixle as mixle_routes
from mixle_mlops.models.mixle_model import register_demo_mixle_model


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    app.include_router(mixle_routes.router, prefix="/v1")  # the integrator does this in app.py
    with TestClient(app) as c:
        register_demo_mixle_model(c.app.state.registry)  # demo model on the live registry
        yield c
    get_settings.cache_clear()
    db._engine = None


def _key(client, email):
    return client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]


# --- unit-level: the core modules over a fitted distribution directly ---
def test_predictive_closed_form_gaussian():
    from mixle.inference import fit
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    g = fit(list(np.random.RandomState(0).normal(5.0, 2.0, 300)), GaussianDistribution(0.0, 1.0), max_its=20)
    preds = predictive_batch(g, [None, None])
    assert preds.path == "closed_form"
    assert preds.density_semantics == "exact"
    r = preds.records[0]
    assert abs(r.mean - 5.0) < 0.6
    assert r.interval[0] < r.mean < r.interval[1]
    # cdf-at-y and the cdf callable agree
    assert 0.4 < preds.cdf(0, r.mean) < 0.6


def test_predictive_ensemble_mixture():
    adapter = register_demo_mixle_model(_FakeRegistry(), name="m")
    preds = predictive_batch(adapter._model, [None])
    assert preds.path == "ensemble"
    assert preds.density_semantics == "estimate"
    assert len(preds.records[0].ensemble) == preds.n_ensemble


def test_bayes_action_picks_low_loss():
    from mixle.inference import fit, posterior
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    g = fit(list(np.random.RandomState(1).normal(10.0, 1.0, 400)), GaussianDistribution(0.0, 1.0), max_its=20)
    post = posterior(g, over="predictive")
    actions = [0.0, 5.0, 10.0, 15.0]
    res = bayes_action(post, lambda a, y: (float(a) - np.asarray(y, float)) ** 2, actions, n=4000)
    assert res["action"] == 10.0  # squared-error optimal ~ the mean
    assert "cvar" in res["risk_profile"]


class _FakeRegistry:
    def __init__(self):
        self._m = {}

    def register(self, a):
        self._m[a.name] = a
        return a


# --- gateway-level: through the HTTP routes with auth ---
def test_capabilities_route(client):
    headers = {"Authorization": f"Bearer {_key(client, 'c@t.com')}"}
    r = client.get("/v1/mixle/capabilities/demo-mixle", headers=headers)
    assert r.status_code == 200
    caps = r.json()["capabilities"]
    assert {"chat", "predict", "score", "latent", "decide"}.issubset(set(caps))
    assert r.json()["kind"] == "mixle"


def test_capabilities_requires_auth(client):
    assert client.get("/v1/mixle/capabilities/demo-mixle").status_code == 401


def test_served_manifest_carries_the_certificate(client):
    # /v1/models/{id} returns the adapter's info(); a mixle model's manifest is enriched with its
    # estimation certificate (guarantee + why-not-ADAM) — the served artifact is self-describing.
    headers = {"Authorization": f"Bearer {_key(client, 'cert@t.com')}"}
    r = client.get("/v1/models/demo-mixle", headers=headers)
    assert r.status_code == 200
    body = r.json()
    cert = body["certificate"]
    assert cert is not None
    assert cert["guarantee"] in {"HEURISTIC", "STATIONARY", "STATIONARY_ESCAPE_TESTED", "GLOBAL", "GLOBAL_UNIQUE"}
    assert isinstance(cert["guarantee_level"], int)
    assert "gradient descent" in cert["why_not_adam"].lower()  # the audit is present
    # the demo model didn't reserve a calibration holdout -> honestly null, not fabricated
    assert body["calibration"] is None


def test_predict_route(client):
    headers = {"Authorization": f"Bearer {_key(client, 'p@t.com')}"}
    r = client.post("/v1/mixle/predict", headers=headers, json={"model": "demo-mixle", "records": [None, None]})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "ensemble"
    assert len(body["records"]) == 2
    rec = body["records"][0]
    assert "mean" in rec and "quantiles" in rec and "interval" in rec


def test_score_route(client):
    headers = {"Authorization": f"Bearer {_key(client, 'sc@t.com')}"}
    r = client.post("/v1/mixle/score", headers=headers, json={"model": "demo-mixle", "records": [-3.0, 3.0, -12.0]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["log_density"]) == 3
    assert body["density_semantics"] == "exact"
    # points near the cluster means (+/-3) are far more likely than a far-tail outlier (-12)
    lp = body["log_density"]
    assert lp[0] > lp[2] and lp[1] > lp[2]


def test_latent_route(client):
    headers = {"Authorization": f"Bearer {_key(client, 'l@t.com')}"}
    r = client.post("/v1/mixle/latent", headers=headers, json={"model": "demo-mixle", "records": [-3.0, 3.0]})
    assert r.status_code == 200
    marg = r.json()["marginals"]
    assert len(marg) == 2 and len(marg[0]) == 2
    # the two records should be assigned to opposite components
    assert np.argmax(marg[0]) != np.argmax(marg[1])


def test_decide_route(client):
    headers = {"Authorization": f"Bearer {_key(client, 'd@t.com')}"}
    r = client.post(
        "/v1/mixle/decide",
        headers=headers,
        json={"model": "demo-mixle", "actions": [-3.0, 0.0, 3.0], "loss": "squared", "over": "predictive", "n": 3000},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in [-3.0, 0.0, 3.0]
    assert "expected_loss" in body and "risk_profile" in body
    assert len(body["alternatives"]) == 3


def test_chat_route_summarizes_predict(client):
    headers = {"Authorization": f"Bearer {_key(client, 'ch@t.com')}"}
    r = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "demo-mixle", "messages": [{"role": "user", "content": "[-3.0, 3.0]"}]},
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "demo-mixle" in content and "predictive distribution" in content


def test_capability_error_maps_to_422(client):
    """A model without Mixle capabilities yields 422 on ``/predict``."""
    headers = {"Authorization": f"Bearer {_key(client, 'e@t.com')}"}
    r = client.post("/v1/mixle/predict", headers=headers, json={"model": "echo", "records": [None]})
    assert r.status_code == 422
