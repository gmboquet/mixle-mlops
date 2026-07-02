"""User training end-to-end: POST a labeled dataset to /v1/fine_tunes, get back a hosted model.

The ``structured`` backend distils a tiny structured probabilistic classifier locally (no GPU, no torch) and
registers it live -- so right after the call it appears in /v1/models and answers /v1/mixle/predict. Also checks
the GPU-backend *plan* path, ownership isolation, listing, and cancel. Background training runs synchronously
under Starlette's TestClient, so the job is already terminal when the POST returns.
"""
from __future__ import annotations

from typing import AsyncIterator

import mixle_mlops.storage.db as db
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.config import get_settings
from mixle_mlops.core.adapters import (
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatRequest,
    ChoiceDelta,
    ModelAdapter,
)
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import fine_tunes as ft_routes


class _RuleTeacher(ModelAdapter):
    """A stand-in hosted teacher (as a native Claude/Gemini adapter would be): applies the churn rule and replies
    with the label. Distilling it exercises the platform's teacher->tiny-student loop without a network call."""

    kind = "llm"

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        import json as _json

        raw = req.messages[-1].text()
        r = _json.loads(raw)
        score = (1.5 if r["region"] == "west" else -0.5) + 0.4 * r["spend"] + 0.3 * r["visits"]
        label = "churn" if score < 1.0 else "retain"
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(content=label))])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    ft_routes._table_ready = False
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _key(client, email):
    return client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]


def _churn_records(n, seed):
    rng = np.random.RandomState(seed)
    recs = []
    for _ in range(n):
        region = rng.choice(["west", "east"])
        spend = float(rng.normal(2.0, 1.5))
        visits = int(rng.poisson(3))
        score = (1.5 if region == "west" else -0.5) + 0.4 * spend + 0.3 * visits
        recs.append({"region": region, "spend": spend, "visits": visits,
                     "label": "churn" if score < 1.0 else "retain"})
    return recs


def test_structured_finetune_trains_and_serves(client):
    headers = {"Authorization": f"Bearer {_key(client, 'ft@t.com')}"}
    body = {"backend": "structured", "model": "churn-clf",
            "records": _churn_records(400, 1), "label_field": "label", "min_gain": 1.0}
    created = client.post("/v1/fine_tunes", headers=headers, json=body)
    assert created.status_code == 200, created.text
    job = created.json()
    assert job["backend"] == "structured" and job["model"] == "churn-clf"

    # background training runs synchronously under TestClient -> the job is already done
    got = client.get(f"/v1/fine_tunes/{job['id']}", headers=headers).json()
    assert got["status"] == "succeeded", got
    assert got["metrics"]["train_agreement"] >= 0.8
    assert got["metrics"]["edges"]  # the discovered dependency structure

    # the trained model is now hosted
    models = {m["id"] for m in client.get("/v1/models", headers=headers).json()["data"]}
    assert "churn-clf" in models

    # and it answers predictions on new records
    test = _churn_records(20, 99)
    pred = client.post("/v1/mixle/predict", headers=headers,
                       json={"model": "churn-clf", "records": [{k: r[k] for k in ("region", "spend", "visits")}
                                                               for r in test]})
    assert pred.status_code == 200, pred.text
    results = pred.json()["results"]
    agree = np.mean([p == r["label"] for p, r in zip(results, test)])
    assert agree >= 0.7


def test_structured_finetune_accepts_parallel_labels(client):
    headers = {"Authorization": f"Bearer {_key(client, 'ft2@t.com')}"}
    recs = _churn_records(300, 2)
    feats = [{k: r[k] for k in ("region", "spend", "visits")} for r in recs]
    labels = [r["label"] for r in recs]
    job = client.post("/v1/fine_tunes", headers=headers,
                      json={"backend": "structured", "records": feats, "labels": labels}).json()
    got = client.get(f"/v1/fine_tunes/{job['id']}", headers=headers).json()
    assert got["status"] == "succeeded"
    assert got["model"].startswith("ft:")  # default served id derived from the job id


def test_bad_structured_request_is_rejected(client):
    headers = {"Authorization": f"Bearer {_key(client, 'ft3@t.com')}"}
    r = client.post("/v1/fine_tunes", headers=headers, json={"backend": "structured", "records": []})
    assert r.status_code == 400


def test_gpu_backend_returns_plan(client):
    headers = {"Authorization": f"Bearer {_key(client, 'ft4@t.com')}"}
    job = client.post("/v1/fine_tunes", headers=headers, json={
        "backend": "llm", "model": "my-lora", "base_model": "sshleifer/tiny-gpt2",
        "dataset": "data.jsonl", "epochs": 1.0}).json()
    assert job["status"] == "planned"
    plan = job["metrics"]["plan"]
    assert "training_command" in plan and "offer_query" in plan


def test_list_and_ownership_isolation(client):
    ha = {"Authorization": f"Bearer {_key(client, 'owner@t.com')}"}
    hb = {"Authorization": f"Bearer {_key(client, 'other@t.com')}"}
    job = client.post("/v1/fine_tunes", headers=ha,
                      json={"backend": "structured", "records": _churn_records(200, 3),
                            "label_field": "label"}).json()
    # owner sees it; the other user does not, and cannot fetch it by id
    assert any(j["id"] == job["id"] for j in client.get("/v1/fine_tunes", headers=ha).json()["data"])
    assert all(j["id"] != job["id"] for j in client.get("/v1/fine_tunes", headers=hb).json()["data"])
    assert client.get(f"/v1/fine_tunes/{job['id']}", headers=hb).status_code == 404


def test_cancel_terminal_job_conflicts(client):
    headers = {"Authorization": f"Bearer {_key(client, 'ft5@t.com')}"}
    job = client.post("/v1/fine_tunes", headers=headers,
                      json={"backend": "structured", "records": _churn_records(200, 4),
                            "label_field": "label"}).json()
    # it already succeeded (sync background) -> cancel is a 409
    assert client.post(f"/v1/fine_tunes/{job['id']}/cancel", headers=headers).status_code == 409


def test_distill_hosted_teacher_into_structured_student(client):
    # register a hosted teacher (stands in for a Claude/Gemini adapter) on the live registry
    client.app.state.registry.register(_RuleTeacher("teacher-llm"))
    headers = {"Authorization": f"Bearer {_key(client, 'distill@t.com')}"}
    # unlabeled feature records + a teacher model -> the platform labels then distills a tiny structured student
    feats = [{k: r[k] for k in ("region", "spend", "visits")} for r in _churn_records(400, 5)]
    job = client.post("/v1/fine_tunes", headers=headers, json={
        "backend": "structured", "model": "distilled-clf", "records": feats,
        "teacher_model": "teacher-llm", "teacher_labels": ["churn", "retain"], "min_gain": 1.0,
    }).json()
    got = client.get(f"/v1/fine_tunes/{job['id']}", headers=headers).json()
    assert got["status"] == "succeeded", got
    assert got["metrics"]["teacher_model"] == "teacher-llm"       # provenance: distilled from the hosted teacher
    assert got["metrics"]["train_agreement"] >= 0.8
    assert "distilled-clf" in {m["id"] for m in client.get("/v1/models", headers=headers).json()["data"]}


def test_unknown_teacher_model_is_rejected(client):
    headers = {"Authorization": f"Bearer {_key(client, 'noteacher@t.com')}"}
    r = client.post("/v1/fine_tunes", headers=headers, json={
        "backend": "structured", "records": [{"a": 1}], "teacher_model": "ghost"})
    assert r.status_code == 404


def test_requires_authentication(client):
    r = client.post("/v1/fine_tunes", json={"backend": "structured", "records": [], "label_field": "label"})
    assert r.status_code == 401
