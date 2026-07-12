"""POST /v1/models/load: reads a completed fine-tune's registry_root/{name}/metadata.json and registers it
live, gated to admins. The underlying adapter-loading mechanism (does a PEFT adapter actually change served
behavior) is proven for real in test_local_engine.py; this file is about the route's own wiring: metadata
reading, backend gating, admin gating, and registry registration -- so load_local_engine itself is patched
to a recording stub rather than re-downloading/re-fitting a real model here.
"""
import json

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

import mixle_mlops.storage.db as db
from mixle_mlops.accounts import service as acct
from mixle_mlops.config import get_settings
from mixle_mlops.core.adapters import ChatCompletion, ModelAdapter
from mixle_mlops.gateway.app import create_app
from mixle_mlops.storage.db import get_engine


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _user_headers(client, email="u@t.com", *, is_admin=False):
    with Session(get_engine()) as s:
        user = acct.create_user(s, email, "pw12345", is_admin=is_admin)
        _rec, raw = acct.create_api_key(s, user)
    return {"Authorization": f"Bearer {raw}"}


def _write_metadata(registry_root, name, meta):
    root = registry_root / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(meta))


class _StubAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    async def chat(self, req):
        return ChatCompletion(model=self._name, choices=[])

    async def stream(self, req):
        return
        yield  # pragma: no cover - never reached, makes this an async generator


def test_non_admin_is_forbidden(client):
    h = _user_headers(client, "plain@t.com", is_admin=False)
    r = client.post("/v1/models/load", json={"name": "whatever"}, headers=h)
    assert r.status_code == 403


def test_unknown_name_is_404(client):
    h = _user_headers(client, "admin1@t.com", is_admin=True)
    r = client.post("/v1/models/load", json={"name": "does-not-exist"}, headers=h)
    assert r.status_code == 404


def test_unsupported_backend_is_400(client):
    h = _user_headers(client, "admin2@t.com", is_admin=True)
    _write_metadata(get_settings().registry_root, "a-mixle-artifact", {"backend": "mixle", "base_model": None})
    r = client.post("/v1/models/load", json={"name": "a-mixle-artifact"}, headers=h)
    assert r.status_code == 400
    assert "llm" in r.json()["detail"]


def test_missing_base_model_is_500(client):
    h = _user_headers(client, "admin3@t.com", is_admin=True)
    _write_metadata(get_settings().registry_root, "no-base-model", {"backend": "llm"})
    r = client.post("/v1/models/load", json={"name": "no-base-model"}, headers=h)
    assert r.status_code == 500


def test_admin_loads_an_llm_artifact_and_it_becomes_servable(client, monkeypatch):
    h = _user_headers(client, "admin4@t.com", is_admin=True)
    _write_metadata(
        get_settings().registry_root, "my-lora-model",
        {"backend": "llm", "base_model": "tiny/base", "artifact": "/artifacts/my-lora-model/adapter"},
    )

    calls = []

    def fake_load_local_engine(name, model_names, *, max_new_tokens=128, adapter_path=None):
        calls.append({"name": name, "model_names": model_names, "adapter_path": adapter_path})
        return _StubAdapter(name)

    monkeypatch.setattr("mixle_mlops.models.local_engine.load_local_engine", fake_load_local_engine)

    r = client.post("/v1/models/load", json={"name": "my-lora-model"}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"model": "my-lora-model", "base_model": "tiny/base", "adapter": "/artifacts/my-lora-model/adapter"}

    # the exact base_model + adapter_path from metadata.json reached load_local_engine
    assert calls == [{"name": "my-lora-model", "model_names": ["tiny/base"],
                      "adapter_path": "/artifacts/my-lora-model/adapter"}]

    # and it's now really in the live registry, discoverable like any other model
    models = client.get("/v1/models", headers=h).json()["data"]
    assert "my-lora-model" in [m["id"] for m in models]
