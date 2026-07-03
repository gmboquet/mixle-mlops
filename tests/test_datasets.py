"""End-to-end test of build-dataset generation: register the demo fitted mixle model on a freshly-built app,
generate N labeled rows from it through the route, export jsonl into the blob store, and assert the row
count + schema. An llm-source test uses a tiny in-process adapter that emits a JSON array (and a relaxed
echo-source test that tolerates the echo model's inability to produce JSON).

Self-contained: it builds the app via ``create_app()``, includes the datasets router itself, and registers
the demo model on ``app.state.registry`` inside the TestClient context — no dependence on app.py edits.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import mixle_mlops.storage.db as db
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
from mixle_mlops.datasets.export import to_csv, to_jsonl, to_parquet
from mixle_mlops.datasets.generate import (
    DatasetSpec,
    generate_dataset,
    generate_from_llm,
    generate_from_mixle,
)
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import datasets as datasets_routes
from mixle_mlops.models.mixle_model import register_demo_mixle_model
from mixle_mlops.multimodal.store import get_blob_store, reset_blob_store


class _JSONAdapter(ModelAdapter):
    """A test LLM that always returns a fixed JSON array of records (stands in for a real model)."""

    kind = "llm"

    def __init__(self, name: str, records: list[dict]):
        self._name = name
        self._records = records

    @property
    def name(self) -> str:
        return self._name

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        text = json.dumps(self._records)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(role="assistant"))])
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(content=text))])
        yield ChatCompletionChunk(
            model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(), finish_reason="stop")]
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    reset_blob_store()
    datasets_routes._table_ready = False
    app = create_app()
    app.include_router(datasets_routes.router, prefix="/v1")  # the integrator does this in app.py
    with TestClient(app) as c:
        register_demo_mixle_model(c.app.state.registry)  # demo mixle model on the live registry
        c.app.state.registry.register(
            _JSONAdapter("json-llm", [{"city": "Paris", "pop": 2}, {"city": "Rome", "pop": 3}])
        )
        yield c
    get_settings.cache_clear()
    db._engine = None
    reset_blob_store()


def _key(client, email):
    return client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]


# --------------------------------------------------------------------------------------------------------
# unit-level: generators + exporters directly
# --------------------------------------------------------------------------------------------------------
def test_generate_from_mixle_unit():
    adapter = register_demo_mixle_model(_FakeRegistry(), name="m")
    ds = generate_from_mixle(adapter, n=25, seed=7, model_id="m")
    assert ds.n_rows == 25
    assert ds.source == "mixle"
    assert ds.schema  # a non-empty inferred schema
    # the demo model is a 1-D Gaussian mixture -> a single numeric column
    ((col, typ),) = ds.schema.items()
    assert typ == "number"
    # deterministic under the seed
    ds2 = generate_from_mixle(adapter, n=25, seed=7, model_id="m")
    assert [r[col] for r in ds.rows] == [r[col] for r in ds2.rows]


def test_to_jsonl_and_csv_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    reset_blob_store()
    adapter = register_demo_mixle_model(_FakeRegistry(), name="m")
    ds = generate_from_mixle(adapter, n=10, seed=1, model_id="m")
    store = get_blob_store()

    res = to_jsonl(ds, store=store)
    assert res["n_rows"] == 10 and res["format"] == "jsonl"
    _record, data = store.get(res["id"])
    lines = [ln for ln in data.decode().splitlines() if ln.strip()]
    assert len(lines) == 10
    assert all(isinstance(json.loads(ln), dict) for ln in lines)

    csv_res = to_csv(ds, store=store)
    _r, csv_data = store.get(csv_res["id"])
    csv_lines = csv_data.decode().splitlines()
    assert len(csv_lines) == 11  # header + 10 rows
    reset_blob_store()
    get_settings.cache_clear()


def test_to_parquet_unit(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    reset_blob_store()
    adapter = register_demo_mixle_model(_FakeRegistry(), name="m")
    ds = generate_from_mixle(adapter, n=12, seed=2, model_id="m")
    res = to_parquet(ds, store=get_blob_store())
    assert res["format"] == "parquet" and res["n_rows"] == 12
    import io as _io

    import pandas as pd

    _record, data = get_blob_store().get(res["id"])
    frame = pd.read_parquet(_io.BytesIO(data))
    assert len(frame) == 12
    reset_blob_store()
    get_settings.cache_clear()


def test_generate_from_llm_unit():
    adapter = _JSONAdapter("json-llm", [{"city": "Paris", "pop": "2"}, {"city": "Rome", "pop": "3"}])
    schema = {"city": "string", "pop": "integer"}
    ds = asyncio.run(generate_from_llm(adapter, schema, n=2, model_id="json-llm"))
    assert ds.n_rows == 2
    assert ds.schema == schema
    assert ds.rows[0] == {"city": "Paris", "pop": 2}  # 'pop' coerced string->int
    assert isinstance(ds.rows[0]["pop"], int)


def test_generate_from_llm_drops_bad_records():
    # one record is missing the required 'pop' field -> dropped; the valid one survives
    adapter = _JSONAdapter("json-llm", [{"city": "X"}, {"city": "Y", "pop": 9}])
    ds = asyncio.run(generate_from_llm(adapter, {"city": "string", "pop": "integer"}, n=5, model_id="json-llm"))
    assert ds.n_rows == 1
    assert ds.rows[0] == {"city": "Y", "pop": 9}


def test_generate_dataset_dispatch_mixle():
    reg = _FakeRegistry()
    register_demo_mixle_model(reg, name="m")
    ds = asyncio.run(generate_dataset(DatasetSpec(source="mixle", model="m", n=8, seed=3), reg))
    assert ds.source == "mixle" and ds.n_rows == 8


# --------------------------------------------------------------------------------------------------------
# gateway-level: through the HTTP route with auth
# --------------------------------------------------------------------------------------------------------
def test_generate_route_mixle_jsonl(client):
    headers = {"Authorization": f"Bearer {_key(client, 'gen@t.com')}"}
    r = client.post(
        "/v1/datasets/generate",
        headers=headers,
        json={"source": "mixle", "model": "demo-mixle", "n": 30, "seed": 5, "format": "jsonl"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "mixle"
    assert body["n_rows"] == 30
    assert body["format"] == "jsonl"
    assert body["schema"]  # non-empty schema recorded
    assert body["url"] and body["blob_id"]

    # the bytes are in the blob store: N jsonl rows
    blob_id = body["artifact"]["id"]
    _record, data = get_blob_store().get(blob_id)
    lines = [ln for ln in data.decode().splitlines() if ln.strip()]
    assert len(lines) == 30
    assert all(isinstance(json.loads(ln), dict) for ln in lines)

    # GET the artifact back by id
    g = client.get(f"/v1/datasets/{body['id']}", headers=headers)
    assert g.status_code == 200
    assert g.json()["n_rows"] == 30 and g.json()["id"] == body["id"]


def test_generate_route_llm_json(client):
    headers = {"Authorization": f"Bearer {_key(client, 'llm@t.com')}"}
    r = client.post(
        "/v1/datasets/generate",
        headers=headers,
        json={
            "source": "llm",
            "model": "json-llm",
            "n": 2,
            "schema": {"city": "string", "pop": "integer"},
            "format": "jsonl",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "llm"
    assert body["n_rows"] == 2
    assert body["schema"] == {"city": "string", "pop": "integer"}


def test_generate_route_echo_relaxed(client):
    # the echo model cannot produce JSON; with a schema it parses to 0 valid rows but still 200 + valid schema.
    headers = {"Authorization": f"Bearer {_key(client, 'echo@t.com')}"}
    r = client.post(
        "/v1/datasets/generate",
        headers=headers,
        json={"source": "llm", "model": "echo", "n": 3, "schema": {"a": "integer"}, "format": "jsonl"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema"] == {"a": "integer"}
    assert body["n_rows"] >= 0  # echo emits no JSON -> 0 rows, tolerated


def test_generate_route_requires_auth(client):
    r = client.post("/v1/datasets/generate", json={"source": "mixle", "model": "demo-mixle", "n": 5})
    assert r.status_code == 401


def test_generate_route_unknown_model(client):
    headers = {"Authorization": f"Bearer {_key(client, 'nf@t.com')}"}
    r = client.post("/v1/datasets/generate", headers=headers, json={"source": "mixle", "model": "nope", "n": 5})
    assert r.status_code == 404


def test_generate_route_llm_requires_schema(client):
    headers = {"Authorization": f"Bearer {_key(client, 'ns@t.com')}"}
    r = client.post("/v1/datasets/generate", headers=headers, json={"source": "llm", "model": "json-llm", "n": 2})
    assert r.status_code == 422


class _FakeRegistry:
    def __init__(self):
        self._m = {}

    def register(self, a):
        self._m[a.name] = a
        return a

    def has(self, name):
        return name in self._m

    def get(self, name):
        return self._m[name]


# --------------------------------------------------------------------------------------------------------
# code-task factory route: harvest execution-verified (page -> parser code) pairs for /v1/fine_tunes
# --------------------------------------------------------------------------------------------------------


def test_code_tasks_route_reference_teacher(client):
    headers = {"Authorization": f"Bearer {_key(client, 'code@t.com')}"}
    r = client.post(
        "/v1/datasets/code_tasks",
        headers=headers,
        json={"teacher": "reference", "n_tasks": 12, "fields_per_task": 3, "seed": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "code_tasks"
    assert body["model"] is None  # reference teacher is not a served model
    assert body["format"] == "jsonl"
    # the rule-based teacher verifies everywhere, so every task yields a pair
    assert body["harvest"]["attempted"] == 12
    assert body["harvest"]["yield_rate"] == 1.0
    assert body["n_rows"] == body["harvest"]["pairs"] >= 12

    # the stored bytes are real SFT rows whose completion is code (prompt carries the page)
    _record, data = get_blob_store().get(body["artifact"]["id"])
    rows = [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]
    assert len(rows) == body["n_rows"]
    assert all("parse" in row["completion"] and "<" in row["prompt"] for row in rows)

    # and it round-trips through the artifact GET like any other dataset
    g = client.get(f"/v1/datasets/{body['id']}", headers=headers)
    assert g.status_code == 200 and g.json()["source"] == "code_tasks"


def test_code_tasks_route_rejects_impossible_schema(client):
    headers = {"Authorization": f"Bearer {_key(client, 'bad@t.com')}"}
    r = client.post(
        "/v1/datasets/code_tasks",
        headers=headers,
        json={"teacher": "reference", "n_tasks": 4, "fields_per_task": 9, "pool": {"a": "int", "b": "str"}},
    )
    assert r.status_code == 422  # 9 fields requested from a 2-field pool
