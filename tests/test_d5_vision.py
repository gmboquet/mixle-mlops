"""D5 — vision-capability routing + image georeferencing sidecar.

Two independent surfaces:

1. the chat route auto-selects a registered vision-capable model when the requested model can't see and the
   request carries an image (``has_vision``/``select_vision_model``, wired into ``chat_completions``);
2. a tiled-raster tile can become a serializable, content-addressed ``StructuredMediaRef`` that survives a
   ``to_dict()``/JSON/``from_dict()`` round trip (i.e. a different process) and still materializes back to an
   ``ImagePart`` with the same content hash and spatial frame — with no process-local ``dict[id(part)]``
   registry anywhere in the module.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json

import mixle_mlops.multimodal.store as store_mod
import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

import mixle_mlops.multimodal.content as content
from mixle_mlops.config import get_settings
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.app import create_app
from mixle_mlops.models.echo import EchoAdapter
from mixle_mlops.multimodal.content import (
    GeoRef,
    MultimodalError,
    StructuredMediaRef,
    attach_georef,
    has_vision,
    media_ref_from_tile,
    select_vision_model,
)
from mixle_mlops.multimodal.store import LocalBlobStore

# a 1x1 transparent PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class VisionEchoAdapter(EchoAdapter):
    """A vision-capable echo model — no real backend needed to exercise the routing gate."""

    def capabilities(self) -> set[str]:
        return {"chat", "vision"}


# --- part 1: vision-capability gate in the chat route -----------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    store_mod.reset_blob_store()
    app = create_app()
    with TestClient(app) as c:
        app.state.registry.register(VisionEchoAdapter("vision-echo"))
        yield c
    get_settings.cache_clear()
    db._engine = None
    store_mod.reset_blob_store()


def _signup(client) -> dict:
    raw = client.post("/auth/signup", json={"email": "v@t.com", "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def test_has_vision():
    assert has_vision(VisionEchoAdapter("v")) is True
    assert has_vision(EchoAdapter("echo")) is False


def test_select_vision_model_prefers_requested_when_capable():
    registry = ModelRegistry()
    registry.register(EchoAdapter("echo"))
    registry.register(VisionEchoAdapter("vision-echo"))
    assert select_vision_model(registry, "vision-echo") == "vision-echo"


def test_select_vision_model_reroutes_and_is_deterministic():
    registry = ModelRegistry()
    registry.register(EchoAdapter("echo"))
    registry.register(VisionEchoAdapter("vision-b"))
    registry.register(VisionEchoAdapter("vision-a"))
    # "echo" has no vision -> falls back to a vision-capable model; ties (no cost signal) break on model id.
    assert select_vision_model(registry, "echo") == "vision-a"


def test_select_vision_model_raises_when_none_registered():
    registry = ModelRegistry()
    registry.register(EchoAdapter("echo"))
    with pytest.raises(MultimodalError):
        select_vision_model(registry, "echo")


def test_chat_route_auto_selects_vision_model(client):
    headers = _signup(client)
    payload = {
        "model": "echo",  # text-only; request also carries an image
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")},
                    },
                ],
            }
        ],
    }
    r = client.post("/v1/chat/completions", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "vision-echo"  # resolved model id, not the requested "echo"
    assert r.headers.get("X-Vision-Model") == "vision-echo"


def test_chat_route_400_when_no_vision_model_available(client, monkeypatch):
    headers = _signup(client)
    # strip every vision-capable model from the registry so the gate must fail loudly
    app = client.app
    for model_id in list(app.state.registry._models):
        if has_vision(app.state.registry.get(model_id)):
            del app.state.registry._models[model_id]
    payload = {
        "model": "echo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")},
                    },
                ],
            }
        ],
    }
    r = client.post("/v1/chat/completions", headers=headers, json=payload)
    assert r.status_code == 400


# --- part 2: RasterTile -> StructuredMediaRef -> JSON -> new process -> ImagePart ---------------------------


@dataclasses.dataclass
class _FakeRasterTile:
    """Duck-typed stand-in for D1's ``RasterTile`` (``png``/``row``/``col``/``extent``/``scale``): D1 has not
    landed in this branch yet, so this fixture exercises the same shape `media_ref_from_tile` reads."""

    png: bytes
    row: int
    col: int
    extent: tuple[float, float, float, float]
    scale: float


def test_raster_tile_survives_json_round_trip_to_a_new_process(tmp_path):
    store = LocalBlobStore(tmp_path / "blobs")
    tile = _FakeRasterTile(png=_PNG, row=0, col=0, extent=(10.0, 20.0, 30.0, 40.0), scale=0.5)

    media = media_ref_from_tile(tile, store, crs="EPSG:32611", provenance={"source": "survey-1"})
    media = attach_georef(
        media,
        GeoRef(
            crs="EPSG:32611",
            extent=media.georef.extent,
            scale=media.georef.scale,
            pixel_to_crs=(0.5, 0, 10.0, 0, -0.5, 40.0),
        ),
    )

    # serialize, ship across a process boundary (a fresh dict via json), and reconstruct
    wire = json.loads(json.dumps(media.to_dict()))
    # a brand-new StructuredMediaRef/GeoRef, unrelated to the original Python objects (simulates a new process)
    reconstructed = StructuredMediaRef.from_dict(wire)

    assert reconstructed.content_hash == media.content_hash
    assert reconstructed.georef.extent == (10.0, 20.0, 30.0, 40.0)
    assert reconstructed.georef.pixel_to_crs == (0.5, 0, 10.0, 0, -0.5, 40.0)

    # materialize back to an ImagePart only at this (new) boundary; bytes + hash are unchanged
    part = reconstructed.to_image_part(store)
    prefix = "data:image/png;base64,"
    assert part.image_url["url"].startswith(prefix)
    recovered = base64.b64decode(part.image_url["url"][len(prefix) :])
    assert recovered == _PNG
    assert hashlib.sha256(recovered).hexdigest() == media.content_hash


def test_no_module_level_id_keyed_registry():
    """The sidecar is pure data: assert `content.py` defines no module-level dict (an id(part)-keyed cache)."""
    dict_globals = {
        name: value for name, value in vars(content).items() if not name.startswith("_") and isinstance(value, dict)
    }
    assert dict_globals == {}
