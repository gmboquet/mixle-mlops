"""I5 -- drillhole plan/collar/trace extraction.

Exercises `extract_collars` against a stubbed VLM response (`tests/fixtures/collars_stub.json`) so it is
deterministic and needs no live model call: reproject pixel-space collar markers and plotted hole traces
through an already-fitted (I1-style) pixel->CRS affine, merge the two near-duplicate detections the stub
gives for "DDH-01" (a re-detection artifact 1-2 px apart) into a single collar, and check every collar
registers to the expected UTM coordinates in `tests/fixtures/collars_truth.csv` within 5 m.

``_query_vlm_collars`` itself is wired to the real capability-routed vision layer (same pattern as
``_query_vlm``): rather than monkeypatching it away wholesale, these tests swap only
``map_digitize._resolve_vision_adapter`` for a *real* :class:`OpenAICompatAdapter` running against an
in-process ``httpx.MockTransport`` -- the same recorded-response style ``tests/test_providers.py`` uses for
the native Anthropic/Gemini adapters. Everything downstream of adapter resolution (prompt construction, the
adapter's actual request/response cycle, JSON-object extraction, reply-shape normalization) runs for real.
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import httpx
import pytest

from mixle_mlops.models.openai_compat import OpenAICompatAdapter
from mixle_mlops.multimodal import map_digitize
from mixle_mlops.multimodal.map_digitize import DrillholeLayer, extract_collars
from mixle_mlops.multimodal.store import LocalBlobStore

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "collars_stub.json").read_text())


def _stub_vision_adapter(monkeypatch, payload: dict, *, content: str | None = None) -> None:
    """Stand in only for ``_resolve_vision_adapter`` -- the seam between map_digitize's business logic and
    "which model/backend serves vision" -- with a real ``OpenAICompatAdapter`` wired to a mock transport
    that returns ``content`` (default: ``payload`` as a plain JSON string) as the model's chat reply.
    ``_query_vlm_collars`` then runs unmodified: real prompt building, a real (in-process) HTTP
    request/response cycle, real ``_extract_json_object``/``json.loads`` parsing."""
    body = content if content is not None else json.dumps(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-stub",
                "model": "qwen-vl-stub",
                "choices": [{"message": {"content": body}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    adapter = OpenAICompatAdapter(
        "qwen-vl-stub",
        base_url="https://vlm.example/v1",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(map_digitize, "_resolve_vision_adapter", lambda vlm: adapter)


with open(Path(__file__).parent / "fixtures" / "collars_truth.csv", newline="") as fh:
    TRUTH = {row["hole_id"]: row for row in csv.DictReader(fh)}

# same 100x100 px, north-up, 1 m/px raster as test_map_digitize.py's control points: pixel (0,0) -> UTM
# (500000, 4100100), y flips down the page. Exact affine (no least-squares residual): x = px + 500000,
# y = -py + 4100100.
_PIXEL_TO_CRS = (1.0, 0.0, 500000.0, 0.0, -1.0, 4100100.0)

# a 1x1 transparent PNG -- a stand-in drillhole-plan image; the "VLM" response is fully stubbed via
# monkeypatch below, so the actual pixel content never matters to this test.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(root=tmp_path / "blobs")


def _put_image(blob_store) -> str:
    record = blob_store.put(_PNG, filename="collar-plan.png", content_type="image/png")
    return record.id


def test_collars_register_to_crs(store, monkeypatch):
    _stub_vision_adapter(monkeypatch, FIXTURE)

    image_ref = _put_image(store)
    result = extract_collars(
        image_ref=image_ref,
        pixel_to_crs=_PIXEL_TO_CRS,
        crs="EPSG:32611",
        store=store,
        vlm="stub-vlm",
    )

    assert isinstance(result, DrillholeLayer)
    assert result.crs == "EPSG:32611"

    # the two near-duplicate DDH-01 detections merge into a single collar
    assert {c["hole_id"] for c in result.collars} == set(TRUTH)
    assert len(result.collars) == len(TRUTH)

    by_hole = {c["hole_id"]: c for c in result.collars}
    for hole_id, truth_row in TRUTH.items():
        collar = by_hole[hole_id]
        assert abs(collar["x"] - float(truth_row["x"])) < 5.0
        assert abs(collar["y"] - float(truth_row["y"])) < 5.0
        assert abs(collar["z"] - float(truth_row["z"])) < 1e-6

    assert result.provenance["n_collars"] == len(TRUTH)
    assert len(result.provenance["image_content_hash"]) == 64

    # plotted traces reproject too, tagged by hole_id
    trace_ids = {t["hole_id"] for t in result.traces}
    assert trace_ids == {"DDH-01", "DDH-02"}
    ddh1_trace = next(t for t in result.traces if t["hole_id"] == "DDH-01")
    assert ddh1_trace["coordinates"][0] == pytest.approx([500020.0, 4100080.0])


def test_collars_require_hole_id(store, monkeypatch):
    _stub_vision_adapter(monkeypatch, {"collars": [{"pixel_coords": [[5, 5]], "z": 1000.0}], "traces": []})
    image_ref = _put_image(store)
    result = extract_collars(
        image_ref=image_ref,
        pixel_to_crs=_PIXEL_TO_CRS,
        crs="EPSG:32611",
        store=store,
    )
    assert result.collars == []


def test_query_vlm_collars_tolerates_markdown_fenced_reply(store, monkeypatch):
    """The VLM sometimes wraps its JSON in a ```json ... ``` fence despite being asked not to;
    ``_extract_json_object`` must still find and parse the object (exercised here through the real
    adapter/parse path, not just monkeypatched around)."""
    fenced = "```json\n" + json.dumps(FIXTURE) + "\n```"
    _stub_vision_adapter(monkeypatch, FIXTURE, content=fenced)
    image_ref = _put_image(store)

    result = extract_collars(
        image_ref=image_ref,
        pixel_to_crs=_PIXEL_TO_CRS,
        crs="EPSG:32611",
        store=store,
    )
    assert {c["hole_id"] for c in result.collars} == set(TRUTH)


def test_query_vlm_collars_empty_tiles_short_circuits(monkeypatch):
    """No tiles (e.g. an empty source image) must return the empty-collars shape without ever resolving a
    vision adapter -- mirrors ``_query_vlm``'s own empty-tiles guard."""

    def _boom(vlm):
        raise AssertionError("must not resolve a vision adapter when there are no tiles")

    monkeypatch.setattr(map_digitize, "_resolve_vision_adapter", _boom)
    assert map_digitize._query_vlm_collars([], vlm=None) == {"collars": [], "traces": []}
