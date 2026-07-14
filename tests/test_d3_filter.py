"""D3 — geoscience metadata filters + hybrid dense/lexical search (``retrieve(..., filters=..., hybrid=...)``).

Covers the Definition of Done: index 4 chunks carrying geoscience ``meta`` (lon/lat/formation/depth/element/date);
``filters={"bbox": ..., "formation": ...}`` returns exactly the matching subset; a hybrid query beats dense-only
on a lexical-exact-match fixture; every winning structured hit (``artifact_ref``/``selector``/hash/access) rides
through filtering and re-ranking unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

import mixle_mlops.rag.vectorstore as vs_mod
import mixle_mlops.storage.db as db
from mixle_mlops.config import get_settings
from mixle_mlops.rag.embeddings import Bm25Index, Embedder
from mixle_mlops.rag.index import retrieve
from mixle_mlops.rag.vectorstore import LocalVectorStore


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    vs_mod.reset_vector_store()
    yield tmp_path
    get_settings.cache_clear()
    db._engine = None
    vs_mod.reset_vector_store()


# (text, lon, lat, formation, depth, element, date)
CHUNKS = [
    ("Sandstone porosity logs recorded near the Eagle Ford well pad.",
     10.0, 20.0, "Eagle Ford", 1500.0, "Si", "2023-11-01"),
    ("Core sample analysis for the Eagle Ford shale interval, high porosity zone.",
     10.5, 20.5, "Eagle Ford", 1620.0, "Si", "2024-04-15"),
    ("Bakken formation seismic survey results across the whole basin.",
     50.0, 60.0, "Bakken", 3100.0, "Fe", "2024-01-10"),
    ("Bakken shale porosity data collected onsite near the rig.",
     10.2, 20.2, "Bakken", 1550.0, "Fe", "2024-02-20"),
]


def _index_chunks(store: LocalVectorStore, emb: Embedder, user_id: str = "geo-user") -> None:
    texts = [c[0] for c in CHUNKS]
    vectors = emb.embed(texts)
    items = []
    for i, (text, lon, lat, formation, depth, element, chunk_date) in enumerate(CHUNKS):
        items.append({
            "text": text,
            "vector": vectors[i],
            "namespace": "document",
            "source_id": "well-report-1",
            "meta": {
                "lon": lon, "lat": lat, "formation": formation, "depth": depth,
                "element": element, "date": chunk_date,
                # structured-source metadata (IC-13-shaped) that must ride through untouched
                "artifact_ref": f"blob-{i}",
                "selector": {"row": i},
                "hash": f"sha256:{i:064x}",
                "access": {"owner": user_id, "visibility": "private"},
            },
        })
    store.add(user_id, items)


def test_bbox_and_formation_filters_return_exact_subset(local_env):
    store = LocalVectorStore()
    emb = Embedder(allow_remote=False)
    _index_chunks(store, emb)

    # box around the 3 points near (10, 20); excludes the Bakken point at (50, 60)
    bbox = (9.5, 19.5, 10.7, 20.7)
    hits = retrieve(
        "geo-user", "porosity", k=10, embedder=emb, store=store,
        filters={"bbox": bbox, "formation": "Eagle Ford"},
    )
    assert {h["text"] for h in hits} == {CHUNKS[0][0], CHUNKS[1][0]}

    # bbox alone (no formation) also keeps the in-box Bakken chunk
    hits_bbox_only = retrieve(
        "geo-user", "porosity", k=10, embedder=emb, store=store, filters={"bbox": bbox},
    )
    assert {h["text"] for h in hits_bbox_only} == {CHUNKS[0][0], CHUNKS[1][0], CHUNKS[3][0]}

    # structured fields survive filtering verbatim
    by_text = {c[0]: i for i, c in enumerate(CHUNKS)}
    for h in hits:
        idx = by_text[h["text"]]
        assert h["meta"]["artifact_ref"] == f"blob-{idx}"
        assert h["meta"]["selector"] == {"row": idx}
        assert h["meta"]["hash"] == f"sha256:{idx:064x}"
        assert h["meta"]["access"] == {"owner": "geo-user", "visibility": "private"}


def test_depth_element_and_date_filters(local_env):
    store = LocalVectorStore()
    emb = Embedder(allow_remote=False)
    _index_chunks(store, emb)

    hits = retrieve("geo-user", "formation", k=10, embedder=emb, store=store,
                     filters={"depth": (1500.0, 1600.0), "element": "Si"})
    assert {h["text"] for h in hits} == {CHUNKS[0][0]}

    hits = retrieve("geo-user", "formation", k=10, embedder=emb, store=store,
                     filters={"date": ("2024-02-01", "2024-04-30")})
    assert {h["text"] for h in hits} == {CHUNKS[1][0], CHUNKS[3][0]}


def test_unknown_filter_key_raises(local_env):
    store = LocalVectorStore()
    emb = Embedder(allow_remote=False)
    _index_chunks(store, emb)
    with pytest.raises(ValueError):
        retrieve("geo-user", "porosity", embedder=emb, store=store, filters={"nonsense": 1})


class _FakeEmbedder:
    """Hand-picked 2-D vectors so dense-only ranking is unambiguously fooled by generic-word overlap, and only
    the BM25 half of hybrid search recovers the exact rare-term match — isolating the DoD's ranking claim from
    the real hashing embedder's incidental behaviour."""

    dim = 2
    _VECTORS = {
        "the equipment performance report shows a normal outcome": np.array([1.0, 0.0]),
        "quartzite metamorphic banding observed in thin-section brx-77": np.array([0.0, 1.0]),
    }

    def embed_one(self, text: str) -> np.ndarray:
        key = text.strip().lower()
        if key in self._VECTORS:
            return self._VECTORS[key]
        return np.array([0.99, 0.14])  # the query probe: dense-close to the "common" document


def test_hybrid_beats_dense_only_on_lexical_exact_match(local_env):
    store = LocalVectorStore()
    fake = _FakeEmbedder()
    common_text = "the equipment performance report shows a normal outcome"
    exact_text = "quartzite metamorphic banding observed in thin-section BRX-77"
    store.add("hy-user", [
        {"text": common_text, "vector": fake._VECTORS[common_text.lower()],
         "namespace": "document", "source_id": "s1",
         "meta": {"artifact_ref": "blob-common", "selector": {"row": 0}, "hash": "sha256:aaa"}},
        {"text": exact_text, "vector": fake._VECTORS[exact_text.lower()],
         "namespace": "document", "source_id": "s1",
         "meta": {"artifact_ref": "blob-exact", "selector": {"row": 1}, "hash": "sha256:bbb"}},
    ])
    query = "thin-section BRX-77 quartzite banding"

    dense_hits = retrieve("hy-user", query, k=2, embedder=fake, store=store, hybrid=False)
    hybrid_hits = retrieve("hy-user", query, k=2, embedder=fake, store=store, hybrid=True)

    assert dense_hits[0]["text"] == common_text     # dense-only is fooled by the probe's generic-word closeness
    assert hybrid_hits[0]["text"] == exact_text      # hybrid's lexical half recovers the exact rare-term match

    winner = hybrid_hits[0]
    assert winner["meta"]["artifact_ref"] == "blob-exact"
    assert winner["meta"]["selector"] == {"row": 1}
    assert winner["meta"]["hash"] == "sha256:bbb"


def test_bm25_index_scores_lexical_overlap_higher():
    bm25 = Bm25Index()
    texts = [
        "porosity of sandstone core samples",
        "an unrelated sentence about baking sourdough bread",
        "porosity porosity porosity",
    ]
    bm25.fit(texts)
    scores = bm25.score("porosity core samples")
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]
    assert bm25.score("")[0] == 0.0
    assert Bm25Index().fit([]).score("anything").shape == (0,)
