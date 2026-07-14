import hashlib

import numpy as np
import pytest

from mixle_mlops.multimodal.store import LocalBlobStore
from mixle_mlops.rag.modality_embeddings import (
    FIELD_SPACE,
    GRAPH_SPACE,
    IMAGE_SPACE,
    TABLE_SPACE,
    TEXT_SPACE,
    IncompatibleSpaceError,
    ModalityEmbedder,
    assert_comparable,
)


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _table_item():
    payload = b"table-payload"
    return {
        "id": "t1",
        "content_hash": _hash(payload),
        "modality": "table",
        "columns": [{"name": "depth", "dtype": "float", "unit": "m"}, {"name": "grade", "dtype": "float"}],
        "rows": [[1.0, 0.5], [2.0, 0.7]],
    }


def _graph_item():
    payload = b"graph-payload"
    return {
        "id": "g1",
        "content_hash": _hash(payload),
        "modality": "graph",
        "nodes": [{"type": "fault", "label": "F1"}, {"type": "unit", "label": "U1"}],
        "edges": [{"type": "crosscuts"}],
    }


def _field_item():
    payload = b"field-payload"
    return {
        "id": "f1",
        "content_hash": _hash(payload),
        "modality": "field",
        "schema": {"dim": "1d", "unit": "ppm"},
        "values": [1.0, 2.0, 3.0, 4.0],
    }


def test_table_and_graph_and_field_embed_in_native_spaces_with_stable_dims():
    embedder = ModalityEmbedder()
    t = embedder.embed_item(_table_item())
    g = embedder.embed_item(_graph_item())
    f = embedder.embed_item(_field_item())

    assert t.space_id == TABLE_SPACE
    assert g.space_id == GRAPH_SPACE
    assert f.space_id == FIELD_SPACE
    assert t.vector.ndim == 1 and t.vector.size > 0
    assert g.vector.shape == t.vector.shape
    assert f.vector.shape == t.vector.shape


def test_repeated_same_hash_is_cache_stable():
    embedder = ModalityEmbedder()
    item = _table_item()
    a = embedder.embed_item(item)
    b = embedder.embed_item(dict(item))
    assert a.vector is b.vector
    assert np.array_equal(a.vector, b.vector)


def test_image_embedding_via_resolver_and_hash_sensitivity():
    import tempfile

    store = LocalBlobStore(root=tempfile.mkdtemp())
    rec1 = store.put(b"\x89PNGimg-bytes-one", filename="a.png", content_type="image/png")
    rec2 = store.put(b"\x89PNGimg-bytes-two-different", filename="b.png", content_type="image/png")

    embedder = ModalityEmbedder()
    item1 = {"id": "i1", "content_hash": _hash(b"\x89PNGimg-bytes-one"), "modality": "image", "ref": rec1.id}
    item2 = {"id": "i2", "content_hash": _hash(b"\x89PNGimg-bytes-two-different"), "modality": "image", "ref": rec2.id}

    e1 = embedder.embed_item(item1, resolver=store)
    e2 = embedder.embed_item(item2, resolver=store)

    assert e1.space_id == IMAGE_SPACE
    assert e1.content_hash != e2.content_hash
    assert not np.array_equal(e1.vector, e2.vector)


def test_embedding_never_mutates_item_payload_or_ref():
    item = _table_item()
    before = dict(item)
    ModalityEmbedder().embed_item(item)
    assert item == before


def test_raw_vectors_from_different_spaces_are_rejected_by_direct_comparison_guard():
    embedder = ModalityEmbedder()
    t = embedder.embed_item(_table_item())
    g = embedder.embed_item(_graph_item())
    with pytest.raises(IncompatibleSpaceError):
        assert_comparable(t, g)
    assert_comparable(t, t)  # same space: no error


def test_embed_query_returns_native_space():
    embedder = ModalityEmbedder()
    space_id, vec = embedder.embed_query("some free text", modality="text")
    assert space_id == TEXT_SPACE
    assert vec.ndim == 1
