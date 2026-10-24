"""Offline regression tests: invalid embeddings must never enter RAG rankings."""

import json
import math

import pytest

from evalforge.rag import VectorStore, cosine_similarity


class ControlledEmbedder:
    api_calls = 0
    cache_hits = 0

    def __init__(self, document_vectors, query_vector=None):
        self.document_vectors = document_vectors
        self.query_vector = query_vector or [1.0, 0.0]

    def embed(self, text):
        if text.startswith("alpha\n"):
            return self.document_vectors[0]
        if text.startswith("beta\n"):
            return self.document_vectors[1]
        return self.query_vector

    def save_cache(self):
        pass


def make_store(tmp_path, embedder):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([
        {"doc_id": "a", "title": "alpha", "text": "first"},
        {"doc_id": "b", "title": "beta", "text": "second"},
    ]), encoding="utf-8")
    store = VectorStore(embedder)
    store.load_corpus(path)
    return store


@pytest.mark.parametrize("invalid", [
    [float("nan"), 0.0],
    [float("inf"), 0.0],
    [1.0, float("-inf")],
    [0.0, 0.0],
    [1.0],
    ["1.0", 0.0],
    [True, 0.0],
    [],
])
def test_bad_document_vector_aborts_atomic_index(tmp_path, invalid):
    embedder = ControlledEmbedder([[1.0, 0.0], invalid])
    store = make_store(tmp_path, embedder)

    with pytest.raises(ValueError, match="document b: embedding"):
        store.build_index()

    assert store.vectors == {}
    with pytest.raises(RuntimeError, match="Index is empty"):
        store.search("question")


@pytest.mark.parametrize("invalid", [
    [float("nan"), 0.0],
    [float("inf"), 0.0],
    [0.0, 0.0],
    [1.0],
    ["1.0", 0.0],
])
def test_bad_query_vector_rejected_before_ranking(tmp_path, invalid):
    embedder = ControlledEmbedder([[1.0, 0.0], [0.0, 1.0]], query_vector=invalid)
    # An empty list is deliberately excluded here: it is tested as a document
    # vector above, while query tests cover non-finite and dimension errors.
    store = make_store(tmp_path, embedder)
    store.build_index()

    with pytest.raises(ValueError, match="query: embedding"):
        store.search("question")
    assert set(store.vectors) == {"a", "b"}


def test_valid_vectors_still_retrieve_in_similarity_order(tmp_path):
    embedder = ControlledEmbedder([[1.0, 0.0], [0.0, 1.0]])
    store = make_store(tmp_path, embedder)
    store.build_index()

    assert [doc["doc_id"] for doc in store.search("question", top_k=2)] == ["a", "b"]


@pytest.mark.parametrize(("a", "b", "expected"), [
    ([1e308, 1e308], [1e308, 1e308], 1.0),
    ([1e308, 1e308], [-1e308, -1e308], -1.0),
    ([1e308, 0.0], [0.0, 1e308], 0.0),
    ([1e-300, 1e-300], [1e-300, 1e-300], 1.0),
    ([1e-300, 1e-300], [-1e-300, -1e-300], -1.0),
    ([0.0, 0.0], [1e308, 0.0], 0.0),
])
def test_cosine_similarity_is_stable_at_extreme_finite_magnitudes(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-12)


def test_extreme_finite_embeddings_preserve_retrieval_order(tmp_path):
    embedder = ControlledEmbedder(
        [[1e308, 1e308], [-1e308, -1e308]],
        query_vector=[1e308, 1e308],
    )
    store = make_store(tmp_path, embedder)
    store.build_index()

    results = store.search("question", top_k=2)
    assert [doc["doc_id"] for doc in results] == ["a", "b"]
    assert all(math.isfinite(doc["score"]) for doc in results)
    assert [doc["score"] for doc in results] == pytest.approx([1.0, -1.0])
