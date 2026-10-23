"""A failed embedding request must never expose a partial RAG index."""

import json

import pytest

from evalforge.rag import VectorStore


class FlakyEmbedder:
    api_calls = 0
    cache_hits = 0

    def __init__(self):
        self.calls = 0
        self.fail_on = None

    def embed(self, text):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("embedding service unavailable")
        return [1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0]

    def save_cache(self):
        pass


def _store(tmp_path, embedder):
    corpus = [
        {"doc_id": "a", "title": "alpha", "text": "first"},
        {"doc_id": "b", "title": "beta", "text": "second"},
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    store = VectorStore(embedder)
    store.load_corpus(path)
    return store


def test_failed_initial_build_exposes_no_partial_index(tmp_path):
    embedder = FlakyEmbedder()
    embedder.fail_on = 2
    store = _store(tmp_path, embedder)

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        store.build_index()

    assert store.vectors == {}
    with pytest.raises(RuntimeError, match="Index is empty"):
        store.search("alpha")

    embedder.fail_on = None
    store.build_index()
    assert set(store.vectors) == {"a", "b"}
    assert store.search("alpha", top_k=1)[0]["doc_id"] == "a"


def test_failed_rebuild_cannot_serve_old_or_partial_vectors(tmp_path):
    embedder = FlakyEmbedder()
    store = _store(tmp_path, embedder)
    store.build_index()
    assert len(store.vectors) == 2

    embedder.fail_on = embedder.calls + 2
    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        store.build_index()

    assert store.vectors == {}
    with pytest.raises(RuntimeError, match="Index is empty"):
        store.search("beta")
