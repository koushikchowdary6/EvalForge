"""Embedding input boundaries protect RAG benchmark cost and index integrity."""

import json
from types import SimpleNamespace

import pytest

from evalforge.rag import (
    EmbeddingClient, MAX_EMBED_INPUT_CHARS, VectorStore,
)


class CountingEmbedder:
    api_calls = 0
    cache_hits = 0

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [1.0, 0.0]

    def save_cache(self):
        pass


def _corpus(tmp_path, name, documents):
    path = tmp_path / name
    path.write_text(json.dumps(documents), encoding="utf-8")
    return path


def _doc(doc_id, text="short evidence"):
    return {"doc_id": doc_id, "title": "Evidence", "text": text}


@pytest.mark.parametrize("invalid", [
    "", " \t\n", "\u200b\u0301", "\ud800", "a" * (MAX_EMBED_INPUT_CHARS + 1),
    None, 123,
])
def test_embedding_client_rejects_invalid_input_before_cache_or_api(tmp_path, invalid):
    client = object.__new__(EmbeddingClient)
    client.cache = {}
    client.cache_hits = 0
    client.api_calls = 0
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])

    client.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with pytest.raises(ValueError, match="embedding input"):
        client.embed(invalid)
    assert calls == []
    assert client.cache == {}
    assert client.api_calls == 0


def test_corpus_preflight_rejects_oversize_before_any_embedding(tmp_path):
    embedder = CountingEmbedder()
    store = VectorStore(embedder)
    documents = [
        _doc("first"),
        _doc("oversized", "x" * MAX_EMBED_INPUT_CHARS),
    ]
    with pytest.raises(ValueError, match="embedding input exceeds"):
        store.load_corpus(_corpus(tmp_path, "oversized.json", documents))
    assert embedder.calls == []
    assert store.documents == []
    assert store.vectors == {}


def test_oversized_reload_preserves_last_good_index(tmp_path):
    embedder = CountingEmbedder()
    store = VectorStore(embedder)
    store.load_corpus(_corpus(tmp_path, "good.json", [_doc("old")]))
    store.build_index()
    previous = store.search("valid question")
    calls_before = list(embedder.calls)

    with pytest.raises(ValueError, match="embedding input exceeds"):
        store.load_corpus(_corpus(tmp_path, "bad.json", [
            _doc("new", "x" * MAX_EMBED_INPUT_CHARS),
        ]))

    assert store.search("valid question") == previous
    assert store.documents[0]["doc_id"] == "old"
    assert embedder.calls[:len(calls_before)] == calls_before


@pytest.mark.parametrize("invalid", [
    "", "\u200b", "\udfff", "x" * (MAX_EMBED_INPUT_CHARS + 1),
])
def test_search_rejects_invalid_query_before_custom_embedder(tmp_path, invalid):
    embedder = CountingEmbedder()
    store = VectorStore(embedder)
    store.load_corpus(_corpus(tmp_path, "good.json", [_doc("one")]))
    store.build_index()
    before = len(embedder.calls)
    with pytest.raises(ValueError, match="embedding input"):
        store.search(invalid)
    assert len(embedder.calls) == before


def test_limit_and_visible_multilingual_queries_remain_supported(tmp_path):
    embedder = CountingEmbedder()
    store = VectorStore(embedder)
    # The combined title/newline/body is exactly the allowed character budget.
    document = {"doc_id": "one", "title": "E", "text": "é" * (MAX_EMBED_INPUT_CHARS - 2)}
    store.load_corpus(_corpus(tmp_path, "limit.json", [document]))
    store.build_index()
    assert store.search("नमस्ते")[0]["doc_id"] == "one"
