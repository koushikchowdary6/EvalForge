"""Offline regression tests for durable, validated RAG embedding caches."""

import json
from types import SimpleNamespace

import pytest

from evalforge import rag


def _client(tmp_path, *, api_vector=None, cache=None):
    """Build a deterministic client without network access or API keys."""
    client = object.__new__(rag.EmbeddingClient)
    client.cache_path = str(tmp_path / "embedding_cache.json")
    client.cache = {} if cache is None else cache
    client.api_calls = 0
    client.cache_hits = 0
    calls = []

    def create(*, model, input):
        calls.append((model, input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=api_vector)])

    client.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    return client, calls


def test_valid_cached_vector_avoids_api(tmp_path):
    key = f"{rag.EMBED_MODEL}:hello"
    client, calls = _client(tmp_path, cache={key: [1.0, 0.0]})
    assert client.embed("hello") == [1.0, 0.0]
    assert client.cache_hits == 1
    assert client.api_calls == 0
    assert calls == []


@pytest.mark.parametrize("bad", [[float("nan"), 0.0], [0.0, 0.0], "bad", []])
def test_invalid_cached_vector_is_replaced_by_fresh_embedding(tmp_path, bad):
    key = f"{rag.EMBED_MODEL}:hello"
    client, calls = _client(tmp_path, cache={key: bad}, api_vector=[0.0, 1.0])
    assert client.embed("hello") == [0.0, 1.0]
    assert client.cache[key] == [0.0, 1.0]
    assert client.cache_hits == 0
    assert client.api_calls == 1
    assert len(calls) == 1


@pytest.mark.parametrize("bad", [[float("inf"), 0.0], [0.0, 0.0], ["1", 2]])
def test_invalid_api_vector_is_never_cached(tmp_path, bad):
    client, _ = _client(tmp_path, api_vector=bad)
    with pytest.raises(ValueError, match="API embedding"):
        client.embed("hello")
    assert client.cache == {}


def test_valid_json_non_mapping_cache_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "embedding_cache.json"
    path.write_text('["unexpected", "array"]', encoding="utf-8")
    # Constructor must not require network access.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = rag.EmbeddingClient(cache_path=str(path))
    assert client.cache == {}


def test_successful_cache_write_is_valid_json(tmp_path):
    client, _ = _client(tmp_path, cache={"test": [1.0, 2.0]})
    client.save_cache()
    assert json.loads((tmp_path / "embedding_cache.json").read_text()) == client.cache
    assert not list(tmp_path.glob(".embedding-cache-*.tmp"))


def test_failed_serialization_preserves_previous_cache(tmp_path):
    path = tmp_path / "embedding_cache.json"
    path.write_text('{"original": [1, 2]}', encoding="utf-8")
    client, _ = _client(tmp_path, cache={"bad": object()})
    with pytest.raises(TypeError):
        client.save_cache()
    assert json.loads(path.read_text()) == {"original": [1, 2]}
    assert not list(tmp_path.glob(".embedding-cache-*.tmp"))


def test_failed_replace_preserves_previous_cache(tmp_path, monkeypatch):
    path = tmp_path / "embedding_cache.json"
    path.write_text('{"original": [1, 2]}', encoding="utf-8")
    client, _ = _client(tmp_path, cache={"new": [3.0, 4.0]})

    def fail_replace(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr(rag.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        client.save_cache()
    assert json.loads(path.read_text()) == {"original": [1, 2]}
    assert not list(tmp_path.glob(".embedding-cache-*.tmp"))
