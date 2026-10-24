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
    key = rag._embedding_cache_key("hello")
    client, calls = _client(tmp_path, cache={key: [1.0, 0.0]})
    assert client.embed("hello") == [1.0, 0.0]
    assert client.cache_hits == 1
    assert client.api_calls == 0
    assert calls == []


@pytest.mark.parametrize("bad", [[float("nan"), 0.0], [0.0, 0.0], "bad", []])
def test_invalid_cached_vector_is_replaced_by_fresh_embedding(tmp_path, bad):
    key = rag._embedding_cache_key("hello")
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


def test_cache_load_discards_corrupt_entries_before_saving(tmp_path):
    """Unrelated corrupt entries must not poison a valid cache on disk."""
    path = tmp_path / "embedding_cache.json"
    good_key = rag._embedding_cache_key("good")
    bad_key = rag._embedding_cache_key("bad")
    huge_key = rag._embedding_cache_key("huge")
    zero_key = rag._embedding_cache_key("zero")
    path.write_text(
        json.dumps({
            good_key: [1.0, 0.0],
            bad_key: [float("nan"), 1.0],
            huge_key: [10 ** 400, 1],
            zero_key: [0, 0],
        }),
        encoding="utf-8",
    )
    client, _ = _client(tmp_path)
    client.cache = rag.EmbeddingClient._load_cache(client)
    assert client.cache == {good_key: [1.0, 0.0]}
    client.save_cache()
    assert json.loads(path.read_text(encoding="utf-8")) == client.cache


def test_cache_load_rejects_wrong_type_and_empty_vectors(tmp_path):
    path = tmp_path / "embedding_cache.json"
    valid = rag._embedding_cache_key("valid")
    invalid = rag._embedding_cache_key("invalid")
    path.write_text(
        json.dumps({valid: [1.0, 2.0], invalid: ["not numeric", 2.0]}),
        encoding="utf-8",
    )
    client, _ = _client(tmp_path)
    client.cache = rag.EmbeddingClient._load_cache(client)
    assert client.cache == {valid: [1.0, 2.0]}


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


def test_cache_key_does_not_persist_raw_document_text(tmp_path):
    secret_document = "Confidential case: patient alice@example.com"
    client, calls = _client(tmp_path, api_vector=[1.0, 0.0])
    assert client.embed(secret_document) == [1.0, 0.0]
    assert len(calls) == 1
    client.save_cache()

    on_disk = (tmp_path / "embedding_cache.json").read_text(encoding="utf-8")
    assert secret_document not in on_disk
    assert "alice@example.com" not in on_disk
    assert list(json.loads(on_disk)) == [rag._embedding_cache_key(secret_document)]

    reloaded = rag.EmbeddingClient._load_cache(client)
    assert reloaded == client.cache


def test_legacy_plaintext_cache_keys_are_ignored_and_not_resaved(tmp_path):
    path = tmp_path / "embedding_cache.json"
    old_key = f"{rag.EMBED_MODEL}:sensitive customer note"
    good_key = rag._embedding_cache_key("safe note")
    path.write_text(
        json.dumps({old_key: [1.0, 0.0], good_key: [0.0, 1.0]}),
        encoding="utf-8",
    )
    client, _ = _client(tmp_path)
    client.cache = rag.EmbeddingClient._load_cache(client)
    assert client.cache == {good_key: [0.0, 1.0]}
    client.save_cache()
    assert "sensitive customer note" not in path.read_text(encoding="utf-8")


def test_embedding_cache_keys_are_model_scoped(monkeypatch):
    first = rag._embedding_cache_key("same input")
    monkeypatch.setattr(rag, "EMBED_MODEL", "another-embedding-model")
    assert rag._embedding_cache_key("same input") != first
