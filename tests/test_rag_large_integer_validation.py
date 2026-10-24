"""Corrupted cache and invalid cosine vectors must fail closed, not crash."""

from types import SimpleNamespace

import pytest

from evalforge import rag


@pytest.mark.parametrize("bad", [10**400, float("nan"), float("inf"), True, "1.0"])
def test_invalid_numeric_components_raise_clear_validation_error(bad):
    with pytest.raises(ValueError, match="embedding contains non-finite"):
        rag._validated_embedding([bad, 1.0], label="cached embedding")


@pytest.mark.parametrize("bad", [10**400, float("nan"), float("-inf"), True, "1.0"])
def test_cosine_rejects_invalid_components_instead_of_returning_a_score(bad):
    with pytest.raises(ValueError, match="finite numeric components"):
        rag.cosine_similarity([bad, 1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="finite numeric components"):
        rag.cosine_similarity([1.0, 0.0], [bad, 1.0])


def test_oversized_cached_json_integer_is_replaced_with_fresh_embedding(tmp_path):
    key = rag._embedding_cache_key("question")
    client = object.__new__(rag.EmbeddingClient)
    client.cache_path = str(tmp_path / "cache.json")
    client.cache = {key: [10**400, 1]}
    client.api_calls = 0
    client.cache_hits = 0
    calls = []

    def create(*, model, input):
        calls.append((model, input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0, 1.0])])

    client.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    assert client.embed("question") == [0.0, 1.0]
    assert client.cache[key] == [0.0, 1.0]
    assert client.cache_hits == 0
    assert client.api_calls == 1
    assert len(calls) == 1


def test_oversized_api_integer_is_rejected_without_cache_poisoning(tmp_path):
    client = object.__new__(rag.EmbeddingClient)
    client.cache_path = str(tmp_path / "cache.json")
    client.cache = {}
    client.api_calls = 0
    client.cache_hits = 0
    client.client = SimpleNamespace(embeddings=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(data=[
            SimpleNamespace(embedding=[10**400, 1.0])
        ])
    ))
    with pytest.raises(ValueError, match="API embedding"):
        client.embed("question")
    assert client.cache == {}
