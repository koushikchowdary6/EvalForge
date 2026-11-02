"""Provider provenance must be checked before caching paid RAG embeddings."""

from types import SimpleNamespace

import pytest

from evalforge import rag


def _client(*, model=rag.EMBED_MODEL, index=0, include_model=True):
    client = object.__new__(rag.EmbeddingClient)
    client.cache = {}
    client.api_calls = 0
    client.cache_hits = 0
    entry = SimpleNamespace(index=index, embedding=[1.0, 0.0])
    response = SimpleNamespace(data=[entry])
    if include_model:
        response.model = model
    client.client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kwargs: response)
    )
    return client


@pytest.mark.parametrize("wrong_model", [
    "text-embedding-ada-002",
    "text-embedding-3-large",
    "",
    123,
])
def test_mismatched_provider_model_never_enters_model_scoped_cache(wrong_model):
    client = _client(model=wrong_model)
    with pytest.raises(ValueError, match="unexpected response model"):
        client.embed("benchmark question")
    assert client.cache == {}
    assert client.api_calls == 0
    assert client.cache_hits == 0


@pytest.mark.parametrize("invalid_index", [0.0, "0", False, None, 1])
def test_non_integer_or_wrong_provider_index_never_enters_cache(invalid_index):
    client = _client(index=invalid_index)
    with pytest.raises(ValueError, match="unexpected result index"):
        client.embed("benchmark question")
    assert client.cache == {}
    assert client.api_calls == 0


@pytest.mark.parametrize("include_model", [True, False])
def test_valid_provider_embedding_retains_cache_reuse(include_model):
    client = _client(include_model=include_model)
    assert client.embed("benchmark question") == [1.0, 0.0]
    assert client.embed("benchmark question") == [1.0, 0.0]
    assert client.api_calls == 1
    assert client.cache_hits == 1
    assert list(client.cache) == [rag._embedding_cache_key("benchmark question")]
