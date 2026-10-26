"""Duplicate JSON cache keys must never silently select a conflicting vector."""

import json

from evalforge import rag


def test_duplicate_embedding_cache_keys_fail_closed(tmp_path):
    key = rag._embedding_cache_key("private question")
    path = tmp_path / "embedding_cache.json"
    # A normal json.load silently uses the second vector for the same key.
    path.write_text(
        "{" + json.dumps(key) + ": [1.0, 0.0], "
        + json.dumps(key) + ": [0.0, 1.0]}",
        encoding="utf-8",
    )
    client = object.__new__(rag.EmbeddingClient)
    client.cache_path = str(path)

    assert client._load_cache() == {}
    # Reading a corrupt cache must not rewrite or partially trust its data.
    assert path.read_text(encoding="utf-8").count(key) == 2


def test_distinct_embedding_cache_keys_remain_loadable(tmp_path):
    first = rag._embedding_cache_key("first")
    second = rag._embedding_cache_key("second")
    path = tmp_path / "embedding_cache.json"
    path.write_text(
        json.dumps({first: [1.0, 0.0], second: [0.0, 1.0]}),
        encoding="utf-8",
    )
    client = object.__new__(rag.EmbeddingClient)
    client.cache_path = str(path)

    assert client._load_cache() == {first: [1.0, 0.0], second: [0.0, 1.0]}
