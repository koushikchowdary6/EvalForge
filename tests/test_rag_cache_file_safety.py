"""Untrusted embedding cache files must not block, escape, or exhaust evaluation."""

import json
import os

import pytest

from evalforge import rag


def _load(path):
    client = object.__new__(rag.EmbeddingClient)
    client.cache_path = str(path)
    return client._load_cache()


def test_regular_cache_still_loads(tmp_path):
    key = rag._embedding_cache_key("query")
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({key: [1.0, 0.0]}), encoding="utf-8")
    assert _load(path) == {key: [1.0, 0.0]}


def test_symlink_cache_is_not_followed(tmp_path):
    key = rag._embedding_cache_key("private query")
    target = tmp_path / "outside.json"
    target.write_text(json.dumps({key: [1.0, 0.0]}), encoding="utf-8")
    link = tmp_path / "cache.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert _load(link) == {}
    assert json.loads(target.read_text(encoding="utf-8"))[key] == [1.0, 0.0]


def test_oversized_cache_is_skipped_before_json_parsing(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    with path.open("wb") as handle:
        handle.truncate(rag.MAX_EMBED_CACHE_BYTES + 1)
    def should_not_parse(*args, **kwargs):
        pytest.fail("oversized cache must not reach JSON parser")
    monkeypatch.setattr(rag.json, "loads", should_not_parse)
    assert _load(path) == {}


@pytest.mark.skipif(os.name != "posix", reason="FIFOs require POSIX")
def test_fifo_cache_is_not_opened(tmp_path):
    path = tmp_path / "cache.json"
    os.mkfifo(path)
    assert _load(path) == {}


def test_malformed_utf8_and_deeply_nested_cache_are_ignored(tmp_path):
    path = tmp_path / "cache.json"
    path.write_bytes(b"\\xff")
    assert _load(path) == {}
    path.write_text("[" * 3000 + "0" + "]" * 3000, encoding="utf-8")
    assert _load(path) == {}
