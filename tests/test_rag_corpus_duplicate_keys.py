"""Ambiguous JSON corpus fields must not silently change retrieval evidence."""

import pytest

from evalforge.rag import VectorStore


class StubEmbedder:
    api_calls = 0
    cache_hits = 0

    def embed(self, text):
        return [1.0, 0.0]

    def save_cache(self):
        pass


@pytest.mark.parametrize(
    "payload",
    [
        '[{"doc_id":"safe","doc_id":"other","title":"a","text":"b"}]',
        '[{"doc_id":"safe","title":"a","title":"other","text":"b"}]',
        '[{"doc_id":"safe","title":"a","text":"b","text":"overridden"}]',
        '[{"doc_id":"safe","title":"a","text":"b","metadata":{"source":"x","source":"y"}}]',
    ],
)
def test_duplicate_json_keys_are_rejected_at_any_depth(tmp_path, payload):
    path = tmp_path / "corpus.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        VectorStore(StubEmbedder()).load_corpus(path)


def test_invalid_reload_preserves_previous_corpus_and_index(tmp_path):
    store = VectorStore(StubEmbedder())
    valid = tmp_path / "valid.json"
    valid.write_text(
        '[{"doc_id":"old","title":"alpha","text":"original evidence"}]',
        encoding="utf-8",
    )
    store.load_corpus(valid)
    store.build_index()
    before = store.search("alpha")

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        '[{"doc_id":"new","title":"x","text":"x","text":"spoofed"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        store.load_corpus(invalid)

    assert [item["doc_id"] for item in store.documents] == ["old"]
    assert set(store.vectors) == {"old"}
    assert store.search("alpha") == before


def test_valid_corpus_still_loads_with_extra_metadata(tmp_path):
    path = tmp_path / "valid.json"
    path.write_text(
        '[{"doc_id":"a","title":"alpha","text":"evidence",'
        '"metadata":{"source":"published","year":2026}}]',
        encoding="utf-8",
    )
    store = VectorStore(StubEmbedder())
    assert store.load_corpus(path)[0]["metadata"]["source"] == "published"
