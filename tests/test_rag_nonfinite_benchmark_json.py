"""Nonstandard JSON numbers must not enter RAG benchmark provenance."""

import json

import pytest

from evalforge.rag import VectorStore
from evalforge.rag_runner import RAGEvalRunner, _dataset_sha256


class StubEmbedder:
    api_calls = 0
    cache_hits = 0

    def embed(self, text):
        return [1.0, 0.0]

    def save_cache(self):
        pass


class StubStore:
    def __init__(self):
        self.documents = []
        self.built = False

    def load_corpus(self, path):
        self.documents = [{"doc_id": "doc-1"}]

    def build_index(self):
        self.built = True


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_corpus_metadata_rejected_without_losing_existing_index(tmp_path, constant):
    store = VectorStore(StubEmbedder())
    good = tmp_path / "good.json"
    good.write_text(json.dumps([
        {"doc_id": "doc-1", "title": "Evidence", "text": "Some text"}
    ]), encoding="utf-8")
    store.load_corpus(good)
    store.build_index()
    old_docs, old_vectors = store.documents, store.vectors

    bad = tmp_path / "bad.json"
    bad.write_text('[{"doc_id":"doc-2","title":"Evidence","text":"Some text",'
                   f'"metadata":{{"score":{constant}}}}}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-standard JSON numeric constant"):
        store.load_corpus(bad)

    assert store.documents is old_docs
    assert store.vectors is old_vectors
    assert [doc["doc_id"] for doc in store.search("query")] == ["doc-1"]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_question_metadata_rejected_before_embedding(tmp_path, constant):
    payload = {
        "id": "q-1", "question": "Where is evidence?",
        "expected_answer": "Evidence", "answerable": True,
        "relevant_doc_ids": ["doc-1"],
    }
    questions = tmp_path / "questions.json"
    encoded = json.dumps([payload])
    questions.write_text(encoded[:-2] + f',"metadata":{{"score":{constant}}}}}]',
                         encoding="utf-8")
    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.store = StubStore()
    runner.corpus_path = tmp_path / "corpus.json"
    runner.questions_path = questions
    runner.questions = [payload]
    runner.results = [{"case_id": "old"}]
    runner.report = {"stale": True}

    with pytest.raises(ValueError, match="Non-standard JSON numeric constant"):
        runner.load()
    assert runner.store.built is False
    assert runner.questions == []
    assert runner.results == []
    assert runner.report is None
    assert runner.corpus_sha256 is None
    assert runner.questions_sha256 is None


def test_provenance_hash_rejects_nonfinite_python_values():
    with pytest.raises(ValueError, match="Out of range float"):
        _dataset_sha256({"metadata": {"score": float("nan")}})
    assert len(_dataset_sha256({"metadata": {"score": 0.5}})) == 64
