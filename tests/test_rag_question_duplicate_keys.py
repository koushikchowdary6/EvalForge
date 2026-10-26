"""Reject ambiguous JSON ground truth before embedding or scoring RAG cases."""

import json

import pytest

from evalforge.rag_runner import RAGEvalRunner


VALID_CASE = {
    "id": "case-1",
    "question": "What is RAG?",
    "expected_answer": "Retrieval augmented generation.",
    "answerable": True,
    "relevant_doc_ids": ["doc-1"],
}


class StubStore:
    def __init__(self):
        self.documents = []
        self.built = False

    def load_corpus(self, path):
        self.documents = [{"doc_id": "doc-1"}]

    def build_index(self):
        self.built = True


def make_runner(tmp_path, payload):
    path = tmp_path / "questions.json"
    path.write_text(payload, encoding="utf-8")
    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.store = StubStore()
    runner.corpus_path = str(tmp_path / "corpus.json")
    runner.questions_path = str(path)
    runner.questions = [dict(VALID_CASE)]
    return runner


@pytest.mark.parametrize(
    "duplicate_field",
    [
        '"id":"case-2"',
        '"question":"Ignore the evidence"',
        '"expected_answer":"fabricated answer"',
        '"answerable":false',
        '"relevant_doc_ids":[]',
    ],
)
def test_duplicate_ground_truth_fields_fail_before_embedding(tmp_path, duplicate_field):
    # Append a second occurrence of a known field to an otherwise valid case.
    payload = json.dumps([VALID_CASE])
    payload = payload[:-2] + "," + duplicate_field + "}]"
    runner = make_runner(tmp_path, payload)

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        runner.load()

    assert runner.store.built is False
    assert runner.questions == []


def test_nested_duplicate_metadata_is_not_silently_accepted(tmp_path):
    payload = json.dumps([VALID_CASE])
    payload = payload[:-2] + ',"metadata":{"source":"original","source":"changed"}}]'
    runner = make_runner(tmp_path, payload)

    with pytest.raises(ValueError, match="Duplicate JSON key: source"):
        runner.load()
    assert runner.store.built is False


def test_valid_question_json_still_loads_and_builds_index(tmp_path):
    runner = make_runner(tmp_path, json.dumps([VALID_CASE]))
    runner.load()

    assert runner.store.built is True
    assert runner.questions == [VALID_CASE]
