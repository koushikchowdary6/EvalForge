"""Ground-truth validation must fail before any paid embedding/index build."""

import json
from types import SimpleNamespace

import pytest

from evalforge.rag_runner import RAGEvalRunner, validate_rag_questions


def valid_case():
    return {
        "id": "case-1",
        "question": "What is RAG?",
        "expected_answer": "Retrieval augmented generation.",
        "answerable": True,
        "relevant_doc_ids": ["doc-1"],
    }


@pytest.mark.parametrize("mutate, error", [
    (lambda cases: cases.append(dict(cases[0])), "Duplicate RAG question id"),
    (lambda cases: cases[0].update(answerable="false"), "answerable"),
    (lambda cases: cases[0].update(relevant_doc_ids=["missing"]), "unknown document IDs"),
    (lambda cases: cases[0].update(relevant_doc_ids=["doc-1", "doc-1"]), "duplicate relevant"),
    (lambda cases: cases[0].update(expected_answer=""), "expected_answer"),
    (lambda cases: cases[0].update(question=None), "question must"),
    (lambda cases: cases[0].pop("relevant_doc_ids"), "relevant_doc_ids"),
])
def test_invalid_ground_truth_is_rejected(mutate, error):
    cases = [valid_case()]
    mutate(cases)
    with pytest.raises(ValueError, match=error):
        validate_rag_questions(cases, {"doc-1"})


def test_empty_or_nonlist_dataset_is_rejected():
    for questions in ([], {}, None):
        with pytest.raises(ValueError, match="non-empty list"):
            validate_rag_questions(questions, {"doc-1"})


def test_valid_ground_truth_is_accepted():
    validate_rag_questions([valid_case()], {"doc-1"})


def test_load_rejects_invalid_ground_truth_without_building_index(tmp_path):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([dict(valid_case(), relevant_doc_ids=["typo"])]))

    class StubStore:
        def __init__(self):
            self.documents = []
            self.built = False

        def load_corpus(self, path):
            self.documents = [{"doc_id": "doc-1"}]

        def build_index(self):
            self.built = True

    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.store = StubStore()
    runner.corpus_path = str(tmp_path / "corpus.json")
    runner.questions_path = str(questions_path)
    runner.questions = [valid_case()]  # stale state from an earlier load

    with pytest.raises(ValueError, match="unknown document IDs"):
        runner.load()
    assert runner.store.built is False
    assert runner.questions == []


@pytest.mark.parametrize("k_values", [
    (), (0,), (-1,), (1, 1), (True,), (1.5,), ("3",),
])
def test_invalid_k_sweep_fails_before_creating_pipeline(k_values, monkeypatch):
    import evalforge.rag_runner as module

    def forbidden_pipeline(*args, **kwargs):
        raise AssertionError("invalid top-k must not start a pipeline")

    monkeypatch.setattr(module, "RAGPipeline", forbidden_pipeline)
    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.store = object()
    runner.model = "claude"
    runner.questions = [valid_case()]
    runner.results = [{"stale": True}]
    runner.report = {"stale": True}
    with pytest.raises(ValueError, match="k_values"):
        runner.run(k_values=k_values)
    assert runner.results == []
    assert runner.report is None
