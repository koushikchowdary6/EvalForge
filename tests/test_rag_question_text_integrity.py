"""Reject unusable RAG benchmark text before embeddings or judging."""

import json

import pytest

from evalforge.rag_runner import RAGEvalRunner, validate_rag_questions


def _case():
    return {
        "id": "q-1",
        "question": "What does the evidence say?",
        "expected_answer": "Evidence says yes.",
        "answerable": True,
        "relevant_doc_ids": ["doc-1"],
    }


@pytest.mark.parametrize("field", ["question", "expected_answer"])
@pytest.mark.parametrize("invalid", [
    "\\u200b",  # zero-width space passes ordinary str.strip()
    "\\u0301",  # standalone combining accent is not a visible answer
    "\\ud800",  # escaped lone surrogates are legal JSON but invalid UTF-8
    "text\\udfff",  # reject malformed Unicode even with visible content
    " \\u200d\\t",
])
def test_invalid_benchmark_text_is_rejected(field, invalid):
    case = _case()
    case[field] = invalid
    with pytest.raises(ValueError, match=field):
        validate_rag_questions([case], {"doc-1"})


def test_visible_unicode_and_embedded_controls_are_preserved():
    case = _case()
    case["question"] = "Café\\u200b facts?"
    case["expected_answer"] = "Résumé\\u0301 evidence"
    validate_rag_questions([case], {"doc-1"})


def test_invalid_text_prevents_index_build_and_paid_embedding(tmp_path):
    path = tmp_path / "questions.json"
    case = _case()
    case["question"] = "\\u200b"
    path.write_text(json.dumps([case]), encoding="utf-8")

    class StubStore:
        def __init__(self):
            self.documents = []
            self.built = False

        def load_corpus(self, _path):
            self.documents = [{"doc_id": "doc-1"}]

        def build_index(self):
            self.built = True
            raise AssertionError("invalid benchmark must not build an index")

    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.store = StubStore()
    runner.corpus_path = tmp_path / "corpus.json"
    runner.questions_path = path
    runner.questions = [_case()]

    with pytest.raises(ValueError, match="question"):
        runner.load()
    assert runner.store.built is False
    assert runner.questions == []
