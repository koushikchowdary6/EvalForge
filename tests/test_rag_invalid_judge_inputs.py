"""Malformed RAG judge inputs must not crash a sweep or consume API calls."""

import math

import pytest

from evalforge.rag_graders import RAGJudge


def _judge_that_must_not_be_called():
    judge = RAGJudge.__new__(RAGJudge)

    def fail_if_called(_prompt):
        raise AssertionError("invalid evidence must not reach a paid judge")

    judge._call = fail_if_called
    return judge


@pytest.mark.parametrize("invalid_answer", [math.nan, math.inf, -math.inf, object()])
def test_correctness_rejects_non_json_or_nonfinite_evidence(invalid_answer):
    result = _judge_that_must_not_be_called().grade_correctness(
        "Question", "reference", invalid_answer
    )
    assert result == {
        "metric": "correctness",
        "score": None,
        "passed": None,
        "reason": "invalid judge evidence",
        "parse_ok": False,
    }


def test_correctness_rejects_circular_evidence():
    circular = []
    circular.append(circular)
    result = _judge_that_must_not_be_called().grade_correctness(
        circular, "reference", "answer"
    )
    assert result["parse_ok"] is False
    assert result["score"] is None


@pytest.mark.parametrize(
    "documents",
    [
        [{"doc_id": "doc1", "text": math.nan}],
        [{"doc_id": "doc1", "text": math.inf}],
        [{"doc_id": "doc1"}],
        [None],
        None,
    ],
)
def test_groundedness_rejects_invalid_document_evidence(documents):
    result = _judge_that_must_not_be_called().grade_groundedness(
        "answer", documents
    )
    assert result == {
        "metric": "groundedness",
        "score": None,
        "passed": None,
        "reason": "invalid judge evidence",
        "parse_ok": False,
    }


def test_groundedness_rejects_circular_document_value():
    circular = []
    circular.append(circular)
    result = _judge_that_must_not_be_called().grade_groundedness(
        "answer", [{"doc_id": "doc1", "text": circular}]
    )
    assert result["parse_ok"] is False
    assert result["score"] is None


@pytest.mark.parametrize("grader", ["correctness", "groundedness"])
def test_valid_judge_evidence_still_reaches_provider(grader):
    judge = RAGJudge.__new__(RAGJudge)
    prompts = []

    def fake_call(prompt):
        prompts.append(prompt)
        return '{"score": 8, "reason": "supported"}'

    judge._call = fake_call
    result = (
        judge.grade_correctness("Question", "expected", "answer")
        if grader == "correctness"
        else judge.grade_groundedness(
            "answer", [{"doc_id": "doc1", "text": "evidence"}]
        )
    )
    assert result["parse_ok"] is True
    assert result["score"] == pytest.approx(0.8)
    assert len(prompts) == 1
