"""Strict JSON judge evidence cannot silently inflate RAG quality metrics."""
import pytest

from evalforge.rag_graders import RAGJudge


@pytest.mark.parametrize("raw", [
    '{"score": 10, "reason": ""}',
    '{"score": 10, "reason": " \t "}',
    '{"score": 10, "reason": "valid", "confidence": NaN}',
    '{"score": 10, "reason": "valid", "metadata": {"other": Infinity}}',
    '{"score": 10, "reason": "valid", "extra": 1}',
    '{"score": 10, "reason": "valid", "extra": null}',
    '{"score": 10, "reason": "valid", "extra": {}}',
])
def test_malformed_or_ambiguous_judge_evidence_is_not_scored(raw):
    result = RAGJudge._parse_score(raw)
    assert result["parse_ok"] is False
    assert result["score"] == 0.5


@pytest.mark.parametrize("grader", ["correctness", "groundedness"])
def test_invalid_judge_evidence_is_excluded_from_aggregates(grader, monkeypatch):
    judge = RAGJudge.__new__(RAGJudge)
    monkeypatch.setattr(
        judge, "_call",
        lambda prompt: '{"score": 10, "reason": "valid", "metadata": NaN}',
    )
    result = (
        judge.grade_correctness("Q", "reference", "answer")
        if grader == "correctness"
        else judge.grade_groundedness("answer", [{"doc_id": "d1", "text": "context"}])
    )
    assert result["parse_ok"] is False
    assert result["score"] is None
    assert result["passed"] is None


@pytest.mark.parametrize("raw, expected", [
    ('{"score": 0, "reason": "unsupported"}', 0.0),
    ('{"score": 7.5, "reason": "partly supported"}', 0.75),
    ('```json\n{"score": 10, "reason": "supported"}\n```', 1.0),
])
def test_valid_judge_contract_preserves_existing_scores(raw, expected):
    result = RAGJudge._parse_score(raw)
    assert result["parse_ok"] is True
    assert result["score"] == pytest.approx(expected)


@pytest.mark.parametrize("raw", [
    '{"score": 10, "reason": "\\u200b"}',
    '{"score": 10, "reason": "\\u0301"}',
    '{"score": 10, "reason": "ok\\u202e"}',
    '{"score": 10, "reason": "ok\\ud800"}',
])
def test_invisible_or_spoofed_judge_reasons_cannot_count_as_grades(raw):
    result = RAGJudge._parse_score(raw)
    assert result["parse_ok"] is False


@pytest.mark.parametrize("reason", ["Résumé is supported", "Cafe\\u0301 matches the context", "Supported 👍"])
def test_legitimate_unicode_judge_explanations_remain_valid(reason):
    import json

    result = RAGJudge._parse_score(json.dumps({"score": 8, "reason": reason}))
    assert result["parse_ok"] is True
    assert result["score"] == pytest.approx(0.8)
