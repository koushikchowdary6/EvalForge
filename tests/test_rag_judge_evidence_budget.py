"""Oversized RAG evidence is excluded, never truncated or sent to a paid judge."""

import pytest

from evalforge.rag_graders import MAX_JUDGE_EVIDENCE_CHARS, RAGJudge


@pytest.mark.parametrize("grader", ["correctness", "groundedness"])
def test_oversized_evidence_never_reaches_provider(grader):
    judge = RAGJudge.__new__(RAGJudge)

    def fail_if_called(_prompt):
        raise AssertionError("oversized untrusted evidence must not reach a provider")

    judge._call = fail_if_called
    oversized = "a" * (MAX_JUDGE_EVIDENCE_CHARS + 1)
    result = (
        judge.grade_correctness("question", "expected", oversized)
        if grader == "correctness"
        else judge.grade_groundedness("answer", [{"doc_id": "doc1", "text": oversized}])
    )
    assert result == {
        "metric": grader,
        "score": None,
        "passed": None,
        "reason": "judge evidence exceeds size limit",
        "parse_ok": False,
    }


@pytest.mark.parametrize("grader", ["correctness", "groundedness"])
def test_small_evidence_still_reaches_judge_unchanged(grader):
    judge = RAGJudge.__new__(RAGJudge)
    prompts = []

    def fake_call(prompt):
        prompts.append(prompt)
        return '{"score": 8, "reason": "supported"}'

    judge._call = fake_call
    result = (
        judge.grade_correctness("Q", "reference", "answer")
        if grader == "correctness"
        else judge.grade_groundedness("answer", [{"doc_id": "d1", "text": "evidence"}])
    )
    assert result["parse_ok"] is True
    assert result["score"] == pytest.approx(0.8)
    assert len(prompts) == 1
    assert "EVAL_INPUT_JSON:" in prompts[0]
