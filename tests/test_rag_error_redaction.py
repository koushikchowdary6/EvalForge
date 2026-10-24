"""Provider errors and malformed judge payloads must not leak into saved RAG results."""

from types import SimpleNamespace

import pytest

from evalforge.rag import RAGPipeline
from evalforge.rag_graders import RAGJudge


SECRET = "sk-test-sensitive-key-DO-NOT-PERSIST"


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_generation_provider_exception_is_redacted(provider):
    def fail(**_kwargs):
        raise RuntimeError(f"upstream request Authorization: Bearer {SECRET}")

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model = provider
    pipeline.anthropic_client = SimpleNamespace(messages=SimpleNamespace(create=fail))
    pipeline.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )

    result = pipeline.generate("question", [])
    assert result["answer"] == ""
    assert result["error"] == "generation failed (RuntimeError)"
    assert SECRET not in repr(result)


@pytest.mark.parametrize("grader", ["correctness", "groundedness"])
def test_judge_provider_exception_is_redacted(monkeypatch, grader):
    judge = RAGJudge.__new__(RAGJudge)

    def fail(_prompt):
        raise ValueError(f"provider returned private context {SECRET}")

    monkeypatch.setattr(judge, "_call", fail)
    result = (
        judge.grade_correctness("question", "expected", "actual")
        if grader == "correctness"
        else judge.grade_groundedness("actual", [{"doc_id": "d", "text": "evidence"}])
    )
    assert result["score"] is None
    assert result["parse_ok"] is False
    assert "judge error (ValueError)" == result["reason"]
    assert SECRET not in repr(result)


@pytest.mark.parametrize("raw", [
    f"Not JSON. Private provider trace: {SECRET}",
    f'```json\n{{"score": 10, "reason": "{SECRET}"}}',
    f'{{"score": 10, "score": 0, "reason": "{SECRET}"}}',
])
def test_invalid_judge_payload_is_not_repeated_in_diagnostics(raw):
    result = RAGJudge._parse_score(raw)
    assert result["parse_ok"] is False
    assert result["reason"] == "invalid judge response"
    assert SECRET not in repr(result)
