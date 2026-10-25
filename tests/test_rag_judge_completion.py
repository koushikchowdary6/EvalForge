"""Regression coverage for incomplete LLM-as-judge API responses (no API keys)."""

from types import SimpleNamespace

import pytest

from evalforge.rag_graders import RAGJudge


@pytest.mark.parametrize(
    ("provider", "stop_reason"),
    [
        ("claude", "max_tokens"),
        ("claude", "tool_use"),
        ("claude", "stop_sequence"),
        ("claude", None),
        ("gpt", "length"),
        ("gpt", "content_filter"),
    ],
)
@pytest.mark.parametrize("grader", ["correctness", "groundedness"])
def test_incomplete_judge_responses_never_count_as_scores(provider, stop_reason, grader):
    """Even well-formed partial JSON must not inflate evaluation scores."""
    judge = RAGJudge.__new__(RAGJudge)
    judge.judge_model = provider
    plausible_json = '{"score": 10, "reason": "looks complete"}'
    judge.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(
                stop_reason=stop_reason,
                content=[SimpleNamespace(text=plausible_json)],
            )
        )
    )
    judge.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(
                        finish_reason=stop_reason,
                        message=SimpleNamespace(content=plausible_json),
                    )]
                )
            )
        )
    )

    result = (
        judge.grade_correctness("Question", "expected", "actual")
        if grader == "correctness"
        else judge.grade_groundedness("actual", [{"doc_id": "d1", "text": "Evidence"}])
    )
    assert result["parse_ok"] is False
    assert result["score"] is None
    assert result["passed"] is None
    assert "judge error" in result["reason"]


@pytest.mark.parametrize(
    ("provider", "stop_reason"),
    [("claude", "end_turn"), ("gpt", "stop")],
)
def test_completed_judge_response_remains_scoreable(provider, stop_reason):
    judge = RAGJudge.__new__(RAGJudge)
    judge.judge_model = provider
    response = '{"score": 8, "reason": "supported"}'
    judge.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(
                stop_reason=stop_reason,
                content=[SimpleNamespace(text=response)],
            )
        )
    )
    judge.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(
                        finish_reason=stop_reason,
                        message=SimpleNamespace(content=response),
                    )]
                )
            )
        )
    )
    result = judge.grade_correctness("Question", "expected", "actual")
    assert result["parse_ok"] is True
    assert result["score"] == pytest.approx(0.8)
    assert result["passed"] is True
