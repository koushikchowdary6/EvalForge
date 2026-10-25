"""Provider responses must contain exactly one completed text judgment.

All tests are API-free; ambiguous provider output is missing evaluation
coverage, not a passing score or an arbitrarily selected candidate.
"""

from types import SimpleNamespace

import pytest

from evalforge.rag_graders import RAGJudge


GOOD = '{"score": 10, "reason": "supported"}'
BAD = '{"score": 0, "reason": "unsupported"}'


def judge_with(claude_blocks=None, gpt_choices=None):
    judge = RAGJudge.__new__(RAGJudge)
    judge.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                stop_reason="end_turn", content=claude_blocks
            )
        )
    )
    judge.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(choices=gpt_choices)
            )
        )
    )
    return judge


def gpt_choice(text):
    return SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=text)
    )


@pytest.mark.parametrize("grader", ["correctness", "groundedness"])
@pytest.mark.parametrize("provider,payload", [
    ("claude", []),
    ("claude", [SimpleNamespace(type="text", text="")]),
    ("claude", [SimpleNamespace(type="thinking", text=GOOD)]),
    ("claude", [SimpleNamespace(type="text", text=GOOD),
                SimpleNamespace(type="text", text=BAD)]),
    ("gpt", []),
    ("gpt", [gpt_choice(None)]),
    ("gpt", [gpt_choice("  ")]),
    ("gpt", [gpt_choice(GOOD), gpt_choice(BAD)]),
])
def test_ambiguous_or_empty_judge_output_is_excluded(provider, payload, grader):
    judge = judge_with(
        claude_blocks=payload if provider == "claude" else None,
        gpt_choices=payload if provider == "gpt" else None,
    )
    judge.judge_model = provider
    result = (
        judge.grade_correctness("Question", "Expected", "Actual")
        if grader == "correctness"
        else judge.grade_groundedness("Actual", [{"doc_id": "d1", "text": "Evidence"}])
    )
    assert result["score"] is None
    assert result["passed"] is None
    assert result["parse_ok"] is False
    assert "judge error" in result["reason"]


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_single_completed_text_grade_remains_valid(provider):
    judge = judge_with(
        claude_blocks=[SimpleNamespace(type="text", text=GOOD)],
        gpt_choices=[gpt_choice(GOOD)],
    )
    judge.judge_model = provider
    result = judge.grade_correctness("Question", "Expected", "Actual")
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["parse_ok"] is True
