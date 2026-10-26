"""Offline regressions: incomplete judge provider responses must not inflate scores."""

from types import SimpleNamespace

import pytest

from evalforge.graders import LLMJudgeGrader


SCORE_JSON = '{"score": 10, "reason": "plausible but untrusted"}'


def _claude_response(*texts, stop_reason="end_turn", block_type="text"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type=block_type, text=text) for text in texts],
    )


def _gpt_response(*texts, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text),
            )
            for text in texts
        ]
    )


def _grade_claude(response):
    grader = LLMJudgeGrader.__new__(LLMJudgeGrader)
    grader.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **_kwargs: response)
    )
    return grader._judge_with_claude("expected", "actual", "factual")


def _grade_gpt(response):
    grader = LLMJudgeGrader.__new__(LLMJudgeGrader)
    grader.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    return grader._judge_with_gpt("expected", "actual", "factual")


@pytest.mark.parametrize("response", [
    _claude_response(SCORE_JSON, stop_reason="max_tokens"),
    _claude_response(SCORE_JSON, stop_reason="tool_use"),
    _claude_response(SCORE_JSON, stop_reason="pause_turn"),
    _claude_response(SCORE_JSON, stop_reason=None),
    _claude_response(SCORE_JSON, SCORE_JSON),
    _claude_response(SCORE_JSON, block_type="tool_use"),
    _claude_response(""),
    _claude_response(),
])
def test_claude_incomplete_or_ambiguous_judgment_fails_closed(response):
    result = _grade_claude(response)
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "judging error" in result["feedback"].lower()


@pytest.mark.parametrize("response", [
    _gpt_response(SCORE_JSON, finish_reason="length"),
    _gpt_response(SCORE_JSON, finish_reason="content_filter"),
    _gpt_response(SCORE_JSON, finish_reason=None),
    _gpt_response(SCORE_JSON, SCORE_JSON),
    _gpt_response(None),
    _gpt_response("  "),
    _gpt_response(),
])
def test_gpt_incomplete_or_ambiguous_judgment_fails_closed(response):
    result = _grade_gpt(response)
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "judging error" in result["feedback"].lower()


def test_single_completed_judge_response_still_receives_valid_score():
    for result in (
        _grade_claude(_claude_response(SCORE_JSON)),
        _grade_gpt(_gpt_response(SCORE_JSON)),
    ):
        assert result["passed"] is True
        assert result["score"] == 1.0
        assert result["feedback"] == "plausible but untrusted"
