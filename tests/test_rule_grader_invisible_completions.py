"""Invisible-only completions must not inflate deterministic benchmark scores."""

import pytest

from evalforge.graders import RuleBasedGrader


@pytest.mark.parametrize(
    "completion",
    ["\u200b", "\u200d", "\u0301", "\u034f", " \t\u200b\u0301\n"],
)
def test_stopword_only_reference_rejects_invisible_completion(completion):
    # With no extractable keywords, the legacy grader silently awarded 1.0.
    result = RuleBasedGrader().grade("Yes or no?", completion)
    assert result["passed"] is False
    assert result["score"] < 0.6
    assert "Empty model response" in result["details"]


def test_visible_multilingual_answer_remains_gradable():
    result = RuleBasedGrader().grade("Yes or no?", "はい")
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_visible_answer_with_format_character_is_not_treated_as_empty():
    result = RuleBasedGrader().grade("Yes or no?", "Yes\u200b")
    assert result["passed"] is True
    assert result["score"] == 1.0
