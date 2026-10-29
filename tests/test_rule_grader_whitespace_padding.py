"""Whitespace-only padding must not inflate minimum-answer-length grades."""

import pytest

from evalforge.graders import RuleBasedGrader


@pytest.mark.parametrize("padding", [
    " " * 100,
    "\t\n" * 50,
    "\u00a0" * 40,  # non-breaking spaces
    " \u200b \u0301 " * 20,  # spaces with hidden controls and combining marks
])
def test_trailing_whitespace_cannot_satisfy_minimum_length(padding):
    result = RuleBasedGrader().grade(
        "Yes or no?", "Yes" + padding,
        rules=[{"type": "min_length", "value": 10}],
    )
    assert result["passed"] is False
    assert "too short" in result["details"]
    assert result["score"] <= 0.5


def test_leading_whitespace_does_not_count_as_answer_content():
    result = RuleBasedGrader().grade(
        "Yes or no?", "  \tYes\n",
        rules=[{"type": "min_length", "value": 4}],
    )
    assert result["passed"] is False


def test_repeated_interior_whitespace_counts_as_one_separator():
    result = RuleBasedGrader().grade(
        "Yes or no?", "Yes    indeed",
        rules=[{"type": "min_length", "value": 11}],
    )
    assert result["passed"] is False
    assert "too short" in result["details"]


@pytest.mark.parametrize("answer,minimum", [
    ("Yes indeed", 10),
    ("Yes    indeed", 10),
    ("\u4f60\u597d \u4e16\u754c", 5),
])
def test_normal_sentence_separators_still_count_once(answer, minimum):
    result = RuleBasedGrader().grade(
        "Yes or no?", answer,
        rules=[{"type": "min_length", "value": minimum}],
    )
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_maximum_length_still_counts_raw_whitespace():
    result = RuleBasedGrader().grade(
        "Yes or no?", "Yes" + " " * 20,
        rules=[{"type": "max_length", "value": 10}],
    )
    assert result["passed"] is False
    assert "too long" in result["details"]
