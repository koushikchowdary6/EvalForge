"""Regression coverage for deterministic rule-grader scoring integrity."""

import pytest

from evalforge.graders import RuleBasedGrader


@pytest.mark.parametrize(
    "reference,response",
    [
        ("The capital is Canberra", "The capitalization is Canberran"),
        ("Security token", "Security tokenization"),
        ("A database", "A databaseX"),
    ],
)
def test_keyword_substrings_do_not_count_as_correct_answers(reference, response):
    result = RuleBasedGrader().grade(reference, response)
    assert result["passed"] is False
    assert result["score"] < 0.6
    assert "Missing key term" in result["details"]


def test_whole_terms_and_case_insensitive_matching_still_pass():
    result = RuleBasedGrader().grade(
        "The capital is Canberra", "CANBERRA is the capital of Australia."
    )
    assert result["passed"] is True
    assert result["score"] == 1.0


@pytest.mark.parametrize("empty", ["", "  ", "\n\t"])
def test_stopword_only_reference_does_not_credit_blank_completion(empty):
    result = RuleBasedGrader().grade("Yes or no?", empty)
    assert result["passed"] is False
    assert result["score"] < 0.6
    assert "Empty model response" in result["details"]


def test_stopword_only_reference_preserves_nonempty_response_behavior():
    result = RuleBasedGrader().grade("Yes or no?", "Sure")
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_custom_rule_failure_cannot_report_passing_numeric_score():
    result = RuleBasedGrader().grade(
        "Canberra",
        "Canberra",
        rules=[{"type": "not_contains", "value": "Canberra"}],
    )
    assert result["passed"] is False
    assert result["score"] < 0.6
