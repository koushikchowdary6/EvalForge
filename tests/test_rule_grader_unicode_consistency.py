"""Unicode-equivalent text must receive consistent deterministic rubric grades."""

import pytest

from evalforge.graders import RuleBasedGrader


@pytest.mark.parametrize(
    "reference, response",
    [
        ("Café", "Cafe\u0301"),
        ("Straße", "STRASSE"),
        ("Café Straße", "CAFE\u0301 STRASSE"),
    ],
)
def test_reference_keywords_accept_canonical_and_casefold_equivalents(reference, response):
    result = RuleBasedGrader().grade(reference, response)
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["failed_rules"] == []


def test_equivalent_reference_spellings_do_not_double_count_missing_keywords():
    result = RuleBasedGrader().grade("Café Cafe\u0301", "Tea")
    assert result["passed"] is False
    assert result["failed_rules"] == ["Missing key term: 'Café'"]


@pytest.mark.parametrize("required, response", [
    ("Straße", "STRASSE"),
    ("Café", "Cafe\u0301"),
])
def test_contains_rule_accepts_unicode_equivalent_substrings(required, response):
    result = RuleBasedGrader().grade(
        "Yes or no?", response, rules=[{"type": "contains", "value": required}]
    )
    assert result["passed"] is True
    assert result["score"] == 1.0


@pytest.mark.parametrize("forbidden, response", [
    ("Straße", "STRASSE"),
    ("Café", "Cafe\u0301"),
])
def test_not_contains_rule_cannot_be_bypassed_with_unicode_equivalence(forbidden, response):
    result = RuleBasedGrader().grade(
        "Yes or no?", response, rules=[{"type": "not_contains", "value": forbidden}]
    )
    assert result["passed"] is False
    assert "Must NOT contain" in result["details"]
    assert result["score"] <= 0.5


def test_distinct_unicode_keywords_still_fail_independently():
    result = RuleBasedGrader().grade("Café Straße", "CAFE\u0301")
    assert result["passed"] is False
    assert result["failed_rules"] == ["Missing key term: 'Straße'"]


def test_unicode_normalization_does_not_allow_substring_keyword_credit():
    result = RuleBasedGrader().grade("Straße", "STRASSEN")
    assert result["passed"] is False
    assert "Missing key term" in result["details"]
