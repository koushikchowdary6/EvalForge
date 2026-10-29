"""Repeated reference keywords must not overweight deterministic grades."""

from evalforge.graders import RuleBasedGrader


def test_repeated_missing_keyword_counts_as_one_failure():
    result = RuleBasedGrader().grade("Canberra " * 8, "Sydney")

    assert result["passed"] is False
    assert result["failed_rules"] == ["Missing key term: 'Canberra'"]
    # One missing distinct term earns one penalty, then the existing
    # pass/fail score cap applies. Repetition must not drive the score to zero.
    assert result["score"] == 0.5


def test_case_variants_of_same_reference_term_are_not_double_counted():
    result = RuleBasedGrader().grade(
        "Canberra canberra CANBERRA", "CANBERRA is the answer."
    )
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_distinct_reference_terms_still_receive_independent_checks():
    result = RuleBasedGrader().grade(
        "Canberra Australia Canberra AUSTRALIA", "Canberra"
    )
    assert result["passed"] is False
    assert result["failed_rules"] == ["Missing key term: 'Australia'"]
