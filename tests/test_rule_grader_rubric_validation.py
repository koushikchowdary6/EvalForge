"""Rule-based benchmark rubrics must fail closed when misconfigured."""

import pytest

from evalforge.graders import RuleBasedGrader


@pytest.mark.parametrize("rules", [
    {"type": "contains", "value": "Canberra"},
    "contains",
    [None],
    ["contains"],
    [{}],
    [{"type": "contians", "value": "Canberra"}],
    [{"type": "contains"}],
    [{"type": "contains", "value": ""}],
    [{"type": "contains", "value": "  "}],
    [{"type": "contains", "value": 5}],
    [{"type": "not_contains", "value": ""}],
    [{"type": "min_length", "value": True}],
    [{"type": "min_length", "value": -1}],
    [{"type": "min_length", "value": 0}],
    [{"type": "max_length", "value": -1}],
    [{"type": "max_length", "value": 1.5}],
    [{"type": "max_length", "value": "10"}],
])
def test_malformed_rules_cannot_silently_credit_answers(rules):
    with pytest.raises(ValueError):
        RuleBasedGrader().grade("Canberra", "Canberra", rules=rules)


def test_valid_text_and_length_rules_preserve_grading():
    result = RuleBasedGrader().grade(
        "Canberra",
        "Canberra is the capital of Australia",
        rules=[
            {"type": "contains", "value": "Australia"},
            {"type": "not_contains", "value": "Sydney"},
            {"type": "min_length", "value": 10},
            {"type": "max_length", "value": 100},
        ],
    )
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_default_length_bounds_remain_supported():
    result = RuleBasedGrader().grade(
        "Canberra", "Canberra", rules=[{"type": "min_length"}]
    )
    assert result["passed"] is False
    assert "too short" in result["details"]


def test_valid_zero_maximum_length_is_an_explicit_constraint():
    result = RuleBasedGrader().grade(
        "Canberra", "Canberra", rules=[{"type": "max_length", "value": 0}]
    )
    assert result["passed"] is False
    assert "too long" in result["details"]
