"""Regression tests for separate minimum-content and maximum-raw-length rules."""

from evalforge.graders import RuleBasedGrader


def test_max_length_rejects_nonvisible_padding():
    grader = RuleBasedGrader()
    result = grader.grade("", "ok" + "\u200b" * 100, [{"type": "max_length", "value": 2}])
    assert result["passed"] is False
    assert "Response too long" in result["details"]


def test_max_length_accepts_response_within_raw_budget():
    grader = RuleBasedGrader()
    result = grader.grade("", "ok", [{"type": "max_length", "value": 2}])
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_max_length_still_rejects_visible_overflow():
    grader = RuleBasedGrader()
    result = grader.grade("", "three words here", [{"type": "max_length", "value": 5}])
    assert result["passed"] is False
    assert "Response too long" in result["details"]


def test_minimum_uses_visible_content_but_maximum_uses_raw_length():
    grader = RuleBasedGrader()
    rules = [{"type": "min_length", "value": 3}, {"type": "max_length", "value": 3}]
    result = grader.grade("", "a   b", rules)
    assert result["passed"] is False
    assert "Response too long" in result["details"]


def test_minimum_and_maximum_can_both_pass_without_padding():
    grader = RuleBasedGrader()
    rules = [{"type": "min_length", "value": 3}, {"type": "max_length", "value": 3}]
    result = grader.grade("", "a b", rules)
    assert result["passed"] is True
    assert result["score"] == 1.0
