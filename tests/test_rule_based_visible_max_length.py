"""Regression tests for consistent visible-length rule grading."""

from evalforge.graders import RuleBasedGrader


def test_max_length_ignores_nonvisible_padding():
    grader = RuleBasedGrader()
    result = grader.grade("", "ok" + "\u200b" * 100, [{"type": "max_length", "value": 2}])
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_max_length_still_rejects_visible_overflow():
    grader = RuleBasedGrader()
    result = grader.grade("", "three words here", [{"type": "max_length", "value": 5}])
    assert result["passed"] is False
    assert "Response too long" in result["details"]


def test_min_and_max_length_share_whitespace_semantics():
    grader = RuleBasedGrader()
    rules = [{"type": "min_length", "value": 3}, {"type": "max_length", "value": 3}]
    result = grader.grade("", "a   b", rules)
    assert result["passed"] is True
