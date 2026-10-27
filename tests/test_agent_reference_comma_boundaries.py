"""Prevent malformed comma grouping from inflating agent answer grades."""

import pytest

from evalforge.agent_graders import answer_contains_all, grade_trajectory


@pytest.mark.parametrize(("response", "required", "expected"), [
    ("The result is 12.", "1,2", False),
    ("The result is 1,2.", "1,2", True),
    ("The result is 1,20.", "1,2", False),
    ("The result is 1,2,3.", "1,2", False),
    ("The result is -1,2.", "1,2", False),
    ("The result is 1234.", "1,234", True),
    ("The result is 1,234.", "1234", True),
    ("The result is 1,2 apples.", "12 apples", False),
    ("The result is 1,234 apples.", "1234 apples", True),
])
def test_numeric_and_text_reference_comma_boundaries(response, required, expected):
    assert answer_contains_all(response, [required]) is expected


def test_malformed_comma_reference_cannot_pass_trajectory():
    task = {
        "id": "numeric-comma-reference", "category": "no_tool",
        "expected_tools": [], "expected_args": {},
        "answer_must_contain": ["1,2"],
    }
    result = grade_trajectory(task, {
        "tool_calls": [], "final_answer": "The result is 12.", "turns": 1,
    })
    assert result["answer_ok"] is False
    assert result["success"] is False
    assert result["failure_category"] == "wrong_answer"
