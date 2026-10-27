"""Textual benchmark labels must match complete words, not substrings."""

import pytest

from evalforge.agent_graders import answer_contains_all, grade_trajectory


@pytest.mark.parametrize(("response", "required", "expected"), [
    ("The answer is yesterday.", "yes", False),
    ("The user is unsuspended.", "suspended", False),
    ("The city is Parisian.", "Paris", False),
    ("It is a New Yorker.", "New York", False),
    ("Application programming interfaces are useful.", "application programming interface", False),
    ("The user is suspended.", "suspended", True),
    ("Yes, that's correct.", "yes", True),
    ("The API works.", "API", True),
    ("MFA-enabled login", "MFA", True),
    ("The capital is New York City.", "New York", True),
    ("status=suspended.", "status=suspended", True),
    ("status=suspendedish", "status=suspended", False),
    ("The result is 1,234 apples.", "1234 apples", True),
    ("The result is 1,2 apples.", "12 apples", False),
])
def test_textual_reference_respects_word_boundaries(response, required, expected):
    assert answer_contains_all(response, [required]) is expected


def test_partial_word_does_not_pass_agent_trajectory():
    task = {
        "id": "text-label-boundary", "category": "no_tool",
        "expected_tools": [], "expected_args": {},
        "answer_must_contain": ["suspended"],
    }
    result = grade_trajectory(task, {
        "tool_calls": [], "final_answer": "The user is unsuspended.", "turns": 1,
    })
    assert result["answer_ok"] is False
    assert result["success"] is False
    assert result["failure_category"] == "wrong_answer"
