"""Agent evaluation must not credit a different numeric answer."""

import pytest

from evalforge.agent_graders import answer_contains_all, grade_trajectory


@pytest.mark.parametrize("response", [
    "The result is 312.",
    "The result is 120.",
    "The result is 12.5.",
    "The result is 1,200.",
    "The result is -12.",
    "The result is −12.",
    "The result is 1e12.",
    "The result is 12e3.",
    "The result is 12E+3.",
    "The result is 0x12.",
    "The result is v12.",
])
def test_required_number_is_not_a_substring_of_another_value(response):
    assert not answer_contains_all(response, ["12"])


@pytest.mark.parametrize("response", [
    "The result is 12.",
    "The temperature is 12C.",
    "The balance is +12.",
    "The rate is 12%.",
    "The total is 1,081.",
])
def test_numeric_answer_preserves_legitimate_formats(response):
    required = "1081" if "1,081" in response else "12"
    assert answer_contains_all(response, [required])


def test_wrong_numeric_result_cannot_pass_agent_task():
    task = {
        "id": "numeric-boundary", "category": "single_tool",
        "expected_tools": ["calculate"], "expected_args": {},
        "forbidden_tools": [], "answer_must_contain": ["12"],
    }
    trajectory = {
        "tool_calls": [{"tool": "calculate", "args": {"expression": "6 * 2"}}],
        "final_answer": "The result is 120.", "turns": 2,
    }
    grade = grade_trajectory(task, trajectory)
    assert grade["tool_selection_score"] == 1.0
    assert grade["answer_ok"] is False
    assert grade["success"] is False
    assert grade["failure_category"] == "wrong_answer"


@pytest.mark.parametrize("response", [
    "The result is 1,2.",
    "The result is 12,0.",
    "The result is 1, 2.",
    "The result is 1,2,3.",
    "The result is 1,2345.",
])
def test_comma_separated_values_do_not_become_a_different_number(response):
    assert not answer_contains_all(response, ["12"])


@pytest.mark.parametrize(("response", "expected"), [
    ("The result is 1,081.", "1081"),
    ("The result is 1,234,567.", "1234567"),
    ("The result is +1,234.", "1234"),
])
def test_only_valid_thousands_grouping_is_normalized(response, expected):
    assert answer_contains_all(response, [expected])


def test_invalid_grouping_cannot_pass_agent_task():
    task = {
        "id": "comma-boundary", "category": "no_tool",
        "expected_tools": [], "expected_args": {},
        "forbidden_tools": [], "answer_must_contain": ["12"],
    }
    grade = grade_trajectory(task, {
        "tool_calls": [], "final_answer": "The answer is 1,2.", "turns": 1,
    })
    assert grade["answer_ok"] is False
    assert grade["success"] is False
    assert grade["failure_category"] == "wrong_answer"
