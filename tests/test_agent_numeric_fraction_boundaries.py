"""Fraction and clock-time components must not inflate agent answer accuracy."""

import pytest

from evalforge.agent_graders import answer_contains_all, grade_trajectory


@pytest.mark.parametrize(("response", "required"), [
    ("The result is 12/5.", "12"),
    ("The result is 3/12.", "12"),
    ("The result is 12:30.", "12"),
    ("The result is -12/3.", "-12"),
    ("The result is 3/-12.", "-12"),
    ("The result is 12.5/2.", "12.5"),
    ("The result is 3/12.5.", "12.5"),
    ("The result is 12.5:30.", "12.5"),
])
def test_fraction_and_time_components_are_not_complete_numeric_answers(response, required):
    assert not answer_contains_all(response, [required])


@pytest.mark.parametrize(("response", "required"), [
    ("The result is 12/5, but the answer is 12.", "12"),
    ("The time is 12:30; the answer is 12.", "12"),
    ("The temperature is -12°C.", "-12"),
    ("The value is +12.5.", "12.5"),
    ("The rate is 12%.", "12"),
])
def test_separate_numeric_answer_and_existing_formats_remain_valid(response, required):
    assert answer_contains_all(response, [required])


def test_fraction_only_response_is_a_failed_agent_answer():
    task = {
        "id": "numeric-fraction-boundary",
        "category": "no_tool",
        "expected_tools": [],
        "expected_args": {},
        "forbidden_tools": [],
        "answer_must_contain": ["12"],
    }
    grade = grade_trajectory(task, {
        "tool_calls": [],
        "final_answer": "The answer is 12/5.",
        "turns": 1,
    })
    assert grade["answer_ok"] is False
    assert grade["success"] is False
    assert grade["failure_category"] == "wrong_answer"
