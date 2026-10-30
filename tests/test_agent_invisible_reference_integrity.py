"""Invisible-only benchmark labels must not inflate agent evaluation scores."""

import pytest

from evalforge.agent_graders import answer_contains_all, grade_trajectory
from evalforge.agent_runner import validate_agent_tasks


@pytest.mark.parametrize(("field", "value"), [
    ("prompt", "\u200b"),
    ("prompt", "\u0301"),
    ("category", "\u2060"),
    ("answer_must_contain", "\u200b"),
    ("answer_must_contain", " \u200b\t"),
])
def test_invisible_only_benchmark_fields_are_rejected(field, value):
    task = {
        "id": "visible-case",
        "prompt": "Answer the question",
        "category": "no_tool",
        "answer_must_contain": ["yes"],
    }
    if field == "answer_must_contain":
        task[field] = [value]
    else:
        task[field] = value
    with pytest.raises(ValueError, match=field):
        validate_agent_tasks([task])


@pytest.mark.parametrize("reference", ["", " \t", "\u200b", "\u0301", None])
def test_direct_grader_never_credits_invalid_reference(reference):
    assert answer_contains_all("Done \u200b", [reference]) is False
    result = grade_trajectory({
        "id": "unvalidated-case",
        "expected_tools": [],
        "answer_must_contain": [reference],
    }, {"tool_calls": [], "final_answer": "Done \u200b", "turns": 1})
    assert result["success"] is False
    assert result["failure_category"] == "wrong_answer"


def test_visible_text_with_format_controls_remains_valid():
    task = {
        "id": "visible-case",
        "prompt": "Please answer\u200b",
        "category": "no_tool",
        "answer_must_contain": ["yes\u200b"],
    }
    validate_agent_tasks([task])
    assert answer_contains_all("yes\u200b", ["yes\u200b"])
