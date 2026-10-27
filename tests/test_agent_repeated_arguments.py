"""Repeated tool calls must be graded against distinct expected arguments.

All cases are deterministic and require no model API keys.
"""

import pytest

from evalforge.agent_graders import argument_accuracy, grade_trajectory
from evalforge.agent_runner import validate_agent_tasks


def task():
    return {
        "id": "two-recipients",
        "prompt": "Send a separate status message to each recipient.",
        "category": "multi_tool",
        "expected_tools": ["send_email", "send_email"],
        "expected_args": {
            "send_email": [
                {"to": "first@example.org"},
                {"to": "second@example.org"},
            ],
        },
        "answer_must_contain": ["sent"],
    }


def trajectory(recipients):
    return {
        "tool_calls": [
            {"tool": "send_email", "args": {"to": recipient}}
            for recipient in recipients
        ],
        "final_answer": "Sent both messages.",
    }


def test_distinct_required_calls_receive_full_credit():
    case = task()
    validate_agent_tasks([case])
    result = grade_trajectory(case, trajectory(["first@example.org", "second@example.org"]))
    assert result["argument_accuracy"] == 1.0
    assert result["success"] is True


def test_repeating_first_action_cannot_get_full_credit():
    case = task()
    result = grade_trajectory(case, trajectory(["first@example.org", "first@example.org"]))
    assert result["tool_selection_score"] == 1.0
    assert result["tool_sequence_score"] == 1.0
    assert result["argument_accuracy"] == 0.5
    assert result["failure_category"] == "bad_arguments"
    assert result["success"] is False


def test_reversing_distinct_targets_is_not_equivalent():
    case = task()
    result = grade_trajectory(case, trajectory(["second@example.org", "first@example.org"]))
    assert result["argument_accuracy"] == 0.0
    assert result["failure_category"] == "bad_arguments"


def test_missing_second_call_has_no_second_argument_credit():
    case = task()
    result = grade_trajectory(case, trajectory(["first@example.org"]))
    assert result["argument_accuracy"] == 0.5
    assert result["failure_category"] == "missing_tool"


def test_interleaved_tools_are_matched_per_tool_in_order():
    expected = {
        "send_email": [{"to": "first@example.org"}, {"to": "second@example.org"}],
        "get_weather": {"city": "Boston"},
    }
    calls = [
        {"tool": "send_email", "args": {"to": "first@example.org"}},
        {"tool": "get_weather", "args": {"city": "Boston"}},
        {"tool": "send_email", "args": {"to": "second@example.org"}},
    ]
    assert argument_accuracy(calls, expected) == 1.0


@pytest.mark.parametrize("bad", [
    [],
    ["first@example.org", {"to": "second@example.org"}],
    [{"to": "first@example.org"}],
    [
        {"to": "first@example.org"},
        {"to": "second@example.org"},
        {"to": "third@example.org"},
    ],
])
def test_invalid_repeated_argument_labels_are_rejected(bad):
    case = task()
    case["expected_args"] = {"send_email": bad}
    with pytest.raises(ValueError, match="expected_args"):
        validate_agent_tasks([case])


def test_legacy_single_argument_map_remains_compatible():
    case = task()
    case["expected_args"] = {"send_email": {"to": "second@example.org"}}
    validate_agent_tasks([case])
    assert argument_accuracy(
        trajectory(["first@example.org", "second@example.org"])["tool_calls"],
        case["expected_args"],
    ) == 1.0
