"""Benchmark labels must reference actual tool arguments before paid evaluation."""

import pytest

from evalforge.agent_runner import validate_agent_tasks


def case(expected_args):
    return {
        "id": "schema-integrity",
        "prompt": "Look up user U-1 and report the result.",
        "expected_tools": ["lookup_user"],
        "expected_args": expected_args,
        "forbidden_tools": [],
    }


@pytest.mark.parametrize("expected_args", [
    {"lookup_user": {"user": "U-1"}},  # wrong parameter name
    {"lookup_user": {"user_id": 123}},  # not the declared string type
    {"lookup_user": {"user_id": "   "}},
    {"lookup_user": {}},  # explicit but unscorable expectation
    {"lookup_user": {"user_id": "U-1", "unexpected": "x"}},
])
def test_invalid_expected_arguments_fail_before_model_calls(expected_args):
    with pytest.raises(ValueError, match="expected_args"):
        validate_agent_tasks([case(expected_args)])


def test_valid_partial_and_repeated_tool_arguments_remain_supported():
    validate_agent_tasks([case({"lookup_user": {"user_id": "U-1"}})])
    repeated = case({
        "lookup_user": [{"user_id": "U-1"}, {"user_id": "U-2"}]
    })
    repeated["expected_tools"] = ["lookup_user", "lookup_user"]
    validate_agent_tasks([repeated])


def test_repeated_argument_list_cannot_hide_an_invalid_second_call():
    repeated = case({
        "lookup_user": [{"user_id": "U-1"}, {"user": "U-2"}]
    })
    repeated["expected_tools"] = ["lookup_user", "lookup_user"]
    with pytest.raises(ValueError, match="expected_args"):
        validate_agent_tasks([repeated])
