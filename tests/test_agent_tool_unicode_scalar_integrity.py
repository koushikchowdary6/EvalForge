"""Agent tool arguments must remain serializable Unicode scalar text."""

import pytest

from evalforge.agent_runner import validate_agent_tasks
from evalforge.agent_tools import valid_tool_arguments


@pytest.mark.parametrize("invalid", [
    "\ud800", "\udfff", "Boston\ud800", "\udfffU-123",
])
def test_invalid_unicode_scalars_cannot_enter_agent_tool_evidence(invalid):
    assert not valid_tool_arguments("get_weather", {"city": invalid})
    task = {
        "id": "invalid-scalar-argument",
        "prompt": "Check the weather in Boston.",
        "expected_tools": ["get_weather"],
        "expected_args": {"get_weather": {"city": invalid}},
        "forbidden_tools": [],
    }
    with pytest.raises(ValueError, match="expected_args"):
        validate_agent_tasks([task])


@pytest.mark.parametrize("valid", [
    "Boston", "München", "東京", "City \U0001f30d", "Cafe\u0301",
])
def test_valid_unicode_scalar_arguments_still_pass(valid):
    assert valid_tool_arguments("get_weather", {"city": valid})
    validate_agent_tasks([{
        "id": "valid-scalar-argument",
        "prompt": "Check the weather.",
        "expected_tools": ["get_weather"],
        "expected_args": {"get_weather": {"city": valid}},
        "forbidden_tools": [],
    }])
