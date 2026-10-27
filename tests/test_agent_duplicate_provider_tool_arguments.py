"""Provider tool-call JSON must not silently overwrite duplicate argument keys."""

import json
from types import SimpleNamespace

import pytest

import evalforge.agent as agent_module
from evalforge.agent import ToolCallingAgent, _reject_duplicate_tool_argument_keys
from evalforge.agent_graders import grade_trajectory


@pytest.mark.parametrize("payload", [
    '{"city":"Boston","city":"Miami"}',
    '{"city":"Boston","metadata":{"target":"safe","target":"unsafe"}}',
])
def test_duplicate_keys_are_rejected_even_inside_nested_arguments(payload):
    with pytest.raises(ValueError, match="duplicate tool argument key"):
        json.loads(payload, object_pairs_hook=_reject_duplicate_tool_argument_keys)


@pytest.mark.parametrize("ambiguous_args", [
    '{"city":"Boston","city":"Miami"}',
    '{"city":"Boston","city":" "}',
])
def test_openai_duplicate_key_invalidates_entire_tool_batch(monkeypatch, ambiguous_args):
    executed = []
    monkeypatch.setattr(
        agent_module, "execute_tool",
        lambda *args: executed.append(args) or "executed",
    )
    calls = [
        SimpleNamespace(
            id=f"call-{i}",
            function=SimpleNamespace(name="get_weather", arguments=arguments),
        )
        for i, arguments in enumerate([
            '{"city":"Seattle"}', ambiguous_args,
        ])
    ]
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(content=None, tool_calls=calls),
        )],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
    )
    agent = ToolCallingAgent(model="gpt")
    agent.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response),
        ),
    )
    trajectory = agent.run("Check weather")

    assert executed == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert trajectory["final_answer"] == ""
    assert trajectory["error"] is None
    grade = grade_trajectory(
        {"id": "ambiguous-tool-call", "expected_tools": ["get_weather"]},
        trajectory,
    )
    assert grade["success"] is False
    assert grade["failure_category"] == "incomplete_response"
