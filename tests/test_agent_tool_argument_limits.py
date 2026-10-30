"""Bound provider-generated arguments and reject impossible benchmark labels."""

import json
from types import SimpleNamespace

import pytest

import evalforge.agent as agent_module
from evalforge.agent import ToolCallingAgent
from evalforge.agent_runner import validate_agent_tasks
from evalforge.agent_tools import MAX_TOOL_ARGUMENT_CHARS, valid_tool_arguments


def _case(value):
    return {
        "id": "bounded-arguments",
        "prompt": "Look up user U-1.",
        "expected_tools": ["lookup_user"],
        "expected_args": {"lookup_user": {"user_id": value}},
        "forbidden_tools": [],
    }


def test_tool_argument_boundary_and_dataset_contract_agree():
    maximum = "U" * MAX_TOOL_ARGUMENT_CHARS
    too_large = maximum + "U"
    assert valid_tool_arguments("lookup_user", {"user_id": maximum})
    assert not valid_tool_arguments("lookup_user", {"user_id": too_large})
    validate_agent_tasks([_case(maximum)])
    with pytest.raises(ValueError, match="expected_args"):
        validate_agent_tasks([_case(too_large)])


@pytest.mark.parametrize("provider", ["gpt", "claude"])
def test_oversized_call_blocks_entire_provider_batch(provider, monkeypatch):
    executed = []
    monkeypatch.setattr(
        agent_module, "execute_tool",
        lambda *args: executed.append(args) or "simulated",
    )
    calls = [
        {"city": "Boston"},
        {"city": "x" * (MAX_TOOL_ARGUMENT_CHARS + 1)},
    ]
    if provider == "gpt":
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=f"call-{index}",
                            function=SimpleNamespace(
                                name="get_weather", arguments=json.dumps(arguments),
                            ),
                        )
                        for index, arguments in enumerate(calls)
                    ],
                ),
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
        )
        agent = ToolCallingAgent(model="gpt")
        agent.openai_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response),
            ),
        )
    else:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use", name="get_weather",
                    input=arguments, id=f"call-{index}",
                )
                for index, arguments in enumerate(calls)
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=5, output_tokens=5),
        )
        agent = ToolCallingAgent(model="claude")
        agent.anthropic_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kwargs: response),
        )

    trajectory = agent.run("Check the weather")
    assert executed == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert trajectory["error"] is None
