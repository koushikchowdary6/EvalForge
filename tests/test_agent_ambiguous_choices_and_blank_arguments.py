"""Fail closed on ambiguous provider choices and empty required tool inputs."""

import json
from types import SimpleNamespace

import pytest

import evalforge.agent as agent_module
from evalforge.agent import ToolCallingAgent
from evalforge.agent_graders import grade_trajectory
from evalforge.agent_tools import valid_tool_arguments


@pytest.mark.parametrize("tool,args", [
    ("get_weather", {"city": ""}),
    ("get_weather", {"city": " \t\n "}),
    ("search_knowledge_base", {"query": ""}),
    ("calculate", {"expression": "  "}),
    ("lookup_user", {"user_id": "\t"}),
    ("delete_file", {"path": ""}),
    ("send_email", {"to": "person@example.test", "subject": " ", "body": "message"}),
    ("send_email", {"to": "person@example.test", "subject": "hello", "body": ""}),
])
def test_required_tool_strings_cannot_be_empty_or_whitespace(tool, args):
    assert not valid_tool_arguments(tool, args)


@pytest.mark.parametrize("provider", ["gpt", "claude"])
def test_blank_argument_invalidates_entire_provider_batch(provider, monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent_module, "execute_tool",
        lambda *args: calls.append(args) or "executed",
    )
    valid, blank = {"city": "Boston"}, {"city": " "}
    if provider == "gpt":
        tool_calls = [
            SimpleNamespace(
                id=f"call-{i}",
                function=SimpleNamespace(name="get_weather", arguments=json.dumps(args)),
            )
            for i, args in enumerate([valid, blank])
        ]
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, tool_calls=tool_calls),
            )],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
        )
        agent = ToolCallingAgent(model="gpt")
        agent.openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
        )
    else:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="tool_use", name="get_weather", input=args, id=f"call-{i}")
                for i, args in enumerate([valid, blank])
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=2, output_tokens=2),
        )
        agent = ToolCallingAgent(model="claude")
        agent.anthropic_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: response)
        )

    trajectory = agent.run("Check weather")
    assert calls == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert grade_trajectory({"id": "invalid", "expected_tools": ["get_weather"]}, trajectory)[
        "failure_category"
    ] == "incomplete_response"


@pytest.mark.parametrize("choice_count", [0, 2])
def test_openai_missing_or_multiple_choices_never_execute_tools(
    choice_count, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        agent_module, "execute_tool",
        lambda *args: calls.append(args) or "executed",
    )
    choice = SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="get_weather", arguments='{"city":"Boston"}'
                ),
            )],
        ),
    )
    response = SimpleNamespace(
        choices=[choice] * choice_count,
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
    )
    agent = ToolCallingAgent(model="gpt")
    agent.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
    )

    trajectory = agent.run("Check weather")
    assert calls == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert trajectory["error"] is None
    assert grade_trajectory({"id": "ambiguous", "expected_tools": ["get_weather"]}, trajectory)[
        "success"
    ] is False
