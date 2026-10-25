"""Malformed provider tool arguments must never execute simulated actions."""

import json
from types import SimpleNamespace

import pytest

import evalforge.agent as agent_module
from evalforge.agent import ToolCallingAgent
from evalforge.agent_graders import grade_trajectory
from evalforge.agent_tools import valid_tool_arguments


def _agent(provider, response):
    agent = ToolCallingAgent(model=provider)
    endpoint = SimpleNamespace(create=lambda **kwargs: response)
    if provider == "gpt":
        agent.openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=endpoint),
        )
    else:
        agent.anthropic_client = SimpleNamespace(messages=endpoint)
    return agent


def _openai_response(arguments):
    calls = [
        SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(name="get_weather", arguments=value),
        )
        for index, value in enumerate(arguments)
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(content=None, tool_calls=calls),
        )],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
    )


def _anthropic_response(arguments):
    blocks = [
        SimpleNamespace(
            type="tool_use", name="get_weather", input=value, id=f"call-{index}",
        )
        for index, value in enumerate(arguments)
    ]
    return SimpleNamespace(
        content=blocks,
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=5, output_tokens=5),
    )


@pytest.mark.parametrize("invalid", [
    "{bad", "[]", "null", "42", '"hello"', "", None,
])
def test_openai_malformed_json_and_non_object_args_are_not_executed(invalid, monkeypatch):
    executed = []
    monkeypatch.setattr(
        agent_module, "execute_tool",
        lambda *args: executed.append(args) or "ok",
    )
    # A valid earlier call must not execute if a later call is malformed.
    response = _openai_response(['{"city": "Boston"}', invalid])
    trajectory = _agent("gpt", response).run("Check weather")
    assert executed == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert trajectory["turn_limit_exceeded"] is False
    assert trajectory["final_answer"] == ""
    assert trajectory["error"] is None

    task = {
        "id": "malformed-batch", "expected_tools": ["get_weather"],
        "answer_must_contain": [],
    }
    grade = grade_trajectory(task, trajectory)
    assert grade["success"] is False
    assert grade["failure_category"] == "incomplete_response"


@pytest.mark.parametrize("invalid", [None, [], "Boston", 42])
def test_anthropic_non_object_args_are_not_executed(invalid, monkeypatch):
    executed = []
    monkeypatch.setattr(
        agent_module, "execute_tool",
        lambda *args: executed.append(args) or "ok",
    )
    response = _anthropic_response([{"city": "Boston"}, invalid])
    trajectory = _agent("claude", response).run("Check weather")
    assert executed == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert trajectory["final_answer"] == ""
    assert trajectory["error"] is None


@pytest.mark.parametrize("provider", ["gpt", "claude"])
@pytest.mark.parametrize("invalid", [
    {},  # required city is absent
    {"city": None},
    {"city": 123},
    {"city": "Boston", "unexpected": "untrusted"},
])
def test_schema_invalid_tool_batches_fail_closed(provider, invalid, monkeypatch):
    executed = []
    monkeypatch.setattr(
        agent_module, "execute_tool",
        lambda *args: executed.append(args) or "ok",
    )
    valid = {"city": "Boston"}
    response = (
        _openai_response([json.dumps(valid), json.dumps(invalid)])
        if provider == "gpt"
        else _anthropic_response([valid, invalid])
    )
    trajectory = _agent(provider, response).run("Check weather")
    assert executed == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert trajectory["error"] is None
    assert grade_trajectory(
        {"id": "schema", "expected_tools": ["get_weather"]}, trajectory
    )["failure_category"] == "incomplete_response"


def test_schema_validation_accepts_valid_calls_and_rejects_unknown_tools():
    assert valid_tool_arguments("get_weather", {"city": "Boston"})
    assert valid_tool_arguments(
        "send_email", {"to": "a@example.test", "subject": "Hi", "body": "Text"}
    )
    assert not valid_tool_arguments("not_registered", {})
    assert not valid_tool_arguments("send_email", {"to": "a@example.test"})
    assert not valid_tool_arguments("get_weather", ["Boston"])
