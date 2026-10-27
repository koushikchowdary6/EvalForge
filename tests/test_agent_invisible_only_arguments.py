"""Invisible-only tool arguments must not be treated as valid evaluation evidence."""

import json
from types import SimpleNamespace

import pytest

import evalforge.agent as agent_module
from evalforge.agent import ToolCallingAgent
from evalforge.agent_runner import validate_agent_tasks
from evalforge.agent_tools import has_visible_argument_text, valid_tool_arguments


@pytest.mark.parametrize("value", [
    "\u200b",  # zero-width space
    "\u2060",  # word joiner
    "\u0301",  # standalone combining mark
    "\u034f",  # combining grapheme joiner
    " \u200b\t",
    "\x00\n",
])
def test_invisible_only_arguments_are_invalid(value):
    assert not has_visible_argument_text(value)
    assert not valid_tool_arguments("get_weather", {"city": value})
    task = {
        "id": "invisible", "prompt": "Get weather",
        "expected_tools": ["get_weather"],
        "expected_args": {"get_weather": {"city": value}},
    }
    with pytest.raises(ValueError, match="expected_args"):
        validate_agent_tasks([task])


@pytest.mark.parametrize("value", ["Boston", "New York", "A\u200bB", "Line one\nLine two"])
def test_visible_arguments_remain_valid(value):
    assert has_visible_argument_text(value)
    assert valid_tool_arguments("get_weather", {"city": value})
    validate_agent_tasks([{
        "id": "visible", "prompt": "Get weather",
        "expected_tools": ["get_weather"],
        "expected_args": {"get_weather": {"city": value}},
    }])


@pytest.mark.parametrize("provider", ["gpt", "claude"])
def test_invisible_tool_call_invalidates_entire_batch(provider, monkeypatch):
    executed = []
    monkeypatch.setattr(
        agent_module, "execute_tool",
        lambda *args: executed.append(args) or "ok",
    )
    agent = ToolCallingAgent(model=provider)
    if provider == "gpt":
        calls = [
            SimpleNamespace(id=f"call-{index}", function=SimpleNamespace(
                name="get_weather", arguments=json.dumps({"city": city})
            ))
            for index, city in enumerate(("Boston", "\u200b"))
        ]
        response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
                content=None, tool_calls=calls,
            ))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
        )
        agent.openai_client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response),
        ))
    else:
        blocks = [
            SimpleNamespace(type="tool_use", name="get_weather",
                            input={"city": city}, id=f"call-{index}")
            for index, city in enumerate(("Boston", "\u200b"))
        ]
        response = SimpleNamespace(
            content=blocks, stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=5, output_tokens=5),
        )
        agent.anthropic_client = SimpleNamespace(messages=SimpleNamespace(
            create=lambda **kwargs: response,
        ))
    trajectory = agent.run("Get weather")
    assert executed == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert trajectory["error"] is None
