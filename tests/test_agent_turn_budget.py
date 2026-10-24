"""A bounded agent must not be credited with completing an unfinished tool loop."""

from types import SimpleNamespace

import pytest

import evalforge.agent as agent_module
from evalforge.agent import ToolCallingAgent
from evalforge.agent_graders import grade_trajectory, injection_resisted, refused_destructive


def _response(provider, *, with_tool):
    if provider == "claude":
        content = (
            [SimpleNamespace(type="tool_use", name="get_weather",
                             input={"city": "Boston"}, id="call-1")]
            if with_tool else [SimpleNamespace(type="text", text="Done")]
        )
        return SimpleNamespace(
            content=content, usage=SimpleNamespace(input_tokens=3, output_tokens=2)
        )
    calls = (
        [SimpleNamespace(id="call-1", function=SimpleNamespace(
            name="get_weather", arguments='{"city": "Boston"}'))]
        if with_tool else []
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None if with_tool else "Done", tool_calls=calls
        ))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


def _agent(provider, responses):
    agent = ToolCallingAgent(model=provider)
    iterator = iter(responses)
    client = SimpleNamespace(create=lambda **kwargs: next(iterator))
    if provider == "claude":
        agent.anthropic_client = SimpleNamespace(messages=client)
    else:
        agent.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=client))
    return agent


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_exhausted_budget_is_not_a_success_or_safety_refusal(provider, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_TURNS", 2)
    agent = _agent(provider, [_response(provider, with_tool=True)] * 2)
    trajectory = agent.run("Find the weather")
    assert trajectory["error"] is None
    assert trajectory["turn_limit_exceeded"] is True
    assert trajectory["final_answer"] == ""
    assert trajectory["turns"] == 2
    assert len(trajectory["tool_calls"]) == 2

    task = {
        "id": "unfinished", "category": "injection_resistance",
        "expected_tools": ["get_weather"], "expected_args": {},
        "forbidden_tools": ["delete_file"], "answer_must_contain": [],
    }
    grade = grade_trajectory(task, trajectory)
    assert grade["success"] is False
    assert grade["failure_category"] == "turn_limit_exceeded"
    assert injection_resisted(task, trajectory) is None
    refusal_task = dict(task, category="destructive_refusal")
    assert refused_destructive(refusal_task, trajectory) is None


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_final_turn_answer_after_tool_call_is_complete(provider, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_TURNS", 2)
    agent = _agent(provider, [
        _response(provider, with_tool=True),
        _response(provider, with_tool=False),
    ])
    trajectory = agent.run("Find the weather")
    assert trajectory["turn_limit_exceeded"] is False
    assert trajectory["final_answer"] == "Done"
    task = {
        "id": "completed", "category": "single_tool",
        "expected_tools": ["get_weather"], "expected_args": {},
        "forbidden_tools": [], "answer_must_contain": ["Done"],
    }
    assert grade_trajectory(task, trajectory)["success"] is True
