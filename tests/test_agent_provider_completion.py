"""Provider stop reasons must gate agent completion and tool execution."""

from types import SimpleNamespace

import pytest

import evalforge.agent as agent_module
from evalforge.agent import ToolCallingAgent
from evalforge.agent_graders import grade_trajectory, injection_resisted, refused_destructive


def _response(provider, reason, *, tool=False, text="Partial answer"):
    if provider == "claude":
        blocks = [SimpleNamespace(type="text", text=text)]
        if tool:
            blocks.append(SimpleNamespace(
                type="tool_use", name="get_weather", input={"city": "Boston"},
                id="call-1",
            ))
        return SimpleNamespace(
            content=blocks, stop_reason=reason,
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        )
    calls = [SimpleNamespace(
        id="call-1", function=SimpleNamespace(
            name="get_weather", arguments='{"city": "Boston"}',
        ),
    )] if tool else []
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=reason,
            message=SimpleNamespace(content=text, tool_calls=calls),
        )],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


def _agent(provider, responses):
    agent = ToolCallingAgent(model=provider)
    client = SimpleNamespace(create=lambda **kwargs: next(responses))
    if provider == "claude":
        agent.anthropic_client = SimpleNamespace(messages=client)
    else:
        agent.openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=client),
        )
    return agent


@pytest.mark.parametrize("provider,reason", [
    ("claude", "max_tokens"), ("claude", "pause_turn"),
    ("claude", "tool_use"), ("claude", "unknown"), ("claude", None),
    ("gpt", "length"), ("gpt", "content_filter"),
    ("gpt", "tool_calls"), ("gpt", "unknown"), ("gpt", None),
])
def test_nonterminal_text_never_passes_benchmark_or_safety(provider, reason):
    agent = _agent(provider, iter([_response(provider, reason)]))
    trajectory = agent.run("Return the answer")
    assert trajectory["error"] is None
    assert trajectory["incomplete_response"] is True
    assert trajectory["turn_limit_exceeded"] is False
    assert trajectory["final_answer"] == ""
    task = {
        "id": "nonterminal", "category": "injection_resistance",
        "expected_tools": [], "forbidden_tools": ["delete_file"],
        "answer_must_contain": [],
    }
    grade = grade_trajectory(task, trajectory)
    assert grade["success"] is False
    assert grade["failure_category"] == "incomplete_response"
    assert injection_resisted(task, trajectory) is False
    assert refused_destructive(
        dict(task, category="destructive_refusal"), trajectory
    ) is False


@pytest.mark.parametrize("provider,reason", [
    ("claude", "end_turn"), ("claude", "stop_sequence"), ("gpt", "stop"),
])
def test_terminal_answer_still_passes(provider, reason):
    agent = _agent(provider, iter([_response(provider, reason, text="Done")]))
    trajectory = agent.run("Return done")
    assert trajectory["final_answer"] == "Done"
    assert trajectory["incomplete_response"] is False
    assert trajectory["turn_limit_exceeded"] is False
    task = {
        "id": "terminal", "expected_tools": [], "forbidden_tools": [],
        "answer_must_contain": ["Done"],
    }
    assert grade_trajectory(task, trajectory)["success"] is True


@pytest.mark.parametrize("provider,bad_reason", [
    ("claude", "max_tokens"), ("claude", "pause_turn"),
    ("gpt", "length"), ("gpt", "content_filter"),
])
def test_nonterminal_tool_payload_is_never_executed(provider, bad_reason, monkeypatch):
    executed = []
    monkeypatch.setattr(agent_module, "execute_tool", lambda *args: executed.append(args) or "ok")
    agent = _agent(provider, iter([_response(provider, bad_reason, tool=True)]))
    trajectory = agent.run("Use the tool")
    assert executed == []
    assert trajectory["tool_calls"] == []
    assert trajectory["incomplete_response"] is True
    assert trajectory["final_answer"] == ""


@pytest.mark.parametrize("provider,tool_reason,terminal_reason", [
    ("claude", "tool_use", "end_turn"), ("gpt", "tool_calls", "stop"),
])
def test_completed_tool_turn_executes_once(provider, tool_reason, terminal_reason, monkeypatch):
    executed = []
    monkeypatch.setattr(agent_module, "execute_tool", lambda *args: executed.append(args) or "ok")
    agent = _agent(provider, iter([
        _response(provider, tool_reason, tool=True),
        _response(provider, terminal_reason, text="Done"),
    ]))
    trajectory = agent.run("Use the tool")
    assert len(executed) == 1
    assert len(trajectory["tool_calls"]) == 1
    assert trajectory["final_answer"] == "Done"
    assert trajectory["incomplete_response"] is False
    assert trajectory["turn_limit_exceeded"] is False
