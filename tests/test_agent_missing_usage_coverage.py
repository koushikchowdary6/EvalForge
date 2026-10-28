"""Missing provider usage must not appear as a measured zero-dollar agent run."""

from types import SimpleNamespace

from evalforge.agent import ToolCallingAgent
from evalforge.agent_graders import estimate_cost


def _gpt_response(*, usage, tool=False):
    calls = (
        [SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="get_weather", arguments='{"city": "Boston"}',
            ),
        )]
        if tool else []
    )
    return SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(
            finish_reason="tool_calls" if tool else "stop",
            message=SimpleNamespace(
                content=None if tool else "Done",
                tool_calls=calls,
            ),
        )],
    )


def _agent(responses):
    agent = ToolCallingAgent(model="gpt")
    agent.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: next(responses))
        )
    )
    return agent


def test_unmetered_completed_turn_has_unknown_cost():
    trajectory = _agent(iter([_gpt_response(usage=None)])).run("Say done")
    assert trajectory["final_answer"] == "Done"
    assert trajectory["error"] is None
    assert trajectory["tokens"] == {}
    assert estimate_cost(trajectory["tokens"], "gpt") is None


def test_one_unmetered_turn_does_not_disappear_from_multi_turn_cost():
    responses = iter([
        _gpt_response(usage=None, tool=True),
        _gpt_response(
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        ),
    ])
    trajectory = _agent(responses).run("Check weather then answer")
    assert trajectory["final_answer"] == "Done"
    assert [call["tool"] for call in trajectory["tool_calls"]] == ["get_weather"]
    assert trajectory["tokens"] == {}
    assert estimate_cost(trajectory["tokens"], "gpt") is None


def test_complete_usage_and_explicit_zero_are_still_measured():
    responses = iter([
        _gpt_response(
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            tool=True,
        ),
        _gpt_response(
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        ),
    ])
    trajectory = _agent(responses).run("Check weather then answer")
    assert trajectory["tokens"] == {"input": 7, "output": 3}
    assert estimate_cost(trajectory["tokens"], "gpt") is not None
    assert estimate_cost({"input": 0, "output": 0}, "gpt") == 0.0
