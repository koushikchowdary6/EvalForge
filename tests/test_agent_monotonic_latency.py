"""Agent benchmark latency must use elapsed monotonic time, including failures."""

import pytest

from evalforge import agent as agent_module
from evalforge.agent import ToolCallingAgent


def _agent(monkeypatch, handler):
    instance = ToolCallingAgent.__new__(ToolCallingAgent)
    instance.model = "claude"
    monkeypatch.setattr(instance, "_run_anthropic", handler)
    return instance


def test_agent_success_latency_ignores_wall_clock_changes(monkeypatch):
    ticks = iter([100.0, 100.035])
    monkeypatch.setattr(agent_module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        agent_module.time, "time",
        lambda: (_ for _ in ()).throw(AssertionError("wall clock used")),
    )
    trajectory = _agent(
        monkeypatch,
        lambda prompt, responses: {
            "tool_calls": [], "final_answer": "done", "turns": 1,
            "tokens": {"input": 2, "output": 1},
        },
    ).run("Evaluate a task")
    assert trajectory["final_answer"] == "done"
    assert trajectory["latency_ms"] == pytest.approx(35.0)


def test_agent_exception_latency_is_still_monotonic(monkeypatch):
    ticks = iter([12.0, 12.075])
    monkeypatch.setattr(agent_module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        agent_module.time, "time",
        lambda: (_ for _ in ()).throw(AssertionError("wall clock used")),
    )

    def fail(prompt, responses):
        raise RuntimeError("sensitive provider message")

    trajectory = _agent(monkeypatch, fail).run("Evaluate a task")
    assert trajectory["error"] == "agent error (RuntimeError)"
    assert "sensitive provider message" not in repr(trajectory)
    assert trajectory["latency_ms"] == pytest.approx(75.0)
