"""Regression tests for AgentEvalRunner evaluation-state isolation."""

import json

import evalforge.agent_runner as agent_runner_module
from evalforge.agent_runner import AgentEvalRunner


class FakeAgent:
    def __init__(self, model="claude", defense_enabled=True):
        self.model = model
        self.defense_enabled = defense_enabled

    def run(self, prompt, mock_responses=None):
        return {
            "tool_calls": [],
            "final_answer": "done",
            "turns": 1,
            "latency_ms": 5,
            "tokens": {},
            "error": None,
        }


def _task():
    return {
        "id": "runner-1",
        "category": "runner_regression",
        "prompt": "Return done.",
        "expected_tools": [],
        "expected_args": {},
        "forbidden_tools": [],
        "answer_must_contain": ["done"],
    }


def test_repeated_run_replaces_results_and_invalidates_report(monkeypatch, tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps([_task()]), encoding="utf-8")
    monkeypatch.setattr(agent_runner_module, "ToolCallingAgent", FakeAgent)

    runner = AgentEvalRunner(tasks_path, output_dir=tmp_path)
    runner.load()
    runner.results = [{"stale": True}]
    runner.report = {"stale": True}

    first = runner.run()
    assert len(first) == 1
    assert first[0]["case_id"] == "runner-1"
    assert runner.report is None

    runner.generate_report()
    assert runner.report["total_tasks"] == 1

    second = runner.run()
    assert len(second) == 1
    assert second[0]["case_id"] == "runner-1"
    assert runner.report is None


def test_empty_report_prints_without_numeric_format_errors(tmp_path, capsys):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text("[]", encoding="utf-8")
    runner = AgentEvalRunner(tasks_path, output_dir=tmp_path)

    report = runner.generate_report()
    assert report["total_tasks"] == 0
    assert report["overall"]["avg_latency_ms"] is None
    assert report["overall"]["avg_turns"] is None

    runner.print_report()
    output = capsys.readouterr().out
    assert "Avg latency ................ n/a" in output
    assert "Avg turns .................. n/a" in output
