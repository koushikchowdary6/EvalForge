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


def test_category_success_rates_exclude_provider_errors(tmp_path):
    """A provider outage must not lower measured agent success rates."""
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [
        {
            "category": "mixed",
            "error": None,
            "grade": {
                "success": True, "tool_selection_score": 1.0,
                "argument_accuracy": None, "unnecessary_calls": [],
                "failure_category": None,
            },
            "injection_resisted": None, "refused_destructive": None,
            "latency_ms": 10, "turns": 1, "cost_usd": 0.0,
        },
        {
            "category": "mixed",
            "error": "provider unavailable",
            "grade": {
                "success": False, "tool_selection_score": None,
                "argument_accuracy": None, "unnecessary_calls": [],
                "failure_category": "execution_error",
            },
            "injection_resisted": None, "refused_destructive": None,
            "latency_ms": 5, "turns": 0, "cost_usd": 0.0,
        },
        {
            "category": "outage_only",
            "error": "provider unavailable",
            "grade": {
                "success": False, "tool_selection_score": None,
                "argument_accuracy": None, "unnecessary_calls": [],
                "failure_category": "execution_error",
            },
            "injection_resisted": None, "refused_destructive": None,
            "latency_ms": 3, "turns": 0, "cost_usd": 0.0,
        },
    ]

    report = runner.generate_report()
    assert report["total_tasks"] == 3
    assert report["evaluated_tasks"] == 1
    assert report["execution_errors"] == 2
    assert report["overall"]["task_success_rate"] == 1.0
    assert report["by_category"]["mixed"]["tasks"] == 2
    assert report["by_category"]["mixed"]["evaluated_tasks"] == 1
    assert report["by_category"]["mixed"]["execution_errors"] == 1
    assert report["by_category"]["mixed"]["success_rate"] == 1.0
    assert report["by_category"]["outage_only"]["evaluated_tasks"] == 0
    assert report["by_category"]["outage_only"]["execution_errors"] == 1
    assert report["by_category"]["outage_only"]["success_rate"] is None
