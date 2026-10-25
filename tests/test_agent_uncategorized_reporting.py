"""Optional task categories must not crash mixed benchmark reports."""

import json

import evalforge.agent_runner as agent_runner_module
from evalforge.agent_runner import AgentEvalRunner


class FakeAgent:
    def __init__(self, model="claude", defense_enabled=True):
        pass

    def run(self, prompt, mock_responses=None):
        return {
            "tool_calls": [], "final_answer": "done", "turns": 1,
            "latency_ms": 1, "tokens": {}, "error": None,
        }


def test_mixed_categorized_and_uncategorized_tasks_report(monkeypatch, tmp_path, capsys):
    tasks = [
        {"id": "unlabeled", "prompt": "Say done.", "expected_tools": [],
         "answer_must_contain": ["done"]},
        {"id": "labeled", "category": "qa", "prompt": "Say done.",
         "expected_tools": [], "answer_must_contain": ["done"]},
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    monkeypatch.setattr(agent_runner_module, "ToolCallingAgent", FakeAgent)

    runner = AgentEvalRunner(path, output_dir=tmp_path)
    runner.load()
    results = runner.run()
    assert [row["category"] for row in results] == ["uncategorized", "qa"]

    report = runner.generate_report()
    assert report["total_tasks"] == 2
    assert report["by_category"]["uncategorized"]["success_rate"] == 1.0
    assert report["by_category"]["qa"]["success_rate"] == 1.0
    runner.print_report()
    assert "uncategorized" in capsys.readouterr().out


def test_legacy_results_with_null_category_remain_reportable(tmp_path):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [{
        "category": None, "error": None,
        "grade": {
            "success": True, "tool_selection_score": 1.0,
            "tool_sequence_score": 1.0, "argument_accuracy": None,
            "unnecessary_calls": [], "failure_category": None,
        },
        "injection_resisted": None, "refused_destructive": None,
        "latency_ms": 2, "turns": 1, "cost_usd": 0.0,
    }]
    report = runner.generate_report()
    assert report["by_category"]["uncategorized"]["evaluated_tasks"] == 1
