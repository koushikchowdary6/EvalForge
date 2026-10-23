"""Agent sequence metrics distinguish correct tool sets from correct workflows."""

from evalforge.agent_graders import grade_trajectory
from evalforge.agent_runner import AgentEvalRunner


def _result(task_id, tools, *, error=None, category="workflow"):
    task = {
        "id": task_id,
        "category": category,
        "expected_tools": ["lookup", "send"],
        "expected_args": {},
    }
    trajectory = {
        "tool_calls": [{"tool": name, "args": {}} for name in tools],
        "final_answer": "done",
        "turns": len(tools),
        "error": error,
    }
    return {
        "category": category,
        "error": error,
        "grade": grade_trajectory(task, trajectory),
        "injection_resisted": None,
        "refused_destructive": None,
        "latency_ms": 12,
        "turns": len(tools),
        "cost_usd": 0.0,
    }


def test_sequence_metric_detects_wrong_order_despite_correct_tool_selection(tmp_path):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    correct = _result("correct", ["lookup", "send"])
    wrong_order = _result("reversed", ["send", "lookup"])
    outage = _result("outage", [], error="provider unavailable", category="outage")
    runner.results = [correct, wrong_order, outage]

    assert correct["grade"]["success"] is True
    assert wrong_order["grade"]["success"] is False
    assert wrong_order["grade"]["failure_category"] == "wrong_tool_order"
    assert correct["grade"]["tool_selection_score"] == 1.0
    assert wrong_order["grade"]["tool_selection_score"] == 1.0

    report = runner.generate_report()
    assert report["evaluated_tasks"] == 2
    assert report["execution_errors"] == 1
    assert report["overall"]["tool_selection_accuracy"] == 1.0
    assert report["overall"]["tool_sequence_accuracy"] == 0.5
    assert report["overall"]["task_success_rate"] == 0.5
    assert report["by_category"]["workflow"]["tool_sequence_accuracy"] == 0.5
    assert report["by_category"]["outage"]["tool_sequence_accuracy"] is None


def test_sequence_accuracy_prints_in_report(tmp_path, capsys):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [_result("reversed", ["send", "lookup"])]
    runner.print_report()
    output = capsys.readouterr().out

    assert "Tool selection accuracy .... 100.0%" in output
    assert "Tool sequence accuracy ..... 0.0%" in output
    assert "seq_acc" in output
