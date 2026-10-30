"""Count-aware tool-call precision and recall diagnostics for agent benchmarks."""

import pytest

from evalforge.agent_graders import grade_trajectory, tool_call_precision_recall
from evalforge.agent_runner import AgentEvalRunner


@pytest.mark.parametrize(
    ("actual", "expected", "precision", "recall"),
    [
        ([], [], 1.0, 1.0),
        ([], ["lookup"], 0.0, 0.0),
        (["lookup"], [], 0.0, 1.0),
        (["lookup", "lookup"], ["lookup", "lookup"], 1.0, 1.0),
        (["lookup"], ["lookup", "lookup"], 1.0, 0.5),
        (["lookup", "lookup"], ["lookup"], 0.5, 1.0),
        (["lookup", "send", "send"], ["lookup", "lookup", "send"], 2 / 3, 2 / 3),
        (["send", "lookup"], ["lookup", "send"], 1.0, 1.0),
    ],
)
def test_tool_call_precision_recall_counts_repeated_invocations(
    actual, expected, precision, recall
):
    observed = tool_call_precision_recall(actual, expected)
    assert observed == pytest.approx((precision, recall))


def _result(case_id, actual, expected, *, error=None, category="workflow"):
    task = {"id": case_id, "category": category, "expected_tools": expected}
    trajectory = {
        "tool_calls": [{"tool": tool, "args": {}} for tool in actual],
        "final_answer": "done",
        "error": error,
    }
    return {
        "category": category,
        "error": error,
        "grade": grade_trajectory(task, trajectory),
        "injection_resisted": None,
        "refused_destructive": None,
        "latency_ms": 10,
        "turns": len(actual),
        "cost_usd": 0.0,
    }


def test_report_separates_missing_duplicate_and_outage_cases(tmp_path, capsys):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [
        _result("missing", ["lookup"], ["lookup", "lookup"]),
        _result("extra", ["lookup", "lookup"], ["lookup"]),
        _result("outage", [], ["lookup"], error="provider down", category="outage"),
    ]
    report = runner.generate_report()
    assert report["evaluated_tasks"] == 2
    assert report["overall"]["tool_call_precision"] == pytest.approx(0.75)
    assert report["overall"]["tool_call_recall"] == pytest.approx(0.75)
    assert report["by_category"]["workflow"]["tool_call_precision"] == pytest.approx(0.75)
    assert report["by_category"]["workflow"]["tool_call_recall"] == pytest.approx(0.75)
    assert report["by_category"]["outage"]["tool_call_precision"] is None
    assert report["by_category"]["outage"]["tool_call_recall"] is None
    assert runner.results[2]["grade"]["tool_call_precision"] is None
    assert runner.results[2]["grade"]["tool_call_recall"] is None

    runner.print_report()
    output = capsys.readouterr().out
    assert "Tool-call precision ........ 75.0%" in output
    assert "Tool-call recall ........... 75.0%" in output
    assert "precision" in output
    assert "recall" in output


def test_no_tool_case_is_not_dropped_from_precision_aggregate(tmp_path):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [
        _result("no-tool", [], [], category="no_tool"),
        _result("missing-tool", [], ["lookup"], category="workflow"),
    ]
    report = runner.generate_report()
    assert report["overall"]["tool_call_precision"] == pytest.approx(0.5)
    assert report["overall"]["tool_call_recall"] == pytest.approx(0.5)
    assert report["by_category"]["no_tool"]["tool_call_precision"] == 1.0
    assert report["by_category"]["workflow"]["tool_call_recall"] == 0.0
