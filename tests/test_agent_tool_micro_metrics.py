"""Pooled tool-call diagnostics must not hide duplicate or missing calls."""

import pytest

from evalforge.agent_graders import grade_trajectory, tool_call_overlap_counts
from evalforge.agent_runner import AgentEvalRunner


def _row(case_id, actual, expected, *, error=None, category="workflow"):
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


def test_overlap_counts_respect_duplicate_required_and_extra_calls():
    assert tool_call_overlap_counts(
        ["lookup", "lookup", "send"], ["lookup", "send", "send"]
    ) == {"matched": 2, "predicted": 3, "required": 3}
    assert tool_call_overlap_counts([], []) == {
        "matched": 0, "predicted": 0, "required": 0
    }


def test_micro_metrics_pool_calls_not_per_case_rates(tmp_path, capsys):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [
        _row("correct", ["lookup"], ["lookup"]),
        _row("missing", ["lookup"], ["lookup", "lookup", "lookup"]),
        _row("duplicate", ["lookup"] * 4, ["lookup"]),
        _row("no-tool", [], [], category="no_tool"),
        _row("outage", [], ["lookup"] * 20, error="provider down", category="outage"),
    ]
    report = runner.generate_report()
    overall = report["overall"]
    assert overall["tool_call_micro_samples"] == 4
    assert (
        overall["tool_call_micro_matched"],
        overall["tool_call_micro_predicted"],
        overall["tool_call_micro_required"],
    ) == (3, 6, 5)
    assert overall["tool_call_micro_precision"] == pytest.approx(0.5)
    assert overall["tool_call_micro_recall"] == pytest.approx(0.6)
    assert overall["tool_call_precision"] == pytest.approx(0.8125)
    assert overall["tool_call_recall"] == pytest.approx((1 + 1 / 3 + 1 + 1) / 4)
    assert report["by_category"]["workflow"]["tool_call_micro_precision"] == 0.5
    assert report["by_category"]["workflow"]["tool_call_micro_recall"] == 0.6
    assert report["by_category"]["no_tool"]["tool_call_micro_precision"] == 1.0
    assert report["by_category"]["no_tool"]["tool_call_micro_recall"] == 1.0
    assert report["by_category"]["outage"]["tool_call_micro_precision"] is None
    assert report["by_category"]["outage"]["tool_call_micro_recall"] is None
    assert report["by_category"]["outage"]["tool_call_micro_samples"] == 0
    assert runner.results[-1]["grade"]["tool_call_counts"] is None

    runner.print_report()
    output = capsys.readouterr().out
    assert "Micro tool precision ....... 50.0%" in output
    assert "Micro tool recall .......... 60.0%" in output
    assert "Pooled matched/pred/req .... 3/6/5" in output


def test_no_observed_cases_do_not_report_perfect_micro_scores(tmp_path):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [_row("outage", [], ["lookup"], error="provider down")]
    report = runner.generate_report()
    assert report["overall"]["tool_call_micro_precision"] is None
    assert report["overall"]["tool_call_micro_recall"] is None
    assert report["overall"]["tool_call_micro_samples"] == 0


def test_only_no_tool_cases_are_vacuously_perfect(tmp_path):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [_row("no-tool", [], [], category="no_tool")]
    report = runner.generate_report()
    assert report["overall"]["tool_call_micro_precision"] == 1.0
    assert report["overall"]["tool_call_micro_recall"] == 1.0
    assert report["overall"]["tool_call_micro_matched"] == 0
    assert report["overall"]["tool_call_micro_predicted"] == 0
    assert report["overall"]["tool_call_micro_required"] == 0
