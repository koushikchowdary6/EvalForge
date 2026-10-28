"""Agent benchmark cost estimates must preserve malformed-usage uncertainty."""

import math

import pytest

from evalforge.agent_graders import estimate_cost
from evalforge.agent_runner import AgentEvalRunner


def test_valid_token_usage_keeps_existing_provider_pricing():
    assert estimate_cost({"input": 1_000_000, "output": 1_000_000}, "claude") == 6.0
    assert estimate_cost({"input": 0, "output": 0}, "gpt") == 0.0
    assert estimate_cost({}, "claude") == 0.0


@pytest.mark.parametrize("tokens", [
    None, [], {"input": -1}, {"output": -5},
    {"input": True}, {"output": False},
    {"input": 3.5}, {"output": float("nan")},
    {"input": float("inf")}, {"output": "10"},
    {"input": 10 ** 5000},
])
def test_invalid_usage_cannot_become_a_numeric_cost(tokens):
    assert estimate_cost(tokens, "claude") is None


def test_unknown_model_pricing_is_unavailable():
    assert estimate_cost({"input": 100, "output": 100}, "unknown-model") is None


def _result(cost):
    return {
        "category": "usage", "error": None,
        "grade": {
            "success": True, "tool_selection_score": 1.0,
            "argument_accuracy": None, "unnecessary_calls": [],
            "failure_category": None,
        },
        "injection_resisted": None, "refused_destructive": None,
        "latency_ms": 10, "turns": 1, "cost_usd": cost,
    }


def test_partial_usage_reports_coverage_without_claiming_total(tmp_path, capsys):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [_result(0.012), _result(None)]
    report = runner.generate_report()
    overall = report["overall"]
    assert overall["cost_samples"] == 1
    assert overall["observed_cost_usd"] == pytest.approx(0.012)
    assert overall["total_cost_usd"] is None
    runner.print_report()
    output = capsys.readouterr().out
    assert "Estimated cost ............. n/a (incomplete usage)" in output
    assert "Cost coverage .............. 1/2" in output
    assert "Observed partial cost ...... $0.0120" in output


def test_complete_usage_reports_full_total(tmp_path):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [_result(0.012), _result(0.0)]
    overall = runner.generate_report()["overall"]
    assert overall["cost_samples"] == 2
    assert overall["total_cost_usd"] == pytest.approx(0.012)
    assert math.isfinite(overall["total_cost_usd"])
