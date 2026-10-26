"""Outages must not make agent performance look faster or safer."""

from evalforge.agent_runner import AgentEvalRunner


def _row(category, *, error=None, latency_ms=100, turns=2,
         injection=None, destructive=None):
    return {
        "category": category,
        "error": error,
        "grade": {
            "success": error is None,
            "tool_selection_score": 1.0 if error is None else None,
            "tool_sequence_score": 1.0 if error is None else None,
            "argument_accuracy": None,
            "unnecessary_calls": [],
            "failure_category": "execution_error" if error else None,
        },
        "injection_resisted": injection,
        "refused_destructive": destructive,
        "latency_ms": latency_ms,
        "turns": turns,
        "cost_usd": 0.0,
    }


def test_outages_do_not_lower_completed_agent_latency_or_turn_means(tmp_path):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [
        _row("injection_resistance", latency_ms=200, turns=4, injection=True),
        _row("injection_resistance", error="provider unavailable",
             latency_ms=1, turns=0, injection=None),
        # A forbidden call observed before an outage is still a safety failure.
        _row("injection_resistance", error="provider unavailable",
             latency_ms=2, turns=1, injection=False),
        _row("destructive_refusal", latency_ms=100, turns=2, destructive=False),
        _row("destructive_refusal", error="provider unavailable",
             latency_ms=1, turns=0, destructive=None),
    ]
    report = runner.generate_report()
    overall = report["overall"]
    assert report["evaluated_tasks"] == 2
    assert report["execution_errors"] == 3
    assert overall["latency_samples"] == overall["turn_samples"] == 2
    assert overall["avg_latency_ms"] == 150
    assert overall["avg_turns"] == 3
    assert report["injection_resistance_evaluated"] == 2
    assert report["destructive_refusal_evaluated"] == 1
    assert overall["injection_resistance_rate"] == 0.5
    assert overall["destructive_refusal_rate"] == 0.0

    category = report["by_category"]["injection_resistance"]
    assert category["tasks"] == 3
    assert category["latency_samples"] == 1
    assert category["avg_latency_ms"] == 200
    assert report["by_category"]["destructive_refusal"]["avg_latency_ms"] == 100


def test_outage_only_category_has_no_success_latency(tmp_path, capsys):
    runner = AgentEvalRunner(tmp_path / "tasks.json", output_dir=tmp_path)
    runner.results = [
        _row("injection_resistance", error="timeout", latency_ms=1,
             turns=0, injection=None),
    ]
    report = runner.generate_report()
    assert report["overall"]["latency_samples"] == 0
    assert report["overall"]["avg_latency_ms"] is None
    assert report["overall"]["avg_turns"] is None
    assert report["injection_resistance_evaluated"] == 0
    assert report["overall"]["injection_resistance_rate"] is None
    assert report["by_category"]["injection_resistance"]["latency_samples"] == 0
    assert report["by_category"]["injection_resistance"]["avg_latency_ms"] is None

    runner.print_report()
    output = capsys.readouterr().out
    assert "Avg latency ................ n/a" in output
    assert "Latency samples ............ 0/1" in output
    assert "Injection safety observed .. 0" in output
