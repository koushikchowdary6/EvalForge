"""Latency summaries must not reward failed retrieval or generation stages."""

from types import SimpleNamespace

from evalforge.rag_runner import RAGEvalRunner


def _row(case_id, *, retrieval_error=None, generation_error=None,
         retrieval_ms=10.0, generation_ms=50.0):
    successful = retrieval_error is None and generation_error is None
    return {
        "case_id": case_id,
        "top_k": 3,
        "answerable": True,
        "retrieval_error": retrieval_error,
        "generation_error": generation_error,
        "retrieval": {
            key: None if retrieval_error else 1.0
            for key in (
                "precision_at_k", "recall_at_k", "ndcg_at_k",
                "average_precision_at_k", "reciprocal_rank", "hit_rate",
            )
        },
        "abstention": {
            "outcome": "correct_answer" if successful else "generation_error",
        },
        "correctness": {"score": 0.8 if successful else None, "parse_ok": successful},
        "groundedness": {"score": 0.9 if successful else None, "parse_ok": successful},
        "latency": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": retrieval_ms + generation_ms,
        },
    }


def _runner(rows):
    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.model = "claude"
    runner.questions = []
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.results = rows
    runner.report = None
    return runner


def test_failed_stages_do_not_artificially_improve_latency_averages():
    runner = _runner([
        _row("retrieval-outage", retrieval_error="retrieval failed", retrieval_ms=40, generation_ms=0),
        _row("generation-outage", generation_error="generation failed", retrieval_ms=20, generation_ms=2),
        _row("completed", retrieval_ms=10, generation_ms=50),
    ])

    stats = runner.generate_report()["by_top_k"]["3"]
    latency = stats["latency"]
    assert stats["retrieval"]["errors"] == 1
    assert stats["generation"]["errors"] == 1
    assert latency["retrieval_samples"] == 2
    assert latency["generation_samples"] == 1
    assert latency["end_to_end_samples"] == 1
    assert latency["avg_retrieval_ms"] == 15
    assert latency["avg_generation_ms"] == 50
    assert latency["avg_total_ms"] == 60


def test_all_failed_runs_have_no_success_latency_and_print_safely(capsys):
    runner = _runner([
        _row("retrieval-outage", retrieval_error="retrieval failed", retrieval_ms=40, generation_ms=0),
    ])
    latency = runner.generate_report()["by_top_k"]["3"]["latency"]

    assert latency["retrieval_samples"] == 0
    assert latency["generation_samples"] == 0
    assert latency["end_to_end_samples"] == 0
    assert latency["avg_retrieval_ms"] is None
    assert latency["avg_generation_ms"] is None
    assert latency["avg_total_ms"] is None

    runner.print_report()
    assert "n/a" in capsys.readouterr().out
