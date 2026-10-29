"""Offline tests for honest uncertainty reporting on small RAG benchmarks."""

from types import SimpleNamespace

import pytest

from evalforge.rag_runner import RAGEvalRunner, _wilson_hit_rate_interval


def _row(*, answerable=True, hit=0.0, retrieval_error=None):
    return {
        "top_k": 3,
        "answerable": answerable,
        "retrieval_error": retrieval_error,
        "generation_error": None,
        "retrieval": {
            "precision_at_k": hit / 3,
            "recall_at_k": hit,
            "ndcg_at_k": hit,
            "average_precision_at_k": hit,
            "reciprocal_rank": hit,
            "hit_rate": hit if retrieval_error is None else None,
        },
        "abstention": {
            "outcome": "correct_answer" if answerable else "correct_abstain"
        },
        "correctness": {"score": None},
        "groundedness": {"score": None},
        "latency": {"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
    }


def _report(rows):
    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.model = "offline-test"
    runner.questions = [{} for _ in rows]
    runner.results = rows
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    return runner.generate_report()["by_top_k"]["3"]["retrieval"]


def test_wilson_interval_handles_no_samples_and_extreme_hit_rates():
    assert _wilson_hit_rate_interval(0, 0) is None
    assert _wilson_hit_rate_interval(0, 1)["lower"] == 0.0
    assert _wilson_hit_rate_interval(1, 1)["upper"] == 1.0
    assert _wilson_hit_rate_interval(1, 1)["lower"] < 1.0
    assert _wilson_hit_rate_interval(0, 1)["upper"] > 0.0
    half = _wilson_hit_rate_interval(5, 10)
    assert half["lower"] == pytest.approx(1 - half["upper"])
    assert half["lower"] < 0.5 < half["upper"]


def test_report_interval_uses_only_answerable_successful_retrievals():
    rows = [
        _row(hit=1.0),
        _row(hit=0.0),
        _row(answerable=False, hit=1.0),  # no relevant documents
        _row(hit=0.0, retrieval_error="retrieval outage"),
    ]
    retrieval = _report(rows)
    assert retrieval["evaluated_queries"] == 2
    assert retrieval["excluded_unanswerable"] == 1
    assert retrieval["errors"] == 1
    assert retrieval["hit_rate"] == 0.5
    assert retrieval["hit_rate_95ci"] == _wilson_hit_rate_interval(1, 2)


def test_report_has_no_interval_without_answerable_retrieval_samples():
    retrieval = _report([
        _row(answerable=False, hit=1.0),
        _row(hit=0.0, retrieval_error="retrieval outage"),
    ])
    assert retrieval["evaluated_queries"] == 0
    assert retrieval["hit_rate"] is None
    assert retrieval["hit_rate_95ci"] is None
