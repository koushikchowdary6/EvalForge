"""Safety-rate uncertainty must reflect only actually evaluated RAG answers."""

from types import SimpleNamespace

from evalforge.rag_runner import RAGEvalRunner, _wilson_hit_rate_interval


def row(answerable, outcome, *, retrieval_error=None, generation_error=None):
    return {
        "top_k": 3,
        "answerable": answerable,
        "retrieval_error": retrieval_error,
        "generation_error": generation_error,
        "retrieval": {
            "precision_at_k": None if retrieval_error else 0.0,
            "recall_at_k": None if retrieval_error else 0.0,
            "ndcg_at_k": None if retrieval_error else 0.0,
            "average_precision_at_k": None if retrieval_error else 0.0,
            "reciprocal_rank": None if retrieval_error else 0.0,
            "hit_rate": None if retrieval_error else 0.0,
        },
        "abstention": {"outcome": outcome},
        "correctness": {"score": None},
        "groundedness": {"score": None},
        "latency": {"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
    }


def safety_report(rows):
    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.model = "offline-test"
    runner.questions = [{} for _ in rows]
    runner.results = rows
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    return runner.generate_report()["by_top_k"]["3"]["hallucination"]


def test_safety_intervals_exclude_failed_and_empty_generations():
    safety = safety_report([
        row(False, "hallucination"),
        row(False, "correct_abstain"),
        row(False, "empty_answer"),
        row(False, "generation_error", generation_error="timeout"),
        row(False, "generation_error", retrieval_error="retriever down"),
        row(True, "over_abstention"),
        row(True, "correct_answer"),
        row(True, "empty_answer"),
        row(True, "generation_error", generation_error="timeout"),
    ])
    assert safety["hallucination_evaluated"] == 2
    assert safety["hallucinations"] == 1
    assert safety["hallucination_rate"] == 0.5
    assert safety["hallucination_rate_95ci"] == _wilson_hit_rate_interval(1, 2)
    assert safety["over_abstention_evaluated"] == 2
    assert safety["over_abstentions"] == 1
    assert safety["over_abstention_rate"] == 0.5
    assert safety["over_abstention_rate_95ci"] == _wilson_hit_rate_interval(1, 2)


def test_safety_intervals_are_null_without_eligible_samples():
    safety = safety_report([
        row(False, "empty_answer"),
        row(True, "empty_answer"),
        row(False, "generation_error", retrieval_error="retriever down"),
    ])
    assert safety["hallucination_rate"] is None
    assert safety["hallucination_rate_95ci"] is None
    assert safety["over_abstention_rate"] is None
    assert safety["over_abstention_rate_95ci"] is None


def test_zero_and_all_event_intervals_do_not_claim_certainty():
    safety = safety_report([
        row(False, "correct_abstain"),
        row(True, "over_abstention"),
    ])
    assert safety["hallucination_rate"] == 0.0
    assert safety["hallucination_rate_95ci"]["upper"] > 0.0
    assert safety["over_abstention_rate"] == 1.0
    assert safety["over_abstention_rate_95ci"]["lower"] < 1.0
