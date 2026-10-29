"""Offline tests for case-matched top-k retrieval comparisons."""

from types import SimpleNamespace

from evalforge.rag_runner import RAGEvalRunner, _paired_retrieval_comparisons


def row(case_id, k, hit, recall, *, answerable=True, error=None):
    return {
        "case_id": case_id,
        "top_k": k,
        "answerable": answerable,
        "retrieval_error": error,
        "generation_error": None,
        "retrieval": {
            "hit_rate": None if error else hit,
            "recall_at_k": None if error else recall,
            "precision_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "average_precision_at_k": 0.0,
            "reciprocal_rank": 0.0,
        },
        "abstention": {
            "outcome": "correct_answer" if answerable else "correct_abstain"
        },
        "correctness": {"score": None},
        "groundedness": {"score": None},
        "latency": {"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
    }


def test_pairwise_diagnostics_report_wins_losses_ties_and_exclusions():
    rows = [
        row("improved", 1, 0, 0), row("improved", 3, 1, 0.5),
        row("regressed", 1, 1, 1), row("regressed", 3, 0, 0),
        row("unchanged", 1, 1, 0.5), row("unchanged", 3, 1, 0.5),
        row("outage", 1, 0, 0), row("outage", 3, 0, 0, error="down"),
        row("unanswerable", 1, 1, 1, answerable=False),
        row("unanswerable", 3, 1, 1, answerable=False),
        row("missing", 1, 0, 0),
        row("duplicate", 1, 0, 0), row("duplicate", 1, 0, 0),
        row("duplicate", 3, 1, 1),
    ]
    stats = _paired_retrieval_comparisons(rows)["1_to_3"]
    assert stats["paired_answerable_queries"] == 3
    assert stats["excluded_unpaired_or_failed"] == 4
    assert stats["hit_rate"] == {"improved": 1, "regressed": 1, "unchanged": 1}
    assert stats["recall_at_k"] == {"improved": 1, "regressed": 1, "unchanged": 1}
    assert stats["regressed_case_ids"] == {
        "hit_rate": ["regressed"], "recall_at_k": ["regressed"]
    }


def test_pairwise_excludes_invalid_metrics_without_false_regression():
    rows = [
        row("invalid", 1, float("nan"), 0),
        row("invalid", 3, 1, 1),
        row("boolean", 1, True, 0),
        row("boolean", 3, 1, 1),
    ]
    stats = _paired_retrieval_comparisons(rows)["1_to_3"]
    assert stats["paired_answerable_queries"] == 0
    assert stats["excluded_unpaired_or_failed"] == 2
    assert stats["regressed_case_ids"] == {"hit_rate": [], "recall_at_k": []}


def test_report_includes_all_pairwise_cutoffs_without_judge_calls():
    rows = [
        row("case", 1, 0, 0),
        row("case", 3, 1, 0.5),
        row("case", 5, 1, 1),
    ]
    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.model = "offline-test"
    runner.questions = [{"id": "case"}]
    runner.results = rows
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    report = runner.generate_report()
    pairs = report["paired_retrieval_comparisons"]
    assert set(pairs) == {"1_to_3", "1_to_5", "3_to_5"}
    assert pairs["1_to_3"]["hit_rate"]["improved"] == 1
    assert pairs["3_to_5"]["hit_rate"]["unchanged"] == 1
    assert pairs["3_to_5"]["recall_at_k"]["improved"] == 1
    assert all(item["paired_answerable_queries"] == 1 for item in pairs.values())
