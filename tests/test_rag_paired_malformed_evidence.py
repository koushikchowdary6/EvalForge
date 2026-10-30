"""Offline regression coverage for malformed persisted paired retrieval evidence."""

from evalforge.rag_runner import _paired_retrieval_comparisons


def _row(case_id, top_k, retrieval):
    return {
        "case_id": case_id,
        "top_k": top_k,
        "answerable": True,
        "retrieval_error": None,
        "retrieval": retrieval,
    }


def test_malformed_retrieval_objects_and_huge_counts_are_excluded():
    valid_low = {"hit_rate": 0, "recall_at_k": 0.0}
    valid_high = {"hit_rate": 1, "recall_at_k": 0.5}
    rows = [
        _row("missing", 1, valid_low), _row("missing", 3, None),
        _row("list", 1, []), _row("list", 3, valid_high),
        _row("huge", 1, {"hit_rate": 10**400, "recall_at_k": 0}),
        _row("huge", 3, valid_high),
        _row("valid", 1, valid_low), _row("valid", 3, valid_high),
    ]
    stats = _paired_retrieval_comparisons(rows)["1_to_3"]
    assert stats["paired_answerable_queries"] == 1
    assert stats["excluded_unpaired_or_failed"] == 3
    assert stats["hit_rate"] == {"improved": 1, "regressed": 0, "unchanged": 0}
    assert stats["recall_at_k"] == {"improved": 1, "regressed": 0, "unchanged": 0}
    assert stats["paired_effects"]["hit_rate"] == {
        "mean_delta": 1.0, "bootstrap_95ci": None
    }


def test_missing_metric_and_nonfinite_scores_remain_excluded():
    rows = [
        _row("missing_metric", 1, {"hit_rate": 0}),
        _row("missing_metric", 3, {"hit_rate": 1, "recall_at_k": 1}),
        _row("nan", 1, {"hit_rate": float("nan"), "recall_at_k": 0}),
        _row("nan", 3, {"hit_rate": 1, "recall_at_k": 1}),
    ]
    stats = _paired_retrieval_comparisons(rows)["1_to_3"]
    assert stats["paired_answerable_queries"] == 0
    assert stats["excluded_unpaired_or_failed"] == 2
    assert stats["regressed_case_ids"] == {"hit_rate": [], "recall_at_k": []}
