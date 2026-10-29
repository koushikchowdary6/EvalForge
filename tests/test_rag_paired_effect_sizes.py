"""Offline checks for reproducible, case-paired RAG effect-size reporting."""

import pytest

from evalforge.rag_runner import _paired_mean_bootstrap, _paired_retrieval_comparisons


def row(case_id, k, hit, recall, *, error=None):
    return {
        "case_id": case_id,
        "top_k": k,
        "answerable": True,
        "retrieval_error": error,
        "retrieval": {"hit_rate": hit, "recall_at_k": recall},
    }


def test_paired_effects_use_within_case_differences_not_unmatched_rows():
    rows = [
        row("a", 1, 0, 0), row("a", 3, 1, 0.5),
        row("b", 1, 1, 1), row("b", 3, 1, 1),
        row("c", 1, 0, 0.5), row("c", 3, 1, 1),
        row("unpaired", 1, 0, 0),
        row("failed", 1, 0, 0), row("failed", 3, 1, 1, error="outage"),
    ]
    stats = _paired_retrieval_comparisons(rows)["1_to_3"]
    assert stats["paired_answerable_queries"] == 3
    assert stats["excluded_unpaired_or_failed"] == 2
    assert stats["paired_effects"]["hit_rate"]["mean_delta"] == pytest.approx(2 / 3)
    assert stats["paired_effects"]["recall_at_k"]["mean_delta"] == pytest.approx(1 / 3)
    for metric in ("hit_rate", "recall_at_k"):
        bounds = stats["paired_effects"][metric]["bootstrap_95ci"]
        assert bounds["lower"] <= stats["paired_effects"][metric]["mean_delta"] <= bounds["upper"]


def test_bootstrap_is_deterministic_and_reports_insufficient_sample_size():
    assert _paired_mean_bootstrap([]) == {"mean_delta": None, "bootstrap_95ci": None}
    assert _paired_mean_bootstrap([0.5]) == {
        "mean_delta": 0.5, "bootstrap_95ci": None
    }
    first = _paired_mean_bootstrap([0.0, 0.5, 1.0, 0.5])
    assert first == _paired_mean_bootstrap([0.0, 0.5, 1.0, 0.5])
    assert first["mean_delta"] == 0.5
    assert 0 <= first["bootstrap_95ci"]["lower"] <= 0.5
    assert 0.5 <= first["bootstrap_95ci"]["upper"] <= 1.0


def test_all_excluded_pairs_have_null_effect_sizes():
    stats = _paired_retrieval_comparisons([
        row("bad", 1, 0, 0),
        row("bad", 3, 1, 1, error="down"),
    ])["1_to_3"]
    assert stats["paired_answerable_queries"] == 0
    assert all(
        effect == {"mean_delta": None, "bootstrap_95ci": None}
        for effect in stats["paired_effects"].values()
    )
