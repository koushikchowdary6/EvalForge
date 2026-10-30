"""Distinguish lexical retrieval evidence from arbitrary BM25 zero-score ties."""
import pytest

from evalforge.offline_rag_demo import evaluate


DOCS = [
    {"doc_id": "a", "title": "battery", "text": "voltage monitor"},
    {"doc_id": "b", "title": "security", "text": "policy checks"},
    {"doc_id": "c", "title": "temperature", "text": "sensor data"},
]


def _question(case_id, query, relevant):
    return {
        "id": case_id,
        "question": query,
        "answerable": bool(relevant),
        "relevant_doc_ids": relevant,
    }


def test_zero_score_tie_cannot_claim_evidence_backed_hit():
    report = evaluate(DOCS, [_question("no_overlap", "galactic nebula", ["a"])], (1,))
    case = report["cases"][0]
    assert case["retrieved_ids"] == ["a"]  # deterministic ID-order fallback
    assert case["metrics"]["hit_rate"] == 1.0  # legacy metric remains available
    assert case["positive_score_retrieved_ids"] == []
    assert case["lexical_evidence_metrics"] == {
        "hit_rate": 0.0,
        "recall_at_k": 0.0,
        "reciprocal_rank": 0.0,
    }
    assert report["by_top_k"]["1"]["tie_only_hit_count"] == 1
    assert report["by_top_k"]["1"]["lexical_evidence_hit_rate"] == 0.0


def test_zero_score_relevant_doc_behind_real_match_is_not_credited():
    report = evaluate(DOCS, [_question("false_hit", "battery", ["c"])], (3,))
    case = report["cases"][0]
    assert case["retrieved_ids"][0] == "a"
    assert "c" in case["retrieved_ids"]
    assert case["positive_score_retrieved_ids"] == ["a"]
    assert case["metrics"]["hit_rate"] == 1.0
    assert case["lexical_evidence_metrics"]["hit_rate"] == 0.0
    assert report["by_top_k"]["3"]["tie_only_hit_count"] == 1


def test_true_lexical_hit_and_no_overlap_are_aggregated_separately():
    questions = [
        _question("matched", "battery", ["a"]),
        _question("unmatched", "galactic nebula", ["a"]),
        _question("unanswerable", "fictional question", []),
    ]
    report = evaluate(DOCS, questions, (1,))
    summary = report["by_top_k"]["1"]
    assert summary["answerable_count"] == 2
    assert summary["unanswerable_excluded"] == 1
    assert summary["lexical_evidence_hit_rate"] == pytest.approx(0.5)
    assert summary["lexical_evidence_recall_at_k"] == pytest.approx(0.5)
    assert summary["lexical_evidence_mrr"] == pytest.approx(0.5)
    assert summary["tie_only_hit_count"] == 1
    assert report["cases"][2]["lexical_evidence_metrics"] is None
