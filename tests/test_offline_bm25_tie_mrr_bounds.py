"""Evidence-only MRR must expose uncertainty from positive-score BM25 ties."""

import pytest

from evalforge.offline_rag_demo import _positive_score_mrr_bounds, evaluate


def _doc(doc_id, text="alpha"):
    return {"doc_id": doc_id, "title": "", "text": text}


def _case(case_id, relevant, *, answerable=True, question="alpha"):
    return {
        "id": case_id, "question": question, "answerable": answerable,
        "relevant_doc_ids": relevant,
    }


def test_mrr_bounds_include_internal_ties_even_without_cutoff_tie():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    report = evaluate(docs, [_case("q", ["c"])], cutoffs=(1, 2, 3))
    for k in (1, 2, 3):
        row = next(row for row in report["cases"] if row["top_k"] == k)
        assert row["lexical_evidence_mrr_lower_bound"] == pytest.approx(
            1 / 3 if k == 3 else 0
        )
        assert row["lexical_evidence_mrr_upper_bound"] == pytest.approx(1)
        summary = report["by_top_k"][str(k)]
        assert summary["lexical_evidence_mrr_lower_bound"] == pytest.approx(
            1 / 3 if k == 3 else 0
        )
        assert summary["lexical_evidence_mrr_upper_bound"] == pytest.approx(1)
        assert summary["answerable_mrr_uncertain_due_to_positive_ties"] == 1
    assert report["cases"][-1]["positive_score_cutoff_tie"] is False


def test_mrr_bounds_respect_score_groups_and_zero_score_evidence():
    assert _positive_score_mrr_bounds(
        [("a", 3), ("b", 2), ("c", 2), ("d", 0)], ["c"], 3
    ) == pytest.approx((1 / 3, 1 / 2))
    assert _positive_score_mrr_bounds(
        [("a", 3), ("b", 2), ("c", 2)], ["a", "c"], 3
    ) == pytest.approx((1, 1))
    assert _positive_score_mrr_bounds(
        [("a", 3), ("b", 2), ("c", 2)], ["c"], 2
    ) == pytest.approx((0, 1 / 2))
    assert _positive_score_mrr_bounds(
        [("a", 3), ("b", 0)], ["b"], 2
    ) == (0, 0)


def test_mrr_bounds_are_exact_when_scores_are_distinct():
    docs = [_doc("a", "alpha alpha"), _doc("b", "beta")]
    report = evaluate(docs, [_case("q", ["a"])], cutoffs=(1, 2))
    for row in report["cases"]:
        assert row["lexical_evidence_mrr_lower_bound"] == 1
        assert row["lexical_evidence_mrr_upper_bound"] == 1


def test_mrr_summary_excludes_unanswerable_cases():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    report = evaluate(docs, [
        _case("uncertain", ["c"]),
        _case("certain", ["a", "b", "c"]),
        _case("unanswerable", [], answerable=False),
    ], cutoffs=(3,))
    summary = report["by_top_k"]["3"]
    assert summary["lexical_evidence_mrr_lower_bound"] == pytest.approx(2 / 3)
    assert summary["lexical_evidence_mrr_upper_bound"] == pytest.approx(1)
    assert summary["answerable_mrr_uncertain_due_to_positive_ties"] == 1
    row = report["cases"][-1]
    assert row["lexical_evidence_mrr_lower_bound"] is None
    assert row["lexical_evidence_mrr_upper_bound"] is None

    no_labels = evaluate(docs, [_case("u", [], answerable=False)], cutoffs=(1,))
    assert no_labels["by_top_k"]["1"]["lexical_evidence_mrr_lower_bound"] is None
    assert no_labels["by_top_k"]["1"]["lexical_evidence_mrr_upper_bound"] is None
    assert no_labels["by_top_k"]["1"]["answerable_mrr_uncertain_due_to_positive_ties"] == 0
