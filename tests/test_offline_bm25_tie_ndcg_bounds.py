"""Exact binary nDCG bounds must not depend on BM25 document-ID tie order."""

import math

import pytest

from evalforge.offline_rag_demo import evaluate


def _doc(identifier, text="alpha"):
    return {"doc_id": identifier, "title": "", "text": text}


def _case(identifier, relevant, *, answerable=True, question="alpha"):
    return {
        "id": identifier, "question": question, "answerable": answerable,
        "relevant_doc_ids": relevant,
    }


def test_ndcg_bounds_cover_cutoff_and_internal_positive_score_ties():
    docs = [_doc(name) for name in ("a", "b", "c", "d")]
    report = evaluate(docs, [_case("q", ["c", "d"])], cutoffs=(2, 3, 4))
    ideal_two = 1 + 1 / math.log2(3)
    expected = {
        2: (0.0, 1.0),
        3: ((1 / math.log2(4)) / ideal_two, 1.0),
        4: ((1 / math.log2(4) + 1 / math.log2(5)) / ideal_two, 1.0),
    }
    for k, bounds in expected.items():
        row = next(row for row in report["cases"] if row["top_k"] == k)
        summary = report["by_top_k"][str(k)]
        assert (row["lexical_evidence_ndcg_lower_bound"],
                row["lexical_evidence_ndcg_upper_bound"]) == pytest.approx(bounds)
        assert (summary["lexical_evidence_ndcg_lower_bound"],
                summary["lexical_evidence_ndcg_upper_bound"]) == pytest.approx(bounds)
        assert summary["answerable_ndcg_uncertain_due_to_positive_ties"] == 1
    # Even k=4, with no tie straddling the cutoff, ranking *within* the
    # positive-score block is arbitrary and can change discounted gain.
    assert report["cases"][-1]["positive_score_cutoff_tie"] is False


def test_ndcg_bounds_are_exact_without_ties_or_lexical_matches():
    docs = [_doc("a", "alpha"), _doc("b", "beta")]
    report = evaluate(docs, [_case("q", ["a"])], cutoffs=(1, 3))
    for row in report["cases"]:
        assert row["lexical_evidence_ndcg_lower_bound"] == pytest.approx(1.0)
        assert row["lexical_evidence_ndcg_upper_bound"] == pytest.approx(1.0)
    no_match = evaluate(docs, [_case("q", ["a"], question="zebra")], cutoffs=(1,))
    row = no_match["cases"][0]
    assert row["lexical_evidence_ndcg_lower_bound"] == 0.0
    assert row["lexical_evidence_ndcg_upper_bound"] == 0.0


def test_ndcg_bounds_average_only_answerable_cases():
    docs = [_doc(name) for name in ("a", "b", "c", "d")]
    report = evaluate(docs, [
        _case("uncertain", ["c", "d"]),
        _case("all-relevant", ["a", "b", "c", "d"]),
        _case("unanswerable", [], answerable=False),
    ], cutoffs=(2,))
    summary = report["by_top_k"]["2"]
    assert summary["lexical_evidence_ndcg_lower_bound"] == pytest.approx(0.5)
    assert summary["lexical_evidence_ndcg_upper_bound"] == pytest.approx(1.0)
    assert summary["answerable_ndcg_uncertain_due_to_positive_ties"] == 1
    unanswerable = report["cases"][-1]
    assert unanswerable["lexical_evidence_ndcg_lower_bound"] is None
    assert unanswerable["lexical_evidence_ndcg_upper_bound"] is None


def test_all_unanswerable_ndcg_bounds_are_undefined():
    report = evaluate([_doc("a")], [_case("q", [], answerable=False)], cutoffs=(1,))
    summary = report["by_top_k"]["1"]
    assert summary["lexical_evidence_ndcg_lower_bound"] is None
    assert summary["lexical_evidence_ndcg_upper_bound"] is None
    assert summary["answerable_ndcg_uncertain_due_to_positive_ties"] == 0
