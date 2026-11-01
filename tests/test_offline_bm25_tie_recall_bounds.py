"""Evidence-only BM25 recall must reflect uncertainty at positive-score ties."""

import pytest

from evalforge.offline_rag_demo import (
    _positive_score_relevant_count_bounds,
    evaluate,
)


def _doc(identifier, text):
    return {"doc_id": identifier, "title": "", "text": text}


def _case(identifier, relevant, answerable=True, question="alpha"):
    return {
        "id": identifier,
        "question": question,
        "answerable": answerable,
        "relevant_doc_ids": relevant,
    }


def test_partial_relevance_tie_has_exact_recall_bounds():
    docs = [_doc(name, "alpha") for name in ("a", "b", "c", "d")]
    report = evaluate(docs, [_case("q1", ["c", "d"])], cutoffs=(2, 3, 4))
    expected = {2: (0.0, 1.0), 3: (0.5, 1.0), 4: (1.0, 1.0)}
    for k, bounds in expected.items():
        row = next(row for row in report["cases"] if row["top_k"] == k)
        summary = report["by_top_k"][str(k)]
        assert (row["lexical_evidence_recall_lower_bound"],
                row["lexical_evidence_recall_upper_bound"]) == bounds
        assert (summary["lexical_evidence_recall_lower_bound"],
                summary["lexical_evidence_recall_upper_bound"]) == bounds
        assert summary["answerable_recall_uncertain_due_to_positive_ties"] == (k < 4)


def test_fixed_higher_score_relevance_is_counted_in_both_bounds():
    scored = [("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 0.0)]
    assert _positive_score_relevant_count_bounds(scored, ["a", "c"], 2) == (1, 2)
    assert _positive_score_relevant_count_bounds(scored, ["a", "c"], 3) == (2, 2)


def test_zero_score_results_cannot_inflate_recall():
    scored = [("a", 0.0), ("b", 0.0)]
    assert _positive_score_relevant_count_bounds(scored, ["a"], 1) == (0, 0)
    report = evaluate([_doc("a", "alpha")], [_case("q1", ["a"], question="zebra")], cutoffs=(1,))
    assert report["by_top_k"]["1"]["lexical_evidence_recall_upper_bound"] == 0


def test_aggregate_recall_bounds_exclude_unanswerable_cases():
    docs = [_doc(name, "alpha") for name in ("a", "b", "c", "d")]
    cases = [
        _case("q1", ["c", "d"]),
        _case("q2", ["a", "b", "c", "d"]),
        _case("q3", [], answerable=False),
    ]
    report = evaluate(docs, cases, cutoffs=(2,))
    summary = report["by_top_k"]["2"]
    assert summary["lexical_evidence_recall_lower_bound"] == pytest.approx(0.25)
    assert summary["lexical_evidence_recall_upper_bound"] == pytest.approx(0.75)
    assert summary["answerable_recall_uncertain_due_to_positive_ties"] == 1
    row = next(row for row in report["cases"] if row["case_id"] == "q3")
    assert row["lexical_evidence_recall_lower_bound"] is None
    assert row["lexical_evidence_recall_upper_bound"] is None


def test_all_unanswerable_reports_undefined_recall_bounds():
    report = evaluate(
        [_doc("a", "alpha")], [_case("q0", [], answerable=False)], cutoffs=(1,),
    )
    summary = report["by_top_k"]["1"]
    assert summary["lexical_evidence_recall_lower_bound"] is None
    assert summary["lexical_evidence_recall_upper_bound"] is None
    assert summary["answerable_recall_uncertain_due_to_positive_ties"] == 0
