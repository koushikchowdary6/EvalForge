"""Evidence-only AP@k must expose exact positive-score BM25 tie uncertainty."""

from itertools import combinations, permutations

import pytest

from evalforge.offline_rag_demo import _positive_score_ap_bounds, evaluate
from evalforge.rag_graders import average_precision_at_k


def _doc(doc_id, text="alpha"):
    return {"doc_id": doc_id, "title": "", "text": text}


def _case(case_id, relevant, *, answerable=True, question="alpha"):
    return {
        "id": case_id, "question": question, "answerable": answerable,
        "relevant_doc_ids": relevant,
    }


def test_ap_bounds_cover_cutoff_ties_and_internal_ties():
    docs = [_doc(doc_id) for doc_id in ("a", "b", "c", "d")]
    report = evaluate(docs, [_case("q", ["c", "d"])], cutoffs=(2, 3, 4))
    expected = {2: (0.0, 1.0), 3: (1 / 6, 1.0), 4: (5 / 12, 1.0)}
    for k, bounds in expected.items():
        row = next(row for row in report["cases"] if row["top_k"] == k)
        summary = report["by_top_k"][str(k)]
        assert (row["lexical_evidence_ap_lower_bound"],
                row["lexical_evidence_ap_upper_bound"]) == pytest.approx(bounds)
        assert (summary["lexical_evidence_ap_lower_bound"],
                summary["lexical_evidence_ap_upper_bound"]) == pytest.approx(bounds)
        assert summary["answerable_ap_uncertain_due_to_positive_ties"] == 1
    # An internal tie changes AP even without a tie crossing the cutoff.
    assert report["cases"][-1]["positive_score_cutoff_tie"] is False


def test_ap_bounds_are_exact_for_distinct_scores_and_no_evidence():
    assert _positive_score_ap_bounds(
        [("a", 3), ("b", 2), ("c", 0)], {"a", "b"}, 3
    ) == pytest.approx((1.0, 1.0))
    assert _positive_score_ap_bounds(
        [("a", 3), ("b", 2), ("c", 0)], {"c"}, 3
    ) == (0.0, 0.0)
    assert _positive_score_ap_bounds(
        [("a", 3), ("b", 2)], set(), 2
    ) == (1.0, 1.0)


def test_ap_bounds_match_exhaustive_permutations_of_tied_scores():
    # Three documents have equal positive scores, one ranks before them,
    # and one has no lexical evidence. Enumerate all relevance subsets,
    # cutoffs, and permissible permutations to check both extrema.
    scored = [("a", 3), ("b", 2), ("c", 2), ("d", 2), ("e", 0)]
    ids = [doc_id for doc_id, _ in scored]
    for count in range(1, len(ids) + 1):
        for relevant_tuple in combinations(ids, count):
            relevant = set(relevant_tuple)
            for k in range(1, len(ids) + 2):
                actual = [
                    average_precision_at_k(["a", *order], relevant, k=k)
                    for order in permutations(("b", "c", "d"))
                ]
                # The zero-score "e" is intentionally omitted: its
                # arbitrary ID order must never earn evidence credit.
                assert _positive_score_ap_bounds(scored, relevant, k) == pytest.approx(
                    (min(actual), max(actual))
                )


def test_ap_bounds_summary_excludes_unanswerable_questions():
    docs = [_doc(doc_id) for doc_id in ("a", "b", "c", "d")]
    report = evaluate(docs, [
        _case("uncertain", ["c", "d"]),
        _case("certain", ["a", "b", "c", "d"]),
        _case("unanswerable", [], answerable=False),
    ], cutoffs=(2,))
    summary = report["by_top_k"]["2"]
    assert summary["lexical_evidence_ap_lower_bound"] == pytest.approx(0.5)
    assert summary["lexical_evidence_ap_upper_bound"] == pytest.approx(1.0)
    assert summary["answerable_ap_uncertain_due_to_positive_ties"] == 1
    row = report["cases"][-1]
    assert row["lexical_evidence_ap_lower_bound"] is None
    assert row["lexical_evidence_ap_upper_bound"] is None

    no_labels = evaluate(docs, [_case("u", [], answerable=False)], cutoffs=(1,))
    summary = no_labels["by_top_k"]["1"]
    assert summary["lexical_evidence_ap_lower_bound"] is None
    assert summary["lexical_evidence_ap_upper_bound"] is None
    assert summary["answerable_ap_uncertain_due_to_positive_ties"] == 0
