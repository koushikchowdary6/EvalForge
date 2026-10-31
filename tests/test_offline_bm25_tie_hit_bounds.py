"""Positive BM25 cutoff ties must not masquerade as certain retrieval hits."""

from evalforge.offline_rag_demo import _positive_score_hit_bounds, evaluate


def _doc(doc_id, text):
    return {"doc_id": doc_id, "title": "", "text": text}


def _case(relevant):
    return {
        "id": "q1", "question": "alpha", "answerable": True,
        "relevant_doc_ids": relevant,
    }


def test_tie_bounds_expose_id_order_uncertainty():
    docs = [_doc("a", "alpha"), _doc("b", "alpha"), _doc("c", "alpha")]
    report = evaluate(docs, [_case(["c"])], cutoffs=(1, 2, 3))
    for k in (1, 2):
        row = next(row for row in report["cases"] if row["top_k"] == k)
        assert row["positive_score_cutoff_tie"] is True
        assert row["lexical_evidence_hit_lower_bound"] == 0
        assert row["lexical_evidence_hit_upper_bound"] == 1
        summary = report["by_top_k"][str(k)]
        assert summary["lexical_evidence_hit_lower_bound"] == 0
        assert summary["lexical_evidence_hit_upper_bound"] == 1
        assert summary["answerable_hit_uncertain_due_to_positive_ties"] == 1

    final = next(row for row in report["cases"] if row["top_k"] == 3)
    assert final["lexical_evidence_hit_lower_bound"] == 1
    assert final["lexical_evidence_hit_upper_bound"] == 1
    assert report["by_top_k"]["3"]["answerable_hit_uncertain_due_to_positive_ties"] == 0


def test_tie_bound_is_certain_when_every_candidate_is_relevant():
    scored = [("a", 2.0), ("b", 2.0), ("c", 2.0)]
    assert _positive_score_hit_bounds(scored, ["a", "b", "c"], 1) == (1, 1)
    assert _positive_score_hit_bounds(scored, ["c"], 1) == (0, 1)


def test_higher_ranked_relevant_hit_is_not_affected_by_cutoff_tie():
    scored = [("a", 3.0), ("b", 2.0), ("c", 2.0)]
    assert _positive_score_hit_bounds(scored, ["a"], 2) == (1, 1)


def test_zero_score_ids_never_count_as_evidence():
    scored = [("a", 0.0), ("b", 0.0)]
    assert _positive_score_hit_bounds(scored, ["a"], 1) == (0, 0)


def test_unanswerable_cases_keep_hit_bounds_undefined():
    docs = [_doc("a", "alpha"), _doc("b", "alpha")]
    cases = [{"id": "q0", "question": "alpha", "answerable": False, "relevant_doc_ids": []}]
    report = evaluate(docs, cases, cutoffs=(1,))
    assert report["cases"][0]["lexical_evidence_hit_lower_bound"] is None
    assert report["cases"][0]["lexical_evidence_hit_upper_bound"] is None
    assert report["by_top_k"]["1"]["lexical_evidence_hit_lower_bound"] is None
    assert report["by_top_k"]["1"]["lexical_evidence_hit_upper_bound"] is None
