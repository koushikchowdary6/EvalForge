"""Evidence-only precision@k must bound arbitrary positive BM25 cutoff ties."""

import pytest

from evalforge.offline_rag_demo import evaluate


def _doc(identifier, text="alpha"):
    return {"doc_id": identifier, "title": "", "text": text}


def _case(identifier, relevant, *, answerable=True, question="alpha"):
    return {
        "id": identifier, "question": question, "answerable": answerable,
        "relevant_doc_ids": relevant,
    }


def test_mixed_relevance_ties_have_exact_precision_bounds():
    docs = [_doc(name) for name in ("a", "b", "c", "d")]
    report = evaluate(docs, [_case("q1", ["c", "d"])], cutoffs=(2, 3, 4))
    expected = {2: (0.0, 1.0), 3: (1 / 3, 2 / 3), 4: (0.5, 0.5)}
    for k, bounds in expected.items():
        row = next(row for row in report["cases"] if row["top_k"] == k)
        summary = report["by_top_k"][str(k)]
        assert (
            row["lexical_evidence_precision_lower_bound"],
            row["lexical_evidence_precision_upper_bound"],
        ) == pytest.approx(bounds)
        assert (
            summary["lexical_evidence_precision_lower_bound"],
            summary["lexical_evidence_precision_upper_bound"],
        ) == pytest.approx(bounds)
        assert summary["answerable_precision_uncertain_due_to_positive_ties"] == (k < 4)


def test_missing_lexical_evidence_and_unused_slots_do_not_inflate_precision():
    docs = [_doc("a", "alpha"), _doc("b", "beta")]
    report = evaluate(
        docs, [_case("q1", ["a"])], cutoffs=(1, 3),
    )
    row = next(row for row in report["cases"] if row["top_k"] == 3)
    assert row["lexical_evidence_precision_lower_bound"] == pytest.approx(1 / 3)
    assert row["lexical_evidence_precision_upper_bound"] == pytest.approx(1 / 3)

    no_match = evaluate(docs, [_case("q2", ["a"], question="zebra")], cutoffs=(1,))
    assert no_match["cases"][0]["lexical_evidence_precision_upper_bound"] == 0.0


def test_precision_bounds_average_answerable_cases_only():
    docs = [_doc(name) for name in ("a", "b", "c", "d")]
    cases = [
        _case("q1", ["c", "d"]),
        _case("q2", ["a", "b", "c", "d"]),
        _case("q3", [], answerable=False),
    ]
    report = evaluate(docs, cases, cutoffs=(2,))
    summary = report["by_top_k"]["2"]
    assert summary["lexical_evidence_precision_lower_bound"] == pytest.approx(0.5)
    assert summary["lexical_evidence_precision_upper_bound"] == pytest.approx(1.0)
    assert summary["answerable_precision_uncertain_due_to_positive_ties"] == 1
    unanswerable = next(row for row in report["cases"] if row["case_id"] == "q3")
    assert unanswerable["lexical_evidence_precision_lower_bound"] is None
    assert unanswerable["lexical_evidence_precision_upper_bound"] is None


def test_all_unanswerable_precision_bounds_are_undefined():
    report = evaluate(
        [_doc("a")], [_case("q0", [], answerable=False)], cutoffs=(1,),
    )
    summary = report["by_top_k"]["1"]
    assert summary["lexical_evidence_precision_lower_bound"] is None
    assert summary["lexical_evidence_precision_upper_bound"] is None
    assert summary["answerable_precision_uncertain_due_to_positive_ties"] == 0
