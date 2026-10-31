"""Regression coverage for positive-score BM25 ties crossing the top-k boundary."""
from evalforge.offline_rag_demo import evaluate


DOCS = [
    {"doc_id": "a", "title": "", "text": "alpha beta"},
    {"doc_id": "z", "title": "", "text": "beta alpha"},
    {"doc_id": "x", "title": "", "text": "gamma"},
]


def question(case_id, text, relevant):
    return {
        "id": case_id, "question": text, "answerable": bool(relevant),
        "relevant_doc_ids": relevant,
    }


def test_positive_score_tie_exposes_arbitrary_cutoff_and_mixed_labels():
    report = evaluate(DOCS, [
        question("partial", "alpha", ["z"]),
        question("both", "alpha", ["a", "z"]),
        question("unanswerable", "alpha", []),
    ], (1, 2))
    first = [row for row in report["cases"] if row["top_k"] == 1]
    assert all(row["positive_score_cutoff_tie"] for row in first)
    assert all(row["positive_score_cutoff_tied_doc_ids"] == ["a", "z"] for row in first)
    assert first[0]["positive_score_cutoff_tie_mixed_relevance"] is True
    assert first[1]["positive_score_cutoff_tie_mixed_relevance"] is False
    assert first[2]["positive_score_cutoff_tie_mixed_relevance"] is False
    assert first[0]["retrieved_ids"] == ["a"]
    assert first[0]["metrics"]["hit_rate"] == 0.0
    assert report["by_top_k"]["1"]["answerable_positive_score_cutoff_tie_count"] == 2
    assert report["by_top_k"]["1"]["answerable_mixed_relevance_cutoff_tie_count"] == 1
    assert report["by_top_k"]["1"]["unanswerable_positive_score_cutoff_tie_count"] == 1
    assert all(not row["positive_score_cutoff_tie"]
               for row in report["cases"] if row["top_k"] == 2)


def test_zero_score_ties_do_not_masquerade_as_positive_evidence():
    report = evaluate(DOCS, [question("unmatched", "neutrino", ["a"])], (1,))
    row = report["cases"][0]
    assert row["positive_score_cutoff_tie"] is False
    assert row["positive_score_cutoff_tied_doc_ids"] == []
    assert row["positive_score_cutoff_tie_mixed_relevance"] is False
    assert report["by_top_k"]["1"]["answerable_positive_score_cutoff_tie_count"] == 0


def test_unicode_mode_detects_only_actual_positive_ties():
    docs = [
        {"doc_id": "a", "title": "", "text": "हिन्दी"},
        {"doc_id": "b", "title": "", "text": "हिन्दी"},
        {"doc_id": "c", "title": "", "text": "english"},
    ]
    cases = [question("hindi", "हिन्दी", ["b"])]
    ascii_row = evaluate(docs, cases, (1,))["cases"][0]
    unicode_row = evaluate(
        docs, cases, (1,), tokenizer_mode="unicode"
    )["cases"][0]
    assert ascii_row["positive_score_cutoff_tie"] is False
    assert unicode_row["positive_score_cutoff_tie"] is True
    assert unicode_row["positive_score_cutoff_tied_doc_ids"] == ["a", "b"]
