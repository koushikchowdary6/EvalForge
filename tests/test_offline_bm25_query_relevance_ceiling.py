"""A BM25-indexable label is not necessarily retrievable for a given query.

These deterministic diagnostics describe lexical overlap, not semantic
relevance, embedding accuracy, or an independent benchmark.
"""

import pytest

from evalforge.offline_rag_demo import evaluate


DOCS = [
    {"doc_id": "r1", "title": "alpha", "text": "one"},
    {"doc_id": "r2", "title": "beta", "text": "two"},
    {"doc_id": "n1", "title": "gamma", "text": "three"},
]
QUESTIONS = [
    {"id": "partial", "question": "alpha gamma", "answerable": True,
     "relevant_doc_ids": ["r1", "r2"]},
    {"id": "none", "question": "unseen", "answerable": True,
     "relevant_doc_ids": ["r2"]},
    {"id": "unanswerable", "question": "alpha", "answerable": False,
     "relevant_doc_ids": []},
]


def test_query_specific_relevance_ceiling_distinguishes_indexability_from_overlap():
    report = evaluate(DOCS, QUESTIONS, cutoffs=(1, 3))
    assert report["query_relevance_coverage"][
        "answerable_cases_with_zero_positive_relevant_documents"
    ] == ["none"]
    assert report["query_relevance_coverage"][
        "answerable_cases_with_partial_positive_relevance"
    ] == ["partial"]
    for k in ("1", "3"):
        rows = {r["case_id"]: r for r in report["cases"] if r["top_k"] == int(k)}
        assert rows["partial"]["relevant_indexable_recall_ceiling"] == 1.0
        assert rows["partial"]["relevant_positive_score_document_count"] == 1
        assert rows["partial"]["relevant_positive_score_recall_ceiling"] == 0.5
        assert rows["none"]["relevant_indexable_recall_ceiling"] == 1.0
        assert rows["none"]["relevant_positive_score_document_count"] == 0
        assert rows["none"]["relevant_positive_score_recall_ceiling"] == 0.0
        assert rows["unanswerable"]["relevant_positive_score_document_count"] is None
        assert rows["unanswerable"]["relevant_positive_score_recall_ceiling"] is None
        summary = report["by_top_k"][k]
        assert summary["answerable_cases_with_query_relevance_gap"] == 2
        assert summary["mean_relevant_positive_score_recall_ceiling"] == pytest.approx(0.25)
        for case_id in ("partial", "none"):
            assert (rows[case_id]["lexical_evidence_recall_upper_bound"]
                    <= rows[case_id]["relevant_positive_score_recall_ceiling"])


def test_unicode_overlap_ceiling_uses_selected_tokenizer():
    docs = [{"doc_id": "jp", "title": "日本語", "text": "資料"}]
    cases = [{"id": "jp-query", "question": "日本語", "answerable": True,
              "relevant_doc_ids": ["jp"]}]
    ascii_row = evaluate(docs, cases, cutoffs=(1,), tokenizer_mode="ascii")["cases"][0]
    unicode_row = evaluate(docs, cases, cutoffs=(1,), tokenizer_mode="unicode")["cases"][0]
    assert ascii_row["relevant_positive_score_recall_ceiling"] == 0.0
    assert unicode_row["relevant_positive_score_recall_ceiling"] == 1.0


def test_unanswerable_only_query_relevance_summary_is_undefined():
    cases = [{"id": "no-answer", "question": "alpha", "answerable": False,
              "relevant_doc_ids": []}]
    report = evaluate(DOCS, cases, cutoffs=(1,))
    assert report["by_top_k"]["1"]["mean_relevant_positive_score_recall_ceiling"] is None
    assert report["by_top_k"]["1"]["answerable_cases_with_query_relevance_gap"] == 0
    assert report["query_relevance_coverage"][
        "answerable_cases_with_zero_positive_relevant_documents"
    ] == []
