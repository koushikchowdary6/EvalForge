"""Corpus tokenization gaps limit the maximum evidence-backed BM25 recall."""

import pytest

from evalforge.offline_rag_demo import evaluate


DOCS = [
    {"doc_id": "japanese", "title": "日本語", "text": "資料"},
    {"doc_id": "english", "title": "battery", "text": "voltage"},
]
QUESTIONS = [
    {"id": "j-query", "question": "日本語", "answerable": True,
     "relevant_doc_ids": ["japanese"]},
    {"id": "e-query", "question": "battery", "answerable": True,
     "relevant_doc_ids": ["english"]},
    {"id": "no-answer", "question": "unrelated", "answerable": False,
     "relevant_doc_ids": []},
]


def test_ascii_corpus_coverage_exposes_unreachable_relevance():
    report = evaluate(DOCS, QUESTIONS, cutoffs=(1, 2), tokenizer_mode="ascii")
    coverage = report["corpus_token_coverage"]
    assert coverage["tokenizer_mode"] == "ascii"
    assert coverage["unindexable_document_count"] == 1
    assert coverage["unindexable_document_ids"] == ["japanese"]
    assert coverage["answerable_cases_with_unindexable_relevance"] == ["j-query"]

    for k in (1, 2):
        rows = {row["case_id"]: row for row in report["cases"] if row["top_k"] == k}
        assert rows["j-query"]["relevant_unindexable_document_count"] == 1
        assert rows["j-query"]["relevant_indexable_recall_ceiling"] == pytest.approx(0.0)
        assert rows["e-query"]["relevant_unindexable_document_count"] == 0
        assert rows["e-query"]["relevant_indexable_recall_ceiling"] == pytest.approx(1.0)
        assert rows["no-answer"]["relevant_unindexable_document_count"] is None
        assert rows["no-answer"]["relevant_indexable_recall_ceiling"] is None
        assert report["by_top_k"][str(k)]["answerable_cases_with_unindexable_relevance"] == 1


def test_unicode_corpus_coverage_restores_indexability_without_claiming_accuracy():
    report = evaluate(DOCS, QUESTIONS, cutoffs=(1,), tokenizer_mode="unicode")
    coverage = report["corpus_token_coverage"]
    assert coverage["unindexable_document_count"] == 0
    assert coverage["unindexable_document_ids"] == []
    assert coverage["answerable_cases_with_unindexable_relevance"] == []
    rows = {row["case_id"]: row for row in report["cases"]}
    assert rows["j-query"]["relevant_unindexable_document_count"] == 0
    assert rows["j-query"]["relevant_indexable_recall_ceiling"] == pytest.approx(1.0)
    assert report["by_top_k"]["1"]["answerable_cases_with_unindexable_relevance"] == 0
