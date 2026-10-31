"""Regression coverage for corpus-index reuse in the offline BM25 baseline."""
from collections import Counter

from evalforge import offline_rag_demo as bm25


DOCS = [
    {"doc_id": "c", "title": "Battery", "text": "voltage monitor"},
    {"doc_id": "a", "title": "Security", "text": "policy rules"},
    {"doc_id": "b", "title": "Temperature", "text": "sensor data"},
]
QUESTIONS = [
    {"id": "battery", "question": "battery voltage", "answerable": True,
     "relevant_doc_ids": ["c"]},
    {"id": "security", "question": "security policy", "answerable": True,
     "relevant_doc_ids": ["a"]},
    {"id": "no_match", "question": "galactic nebula", "answerable": False,
     "relevant_doc_ids": []},
]


def test_multiquery_evaluation_tokenizes_corpus_only_once(monkeypatch):
    original = bm25.tokenize
    calls = Counter()

    def counting_tokenize(text):
        calls[text] += 1
        return original(text)

    monkeypatch.setattr(bm25, "tokenize", counting_tokenize)
    report = bm25.evaluate(DOCS, QUESTIONS, (1, 3))
    for doc in DOCS:
        assert calls[doc["title"] + " " + doc["text"]] == 1
    # Query text is tokenized once for scoring and once for diagnostics.
    for case in QUESTIONS:
        assert calls[case["question"]] == 2
    assert len(report["cases"]) == len(QUESTIONS) * 2
    assert report["by_top_k"]["1"]["lexical_evidence_hit_rate"] == 1.0


def test_cached_evaluation_matches_standalone_ranking_and_zero_score_ties():
    report = bm25.evaluate(DOCS, QUESTIONS, (1, 3))
    for cutoff in (1, 3):
        rows = [row for row in report["cases"] if row["top_k"] == cutoff]
        for case, row in zip(QUESTIONS, rows):
            scored = bm25.bm25_rank_with_scores(case["question"], DOCS)
            assert row["retrieved_ids"] == [doc_id for doc_id, _ in scored[:cutoff]]
            assert row["positive_score_retrieved_ids"] == [
                doc_id for doc_id, score in scored if score > 0
            ][:cutoff]
    no_match = next(row for row in report["cases"]
                    if row["case_id"] == "no_match" and row["top_k"] == 3)
    assert no_match["retrieved_ids"] == ["a", "b", "c"]
    assert no_match["positive_score_retrieved_ids"] == []
