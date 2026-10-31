"""Audit retrieval-label ambiguity when different IDs have identical BM25 evidence."""
from evalforge.offline_rag_demo import evaluate


def test_partial_relevance_with_equivalent_documents_is_reported_without_changing_ranking():
    docs = [
        {"doc_id": "z", "title": "", "text": "beta alpha alpha"},
        {"doc_id": "a", "title": "", "text": "alpha beta alpha"},
        {"doc_id": "x", "title": "", "text": "gamma delta"},
    ]
    questions = [
        {"id": "partial", "question": "alpha", "answerable": True,
         "relevant_doc_ids": ["z"]},
        {"id": "both", "question": "beta", "answerable": True,
         "relevant_doc_ids": ["z", "a"]},
        {"id": "other", "question": "gamma", "answerable": True,
         "relevant_doc_ids": ["x"]},
        {"id": "unanswerable", "question": "missing", "answerable": False,
         "relevant_doc_ids": []},
    ]
    report = evaluate(docs, questions, (1, 3))
    audit = report["corpus_lexical_equivalence"]
    assert audit == {
        "tokenizer_mode": "ascii",
        "equivalent_group_count": 1,
        "equivalent_document_count": 2,
        "equivalent_document_id_groups": [["a", "z"]],
        "answerable_cases_with_partial_equivalence_labels": ["partial"],
        "scope": "identical_bm25_token_frequency_multisets_not_semantic_duplicates",
    }
    # BM25 still scores and ties as before; the audit never changes ranking.
    row = next(row for row in report["cases"]
               if row["case_id"] == "partial" and row["top_k"] == 1)
    assert row["retrieved_ids"] == ["a"]
    assert row["metrics"]["hit_rate"] == 0.0
    assert report["by_top_k"]["3"]["answerable_count"] == 3


def test_unicode_mode_does_not_conflate_distinct_non_ascii_token_bags():
    docs = [
        {"doc_id": "a", "title": "", "text": "日本語"},
        {"doc_id": "b", "title": "", "text": "한국어"},
    ]
    questions = [
        {"id": "jp", "question": "日本語", "answerable": True,
         "relevant_doc_ids": ["a"]},
    ]
    ascii_audit = evaluate(docs, questions, (1,))["corpus_lexical_equivalence"]
    unicode_audit = evaluate(
        docs, questions, (1,), tokenizer_mode="unicode"
    )["corpus_lexical_equivalence"]
    assert ascii_audit["equivalent_document_id_groups"] == [["a", "b"]]
    assert ascii_audit["answerable_cases_with_partial_equivalence_labels"] == ["jp"]
    assert unicode_audit["equivalent_group_count"] == 0
    assert unicode_audit["answerable_cases_with_partial_equivalence_labels"] == []


def test_no_collisions_produces_empty_audit_and_is_deterministic():
    docs = [
        {"doc_id": "a", "title": "alpha", "text": "beta"},
        {"doc_id": "b", "title": "gamma", "text": "delta"},
    ]
    questions = [
        {"id": "q", "question": "alpha", "answerable": True,
         "relevant_doc_ids": ["a"]},
    ]
    first = evaluate(docs, questions, (1,))
    assert first["corpus_lexical_equivalence"]["equivalent_document_id_groups"] == []
    assert first == evaluate(docs, questions, (1,))
