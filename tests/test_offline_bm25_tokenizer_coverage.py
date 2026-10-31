"""ASCII BM25 coverage diagnostics must not masquerade as multilingual retrieval."""
from evalforge.offline_rag_demo import evaluate


DOCS = [
    {"doc_id": "a", "title": "日本語", "text": "資料"},
    {"doc_id": "b", "title": "Battery", "text": "battery voltage"},
]
QUESTIONS = [
    {"id": "japanese", "question": "日本語", "answerable": True,
     "relevant_doc_ids": ["a"]},
    {"id": "mixed", "question": "café battery", "answerable": True,
     "relevant_doc_ids": ["b"]},
    {"id": "unanswerable", "question": "未知", "answerable": False,
     "relevant_doc_ids": []},
]


def test_ascii_tokenizer_blind_spots_are_explicit_in_report():
    report = evaluate(DOCS, QUESTIONS, (1, 2))
    assert report["tokenizer_diagnostics"] == {
        "questions_without_ascii_tokens": 2,
        "questions_with_non_ascii_alphanumeric": 3,
    }
    first, mixed, unanswerable = report["cases"][:3]
    assert first["query_ascii_token_count"] == 0
    assert first["query_has_no_ascii_tokens"] is True
    assert first["query_contains_non_ascii_alphanumeric"] is True
    # A lexically unsupported query may still rank a relevant ID by tie.
    assert first["retrieved_ids"] == ["a"]
    assert first["metrics"]["hit_rate"] == 1.0
    assert first["lexical_evidence_metrics"]["hit_rate"] == 0.0
    assert mixed["query_ascii_token_count"] == 2
    assert mixed["query_has_no_ascii_tokens"] is False
    assert mixed["query_contains_non_ascii_alphanumeric"] is True
    assert unanswerable["query_has_no_ascii_tokens"] is True
    assert unanswerable["metrics"] is None
    for summary in report["by_top_k"].values():
        assert summary["answerable_without_ascii_query_tokens"] == 1
        assert summary["unanswerable_without_ascii_query_tokens"] == 1
        assert summary["questions_with_non_ascii_alphanumeric"] == 3


def test_ascii_only_queries_report_no_multilingual_warning():
    docs = [{"doc_id": "x", "title": "battery", "text": "voltage"}]
    cases = [{"id": "q", "question": "battery", "answerable": True,
              "relevant_doc_ids": ["x"]}]
    report = evaluate(docs, cases, (1,))
    assert report["tokenizer_diagnostics"] == {
        "questions_without_ascii_tokens": 0,
        "questions_with_non_ascii_alphanumeric": 0,
    }
    row = report["cases"][0]
    assert row["query_ascii_token_count"] == 1
    assert row["query_has_no_ascii_tokens"] is False
    assert row["query_contains_non_ascii_alphanumeric"] is False
