"""Attribute lexical retrieval gaps without confusing them with semantic accuracy."""

from evalforge.offline_rag_demo import evaluate


DOCS = [
    {"doc_id": "r1", "title": "alpha", "text": "one"},
    {"doc_id": "r2", "title": "beta", "text": "two"},
    {"doc_id": "n1", "title": "gamma", "text": "three"},
    {"doc_id": "jp", "title": "日本語", "text": "資料"},
]
CASES = [
    {"id": "full", "question": "alpha", "answerable": True,
     "relevant_doc_ids": ["r1"]},
    {"id": "no-tokens", "question": "日本語", "answerable": True,
     "relevant_doc_ids": ["jp"]},
    {"id": "all-oov", "question": "unseen", "answerable": True,
     "relevant_doc_ids": ["r1"]},
    {"id": "no-relevant", "question": "gamma", "answerable": True,
     "relevant_doc_ids": ["r1"]},
    {"id": "partial", "question": "alpha", "answerable": True,
     "relevant_doc_ids": ["r1", "r2"]},
    {"id": "unanswerable", "question": "gamma", "answerable": False,
     "relevant_doc_ids": []},
]
EXPECTED = {
    "full_positive_relevance": ["full"],
    "no_selected_query_tokens": ["no-tokens"],
    "all_selected_query_terms_oov": ["all-oov"],
    "no_positive_relevant_overlap": ["no-relevant"],
    "partial_positive_relevant_overlap": ["partial"],
}


def test_relevance_gap_reasons_are_disjoint_and_cutoff_independent():
    report = evaluate(DOCS, CASES, cutoffs=(1, 3))
    assert report["query_relevance_coverage"]["reason_case_ids"] == EXPECTED
    for k in (1, 3):
        rows = {row["case_id"]: row for row in report["cases"]
                if row["top_k"] == k}
        for reason, case_ids in EXPECTED.items():
            for case_id in case_ids:
                assert rows[case_id]["query_relevance_gap_reason"] == reason
        assert rows["unanswerable"]["query_relevance_gap_reason"] is None
        assert rows["no-relevant"]["lexical_positive_document_count"] > 0
        assert rows["no-relevant"]["relevant_positive_score_recall_ceiling"] == 0
        assert rows["partial"]["relevant_positive_score_recall_ceiling"] == 0.5


def test_selected_unicode_tokenizer_changes_gap_attribution():
    report = evaluate(DOCS, CASES, cutoffs=(1,), tokenizer_mode="unicode")
    reasons = report["query_relevance_coverage"]["reason_case_ids"]
    assert "no-tokens" in reasons["full_positive_relevance"]
    assert reasons["no_selected_query_tokens"] == []
    rows = {row["case_id"]: row for row in report["cases"]}
    assert rows["no-tokens"]["query_relevance_gap_reason"] == "full_positive_relevance"


def test_unanswerable_only_reports_no_relevance_reason():
    report = evaluate(DOCS, [CASES[-1]], cutoffs=(1,))
    assert all(not case_ids for case_ids in
               report["query_relevance_coverage"]["reason_case_ids"].values())
    assert report["cases"][0]["query_relevance_gap_reason"] is None
