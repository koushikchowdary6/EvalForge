"""Out-of-vocabulary query diagnostics distinguish tokenizer and corpus gaps.

The BM25 baseline uses unique query terms, so repeated words must not inflate
coverage. These are lexical diagnostics, not estimates of semantic retrieval.
"""

import pytest

from evalforge.offline_rag_demo import evaluate


def _doc(identifier, text):
    return {"doc_id": identifier, "title": "", "text": text}


def _case(identifier, question, relevant, answerable=True):
    return {
        "id": identifier,
        "question": question,
        "answerable": answerable,
        "relevant_doc_ids": relevant,
    }


def test_unique_term_oov_counts_do_not_count_repeated_words():
    docs = [_doc("d1", "alpha beta"), _doc("d2", "gamma")]
    questions = [
        _case("partial", "alpha alpha zebra", ["d1"]),
        _case("all-oov", "zebra unknown", ["d2"]),
        _case("no-tokens", "!!!", [], answerable=False),
    ]
    report = evaluate(docs, questions, cutoffs=(1, 2))
    rows = {row["case_id"]: row for row in report["cases"] if row["top_k"] == 1}

    partial = rows["partial"]
    assert partial["query_selected_token_count"] == 3
    assert partial["query_unique_selected_term_count"] == 2
    assert partial["query_oov_term_count"] == 1
    assert partial["query_oov_term_fraction"] == pytest.approx(0.5)
    assert partial["query_all_selected_terms_oov"] is False

    all_oov = rows["all-oov"]
    assert all_oov["query_unique_selected_term_count"] == 2
    assert all_oov["query_oov_term_count"] == 2
    assert all_oov["query_oov_term_fraction"] == 1.0
    assert all_oov["query_all_selected_terms_oov"] is True
    assert all_oov["lexical_positive_document_count"] == 0

    no_tokens = rows["no-tokens"]
    assert no_tokens["query_unique_selected_term_count"] == 0
    assert no_tokens["query_oov_term_count"] == 0
    assert no_tokens["query_oov_term_fraction"] is None
    assert no_tokens["query_all_selected_terms_oov"] is False

    assert report["selected_tokenizer_diagnostics"]["questions_without_selected_tokens"] == 1
    diagnostics = report["query_vocabulary_diagnostics"]
    assert diagnostics["questions_with_all_selected_terms_oov"] == 1
    assert diagnostics["questions_with_partial_selected_terms_oov"] == 1
    for k in ("1", "2"):
        summary = report["by_top_k"][k]
        assert summary["answerable_queries_all_selected_terms_oov"] == 1
        assert summary["unanswerable_queries_all_selected_terms_oov"] == 0


def test_unicode_query_coverage_uses_selected_tokenizer_not_ascii():
    docs = [_doc("d1", "café alpha"), _doc("d2", "beta")]
    questions = [_case("q", "café élève", ["d1"])]
    report = evaluate(docs, questions, cutoffs=(1,), tokenizer_mode="unicode")
    row = report["cases"][0]
    assert row["query_unique_selected_term_count"] == 2
    assert row["query_oov_term_count"] == 1
    assert row["query_oov_term_fraction"] == pytest.approx(0.5)
    assert row["positive_score_retrieved_ids"] == ["d1"]


def test_no_oov_terms_when_all_selected_terms_in_corpus():
    report = evaluate(
        [_doc("d1", "alpha beta")],
        [_case("q", "beta alpha beta", ["d1"])],
        cutoffs=(1,),
    )
    row = report["cases"][0]
    assert row["query_oov_term_count"] == 0
    assert row["query_oov_term_fraction"] == 0.0
    assert report["query_vocabulary_diagnostics"]["questions_with_all_selected_terms_oov"] == 0
