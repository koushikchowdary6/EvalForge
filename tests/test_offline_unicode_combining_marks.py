"""Regression tests for Unicode combining marks in lexical BM25 retrieval."""

from evalforge.offline_rag_demo import (
    bm25_rank_with_scores, evaluate, tokenize_unicode,
)


def test_devanagari_vowel_signs_and_virama_remain_in_one_word():
    assert tokenize_unicode("हिन्दी भाषा") == ["हिन्दी", "भाषा"]
    assert tokenize_unicode("नमस्ते_शब्द") == ["नमस्ते", "शब्द"]
    # An unattached accent or vowel sign is not a lexical term.
    assert tokenize_unicode("\u0301_\u093e") == []


def test_casefold_expansion_preserves_turkish_dotted_i_as_one_token():
    assert tokenize_unicode("İSTANBUL") == ["i\u0307stanbul"]
    assert tokenize_unicode("i\u0307stanbul") == ["i\u0307stanbul"]
    assert tokenize_unicode("Cafe\u0301") == ["café"]


def test_bm25_does_not_match_a_fragment_of_a_combining_mark_word():
    docs = [
        {"doc_id": "a", "title": "ह", "text": "other"},
        {"doc_id": "z", "title": "हिन्दी", "text": "भाषा"},
    ]
    scored = bm25_rank_with_scores("हिन्दी", docs, tokenizer_mode="unicode")
    assert scored[0][0] == "z"
    assert scored[0][1] > 0
    assert dict(scored)["a"] == 0.0

    questions = [
        {"id": "hindi", "question": "हिन्दी", "answerable": True,
         "relevant_doc_ids": ["z"]},
    ]
    report = evaluate(docs, questions, (1,), tokenizer_mode="unicode")
    row = report["cases"][0]
    assert row["positive_score_retrieved_ids"] == ["z"]
    assert row["lexical_evidence_metrics"]["hit_rate"] == 1.0
    assert report["parameters"]["tokenizer"] == "unicode_alphanumeric"
    # Preserve the historical ASCII-only baseline unchanged.
    assert evaluate(docs, questions, (1,))["cases"][0][
        "positive_score_retrieved_ids"
    ] == []
