"""Opt-in Unicode lexical retrieval without changing the legacy ASCII benchmark."""
import json
import subprocess
import sys

import pytest

from evalforge.offline_rag_demo import (
    bm25_rank, bm25_rank_with_scores, evaluate, tokenize_unicode,
)


DOCS = [
    {"doc_id": "a", "title": "Battery", "text": "voltage measurements"},
    {"doc_id": "z", "title": "日本語", "text": "資料"},
]
QUESTIONS = [
    {"id": "jp", "question": "日本語", "answerable": True,
     "relevant_doc_ids": ["z"]},
    {"id": "unknown", "question": "未知", "answerable": False,
     "relevant_doc_ids": []},
]


def test_unicode_mode_recovers_lexical_evidence_hidden_by_ascii_ties():
    legacy = evaluate(DOCS, QUESTIONS, (1,))
    unicode_report = evaluate(DOCS, QUESTIONS, (1,), tokenizer_mode="unicode")

    assert legacy["cases"][0]["retrieved_ids"] == ["a"]
    assert legacy["cases"][0]["lexical_evidence_metrics"]["hit_rate"] == 0.0
    assert unicode_report["cases"][0]["retrieved_ids"] == ["z"]
    assert unicode_report["cases"][0]["positive_score_retrieved_ids"] == ["z"]
    assert unicode_report["cases"][0]["lexical_evidence_metrics"]["hit_rate"] == 1.0
    assert unicode_report["cases"][0]["query_ascii_token_count"] == 0
    assert unicode_report["cases"][0]["query_selected_token_count"] == 1
    assert unicode_report["selected_tokenizer_diagnostics"] == {
        "questions_without_selected_tokens": 0, "tokenizer_mode": "unicode",
    }
    assert unicode_report["parameters"]["tokenizer"] == "unicode_alphanumeric"
    assert legacy["parameters"]["tokenizer"] == "ascii_alphanumeric"
    assert legacy["corpus_sha256"] == unicode_report["corpus_sha256"]
    assert legacy["questions_sha256"] == unicode_report["questions_sha256"]
    assert unicode_report == evaluate(DOCS, QUESTIONS, (1,), tokenizer_mode="unicode")


def test_unicode_normalizes_canonical_accent_forms_but_not_semantics():
    assert tokenize_unicode("Cafe\u0301") == ["café"]
    assert tokenize_unicode("CAFÉ") == ["café"]
    assert tokenize_unicode("日本語") == ["日本語"]
    assert bm25_rank("Cafe\u0301", [
        {"doc_id": "a", "title": "coffee", "text": "beans"},
        {"doc_id": "z", "title": "café", "text": "menu"},
    ], tokenizer_mode="unicode")[0] == "z"


def test_standalone_unicode_scoring_and_default_ascii_behavior():
    assert bm25_rank("日本語", DOCS) == ["a", "z"]
    scores = bm25_rank_with_scores("日本語", DOCS, tokenizer_mode="unicode")
    assert scores[0][0] == "z" and scores[0][1] > 0
    assert scores[1] == ("a", 0.0)


@pytest.mark.parametrize("bad_mode", ["Unicode", "", None, True])
def test_unknown_tokenizer_mode_is_rejected(bad_mode):
    with pytest.raises(ValueError, match="tokenizer_mode"):
        evaluate(DOCS, QUESTIONS, (1,), tokenizer_mode=bad_mode)


def test_unicode_cli_mode_emits_explicit_provenance():
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.offline_rag_demo",
         "--tokenizer", "unicode", "--k", "1"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    report = json.loads(result.stdout)
    assert report["parameters"]["tokenizer"] == "unicode_alphanumeric"
    assert report["selected_tokenizer_diagnostics"]["tokenizer_mode"] == "unicode"
    assert report["question_count"] == 14
