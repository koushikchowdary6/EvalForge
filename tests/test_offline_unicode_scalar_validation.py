"""Escaped unpaired Unicode surrogates must fail benchmark validation cleanly."""

import json

import pytest

from evalforge.offline_rag_demo import (
    bm25_rank_with_scores,
    evaluate,
    load_benchmark_json,
)


DOCS = [{"doc_id": "d1", "title": "Valid", "text": "battery voltage"}]
QUESTIONS = [{
    "id": "q1", "question": "battery", "answerable": True,
    "relevant_doc_ids": ["d1"],
}]


@pytest.mark.parametrize("field", ["title", "text"])
def test_invalid_corpus_scalar_fails_before_ranking_and_report_hashing(field):
    docs = [dict(DOCS[0], **{field: "ok" + chr(0xD800)})]
    with pytest.raises(ValueError, match="invalid corpus document"):
        evaluate(docs, QUESTIONS)
    with pytest.raises(ValueError, match="invalid corpus document"):
        bm25_rank_with_scores("battery", docs)


def test_invalid_question_scalar_fails_before_report_hashing():
    cases = [dict(QUESTIONS[0], question="battery" + chr(0xDC00))]
    with pytest.raises(ValueError, match="invalid benchmark question"):
        evaluate(DOCS, cases)
    with pytest.raises(ValueError, match="invalid benchmark question"):
        bm25_rank_with_scores("battery" + chr(0xDC00), DOCS)


def test_escaped_unpaired_surrogate_in_json_file_is_rejected(tmp_path):
    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps([dict(QUESTIONS[0], question="battery" + chr(0xD800))]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid benchmark question"):
        evaluate(DOCS, load_benchmark_json(source))


def test_valid_supplementary_unicode_remains_accepted():
    docs = [dict(DOCS[0], text="battery 🔋")]
    cases = [dict(QUESTIONS[0], question="battery 🔋")]
    assert evaluate(docs, cases, cutoffs=(1,))["by_top_k"]["1"]["answerable_count"] == 1
