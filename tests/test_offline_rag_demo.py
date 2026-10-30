"""Regression tests for the no-keyword-API BM25 retrieval demo."""
import json
import subprocess
import sys

import pytest

from evalforge.offline_rag_demo import bm25_rank, evaluate


DOCS = [
    {"doc_id": "a", "title": "Battery telemetry", "text": "voltage battery voltage"},
    {"doc_id": "b", "title": "Security audit", "text": "policy permission"},
    {"doc_id": "c", "title": "Temperature monitor", "text": "temperature sensor"},
]
QUESTIONS = [
    {"id": "q1", "question": "battery voltage", "answerable": True,
     "relevant_doc_ids": ["a"]},
    {"id": "q2", "question": "unknown fictional benchmark", "answerable": False,
     "relevant_doc_ids": []},
]


def test_retrieves_relevant_document_first():
    assert bm25_rank("battery voltage", DOCS)[0] == "a"


def test_report_has_reproducible_provenance_and_no_generation_claims():
    first = evaluate(DOCS, QUESTIONS, (1, 3))
    assert first == evaluate(DOCS, QUESTIONS, (1, 3))
    assert first["method"] == "lexical_bm25_retrieval_only"
    assert len(first["corpus_sha256"]) == 64
    assert first["by_top_k"]["1"]["answerable_count"] == 1
    assert first["by_top_k"]["1"]["unanswerable_excluded"] == 1
    assert first["by_top_k"]["1"]["hit_rate"] == 1.0
    assert first["cases"][1]["metrics"] is None


def test_precision_counts_requested_slots_even_when_corpus_shorter():
    report = evaluate(DOCS[:1], QUESTIONS[:1], (5,))
    assert report["by_top_k"]["5"]["precision_at_k"] == pytest.approx(0.2)


def test_rank_ties_are_deterministic_by_document_id():
    docs = [
        {"doc_id": "z", "title": "same", "text": "tokens"},
        {"doc_id": "a", "title": "same", "text": "tokens"},
    ]
    assert bm25_rank("absent", docs) == ["a", "z"]


@pytest.mark.parametrize("cutoffs", [(0,), (-1,), (True,), (1, 1), (), [1, "3"]])
def test_invalid_cutoffs_fail_closed(cutoffs):
    with pytest.raises(ValueError):
        evaluate(DOCS, QUESTIONS, cutoffs)


def test_duplicate_and_unknown_document_labels_are_rejected():
    with pytest.raises(ValueError, match="duplicate corpus"):
        evaluate(DOCS + [DOCS[0]], QUESTIONS)
    with pytest.raises(ValueError, match="relevance"):
        evaluate(DOCS, [dict(QUESTIONS[0], relevant_doc_ids=["missing"])])


def test_unanswerable_case_must_have_no_relevance_labels():
    with pytest.raises(ValueError, match="relevance"):
        evaluate(DOCS, [dict(QUESTIONS[1], relevant_doc_ids=["a"])])


def test_unicode_only_corpus_does_not_crash_tokenizer():
    docs = [{"doc_id": "u", "title": "日本語", "text": "日本語"}]
    report = evaluate(docs, [{"id": "q", "question": "battery", "answerable": True,
                              "relevant_doc_ids": ["u"]}], (1,))
    assert report["cases"][0]["retrieved_ids"] == ["u"]


def test_bundled_dataset_cli_runs_without_api_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.offline_rag_demo", "--k", "1", "3"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    report = json.loads(result.stdout)
    assert report["corpus_size"] == 15
    assert report["question_count"] == 14
    assert set(report["by_top_k"]) == {"1", "3"}
    assert report["by_top_k"]["1"]["answerable_count"] == 12
