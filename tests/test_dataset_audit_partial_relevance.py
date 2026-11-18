"""Regression coverage for partially reachable multi-document relevance labels."""

import json
import subprocess
import sys

from evalforge.dataset_audit import audit


def test_partial_relevant_lexical_overlap_is_separate_from_zero_overlap():
    docs = [
        {"doc_id": "matching", "title": "Refund", "text": "Refund policy"},
        {"doc_id": "missing", "title": "Shipping", "text": "Delivery times"},
    ]
    questions = [
        {"id": "partial", "question": "refund", "answerable": True,
         "relevant_doc_ids": ["matching", "missing"]},
        {"id": "full", "question": "refund shipping", "answerable": True,
         "relevant_doc_ids": ["matching", "missing"]},
        {"id": "none", "question": "unknown", "answerable": True,
         "relevant_doc_ids": ["matching", "missing"]},
    ]
    report = audit(docs, questions)
    assert report["answerable_question_ids_with_partial_relevant_doc_token_overlap"] == ["partial"]
    assert report["answerable_question_ids_without_relevant_doc_token_overlap"] == ["none"]
    assert report["answerable_question_ids_without_corpus_token_overlap"] == ["none"]


def test_partial_overlap_is_diagnostic_not_a_strict_preflight_failure(tmp_path):
    docs = [
        {"doc_id": "matching", "title": "Refund", "text": "Refund policy"},
        {"doc_id": "missing", "title": "Shipping", "text": "Delivery times"},
    ]
    questions = [
        {"id": "partial", "question": "refund", "answerable": True,
         "relevant_doc_ids": ["matching", "missing"]},
    ]
    corpus = tmp_path / "corpus.json"
    labels = tmp_path / "questions.json"
    corpus.write_text(json.dumps(docs), encoding="utf-8")
    labels.write_text(json.dumps(questions), encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-m", "evalforge.dataset_audit",
         "--corpus", str(corpus), "--questions", str(labels), "--strict"],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["answerable_question_ids_with_partial_relevant_doc_token_overlap"] == ["partial"]
