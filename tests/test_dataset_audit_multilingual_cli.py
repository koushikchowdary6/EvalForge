"""Regression coverage for multilingual dataset preflight in the installed CLI.

ASCII lexical checks can mark a valid non-Latin benchmark unreachable. The
Unicode option must retain combining marks and make that distinction explicit.
These tests exercise the actual CLI and JSON report, not only the tokenizer.
"""
import json
import subprocess
import sys

from evalforge.dataset_audit import audit


def _fixture():
    docs = [
        {"doc_id": "hindi", "title": "भुगतान", "text": "रसीद उपलब्ध है"},
        {"doc_id": "english", "title": "Shipping", "text": "Track your package"},
    ]
    questions = [
        {"id": "receipt", "question": "रसीद कहाँ है", "answerable": True,
         "relevant_doc_ids": ["hindi"]},
    ]
    return docs, questions


def test_unicode_audit_preserves_devanagari_word_overlap():
    docs, questions = _fixture()
    unicode_report = audit(docs, questions, tokenizer="unicode")
    ascii_report = audit(docs, questions, tokenizer="ascii")

    assert unicode_report["answerable_question_ids_without_relevant_doc_token_overlap"] == []
    assert unicode_report["answerable_question_ids_without_corpus_token_overlap"] == []
    assert ascii_report["answerable_question_ids_without_relevant_doc_token_overlap"] == ["receipt"]
    assert unicode_report["corpus_sha256"] == ascii_report["corpus_sha256"]
    assert unicode_report["questions_sha256"] == ascii_report["questions_sha256"]


def test_strict_cli_reports_tokenizer_specific_result(tmp_path):
    docs, questions = _fixture()
    corpus = tmp_path / "corpus.json"
    labels = tmp_path / "questions.json"
    corpus.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    labels.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")

    command = [sys.executable, "-m", "evalforge.dataset_audit",
               "--corpus", str(corpus), "--questions", str(labels), "--strict"]
    unicode_run = subprocess.run(command + ["--tokenizer", "unicode"],
                                 capture_output=True, text=True)
    ascii_run = subprocess.run(command + ["--tokenizer", "ascii"],
                               capture_output=True, text=True)

    assert unicode_run.returncode == 0, unicode_run.stderr
    assert ascii_run.returncode == 1, ascii_run.stderr
    assert json.loads(unicode_run.stdout)["tokenizer"] == "unicode"
    assert json.loads(ascii_run.stdout)[
        "answerable_question_ids_without_relevant_doc_token_overlap"
    ] == ["receipt"]
