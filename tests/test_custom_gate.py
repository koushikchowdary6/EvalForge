"""Contract tests for the custom-dataset lexical regression gate."""
import json
import subprocess
import sys

import pytest

from evalforge.custom_gate import gate

DOCS = [
    {"doc_id": "a", "title": "Refund policy", "text": "Request refund for purchases"},
    {"doc_id": "b", "title": "Shipping status", "text": "Track shipping packages"},
]
QUESTIONS = [
    {"id": "q1", "question": "refund purchases", "answerable": True, "relevant_doc_ids": ["a"]},
    {"id": "q2", "question": "unknown", "answerable": False, "relevant_doc_ids": []},
]


def test_passes_with_reproducible_evidence():
    report = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    assert report == gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    assert report["passed"] is True
    assert report["benchmark"]["corpus_sha256"] == report["dataset_audit"]["corpus_sha256"]
    assert report["benchmark"]["questions_sha256"] == report["dataset_audit"]["questions_sha256"]
    assert "cases" not in report["benchmark"]


def test_fails_threshold_and_unreachable_labels():
    unreachable = [dict(QUESTIONS[0], question="unmatchedword")]
    report = gate(DOCS, unreachable, min_hit_rate=0)
    assert not report["passed"]
    assert report["preflight_warnings"]["answerable_question_ids_without_relevant_doc_token_overlap"] == ["q1"]


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError):
        gate(DOCS, QUESTIONS, min_hit_rate=value)


def test_cli_exit_codes_and_no_overwrite(tmp_path):
    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    output = tmp_path / "report.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--min-hit-rate", "0", "--output", str(output)]
    success = subprocess.run(cmd, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert json.loads(output.read_text())["passed"]
    assert subprocess.run(cmd, capture_output=True).returncode == 2
    questions.write_text(json.dumps([dict(QUESTIONS[0], question="unmatchedword")]), encoding="utf-8")
    output.unlink()
    failed = subprocess.run(cmd, capture_output=True, text=True)
    assert failed.returncode == 1, failed.stderr
    assert not json.loads(output.read_text())["passed"]
