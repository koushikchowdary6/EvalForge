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


def test_github_annotations_are_opt_in_and_escape_untrusted_ids(tmp_path):
    from evalforge.custom_gate import github_annotations
    assert github_annotations(gate(DOCS, QUESTIONS, min_hit_rate=0)) == []
    unsafe = [dict(QUESTIONS[0], id="q%\n::warning", question="unmatchedword")]
    report = gate(DOCS, unsafe, min_hit_rate=0)
    annotations = github_annotations(report)
    assert annotations and all(line.startswith("::error title=") for line in annotations)
    assert any("q%25%0A%3A%3Awarning" in line for line in annotations)
    assert all("\n" not in line and "\r" not in line for line in annotations)

    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(unsafe), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--min-hit-rate", "0"]
    plain = subprocess.run(cmd, capture_output=True, text=True)
    annotated = subprocess.run(cmd + ["--github-annotations"], capture_output=True, text=True)
    assert plain.returncode == annotated.returncode == 1
    assert "::error" not in plain.stderr
    assert "::error title=EvalForge dataset preflight::" in annotated.stderr
    assert json.loads(plain.stdout) == json.loads(annotated.stdout)


def test_opt_in_case_evidence_preserves_aggregate_contract(tmp_path):
    compact = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    detailed = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0, include_cases=True)
    assert "cases" not in compact
    assert detailed["cases"]
    assert {key: value for key, value in detailed.items() if key != "cases"} == compact
    assert detailed == gate(DOCS, QUESTIONS, k=1, min_hit_rate=0, include_cases=True)
    with pytest.raises(ValueError):
        gate(DOCS, QUESTIONS, include_cases="yes")

    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--k", "1", "--min-hit-rate", "0"]
    plain = subprocess.run(cmd, capture_output=True, text=True)
    detailed_cli = subprocess.run(cmd + ["--include-cases"], capture_output=True, text=True)
    assert plain.returncode == detailed_cli.returncode == 0
    assert "cases" not in json.loads(plain.stdout)
    assert json.loads(detailed_cli.stdout) == detailed
