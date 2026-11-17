"""Regression tests for custom retrieval gate Markdown summaries."""
import json
import subprocess
import sys

import pytest

from evalforge.custom_gate import gate
from evalforge.custom_gate_summary import render


DOCS = [
    {"doc_id": "a", "title": "Refund policy", "text": "refund purchases"},
    {"doc_id": "b", "title": "Shipping", "text": "track packages"},
]
QUESTIONS = [
    {"id": "q1", "question": "refund", "answerable": True, "relevant_doc_ids": ["a"]},
    {"id": "q2", "question": "unknown", "answerable": False, "relevant_doc_ids": []},
]


def test_passing_summary_is_reproducible_and_scoped():
    report = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    first = render(report)
    assert first == render(report)
    assert "**Result:** PASS" in first
    assert "offline BM25 lexical retrieval only" in first
    assert report["dataset_audit"]["corpus_sha256"] in first
    assert "| Answerable questions | 1 | 0 |" in first


def test_failing_summary_has_readable_reasons():
    report = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0, min_answerable_questions=2)
    output = render(report)
    assert "**Result:** FAIL" in output
    assert "Too few answerable questions" in output


@pytest.mark.parametrize("mutation", [
    {"schema": "other"},
    {"scope": "claims_production_safety"},
    {"passed": "true"},
    {"failure_reasons": ["\n::warning::injected"]},
    {"observed_lexical_evidence_hit_rate": float("nan")},
    {"minimum_lexical_evidence_hit_rate": 2},
    {"k": True},
])
def test_rejects_malformed_or_unsafe_artifacts(mutation):
    report = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    report.update(mutation)
    with pytest.raises(ValueError):
        render(report)


def test_cli_no_overwrite_and_invalid_json(tmp_path):
    artifact = tmp_path / "evidence.json"
    summary = tmp_path / "summary.md"
    artifact.write_text(json.dumps(gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate_summary", str(artifact),
           "--output", str(summary)]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    original = summary.read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode == 2
    assert summary.read_bytes() == original
    artifact.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    assert subprocess.run(cmd[:4], capture_output=True).returncode == 2
