"""End-to-end example of evaluating a developer-supplied retrieval dataset.

This fixture intentionally contains a relevant result, an unrelated result,
and an unanswerable query. It tests lexical overlap, not generated answers.
"""
import json
import subprocess
import sys
from pathlib import Path

from evalforge.offline_rag_demo import evaluate


def test_custom_corpus_cli_matches_python_api_and_preserves_provenance(tmp_path):
    corpus = [
        {"doc_id": "runbook", "title": "Credential rotation runbook",
         "text": "Rotate compromised service credentials and revoke old tokens."},
        {"doc_id": "observability", "title": "Telemetry ingestion",
         "text": "Batch sensor events and monitor ingestion latency."},
        {"doc_id": "deployment", "title": "Deployment checklist",
         "text": "Verify deployment health and rollback readiness."},
    ]
    questions = [
        {"id": "credential-response", "question": "rotate compromised credentials",
         "answerable": True, "relevant_doc_ids": ["runbook"]},
        {"id": "missing-topic", "question": "oceanography coral reefs",
         "answerable": False, "relevant_doc_ids": []},
    ]
    corpus_file = tmp_path / "corpus.json"
    questions_file = tmp_path / "questions.json"
    output_file = tmp_path / "report.json"
    corpus_file.write_text(json.dumps(corpus), encoding="utf-8")
    questions_file.write_text(json.dumps(questions), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "evalforge.offline_rag_demo",
         "--corpus", str(corpus_file), "--questions", str(questions_file),
         "--k", "1", "3", "--summary-only", "--output", str(output_file)],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout == ""
    report = json.loads(output_file.read_text(encoding="utf-8"))
    expected = evaluate(corpus, questions, (1, 3))
    assert report == {key: value for key, value in expected.items() if key != "cases"}
    assert report["by_top_k"]["1"]["lexical_evidence_hit_rate"] == 1.0
    assert report["by_top_k"]["1"]["unanswerable_positive_lexical_match_count"] == 0
    assert report["corpus_sha256"] == expected["corpus_sha256"]
    assert report["questions_sha256"] == expected["questions_sha256"]

    # A repeated CI invocation must not silently overwrite a benchmark artifact.
    repeated = subprocess.run(
        [sys.executable, "-m", "evalforge.offline_rag_demo",
         "--corpus", str(corpus_file), "--questions", str(questions_file),
         "--k", "1", "3", "--summary-only", "--output", str(output_file)],
        capture_output=True, text=True,
    )
    assert repeated.returncode != 0
    assert "cannot write report" in repeated.stderr
    assert json.loads(output_file.read_text(encoding="utf-8")) == report
