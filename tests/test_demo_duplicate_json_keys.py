"""Reject ambiguous JSON from offline benchmark subprocesses."""
import json
import subprocess

import pytest

from evalforge import demo


def _valid_report():
    return {
        "method": "lexical_bm25_retrieval_only",
        "question_count": 2,
        "corpus_sha256": "a" * 64,
        "questions_sha256": "b" * 64,
        "by_top_k": {
            str(k): {"lexical_evidence_hit_rate": 0.5} for k in (1, 3, 5)
        },
    }


@pytest.mark.parametrize("field", ["question_count", "by_top_k"])
def test_run_report_rejects_duplicate_top_level_fields(tmp_path, monkeypatch, field):
    report = _valid_report()
    serialized = json.dumps(report)
    duplicate = json.dumps(field) + ":" + json.dumps(report[field])
    serialized = serialized[:-1] + "," + duplicate + "}"
    target = tmp_path / "report.json"

    def fake_run(*args, **kwargs):
        target.write_text(serialized, encoding="utf-8")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="unreadable JSON report"):
        demo.run_report("evalforge.offline_rag_demo", target)


def test_run_report_rejects_duplicate_nested_metric(tmp_path, monkeypatch):
    report = _valid_report()
    serialized = json.dumps(report)
    serialized = serialized.replace(
        '"lexical_evidence_hit_rate": 0.5',
        '"lexical_evidence_hit_rate": 0.5, "lexical_evidence_hit_rate": 1.0',
        1,
    )
    target = tmp_path / "report.json"

    def fake_run(*args, **kwargs):
        target.write_text(serialized, encoding="utf-8")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="unreadable JSON report"):
        demo.run_report("evalforge.offline_rag_demo", target)
