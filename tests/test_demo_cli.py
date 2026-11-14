"""Regression tests for the public, API-free installed demo entry point."""
from __future__ import annotations

import math
import re
import subprocess
import sys

import pytest

from evalforge import demo


def test_demo_runs_offline_from_installed_module(tmp_path):
    """Run from outside the repository, without provider credentials."""
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.demo"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "EvalForge offline quickstart: PASS" in result.stdout
    assert "not LLM correctness or hallucination benchmarks" in result.stdout


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), 1.01, -0.01])
def test_demo_rejects_invalid_hit_rate(monkeypatch, bad_value):
    """Never display a PASS when a retrieval measurement is invalid."""
    def fake_report(module, destination, *options):
        method = (
            "lexical_bm25_retrieval_only"
            if module.endswith("offline_rag_demo")
            else "paired_ascii_unicode_bm25_lexical_retrieval_only"
        )
        rows = {
            str(k): {
                "lexical_evidence_hit_rate": bad_value,
                "paired_mean_recall_delta_unicode_minus_ascii": 0.0,
            }
            for k in (1, 3, 5)
        }
        return {
            "method": method,
            "question_count": 1,
            "by_top_k": rows,
            "corpus_sha256": "a" * 64,
            "questions_sha256": "b" * 64,
        }

    monkeypatch.setattr(demo, "run_report", fake_report)
    with pytest.raises(RuntimeError, match="Invalid lexical evidence hit rate"):
        demo.main()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), 1.01, -1.01])
def test_demo_rejects_invalid_paired_delta(monkeypatch, bad_value):
    def fake_report(module, destination, *options):
        method = (
            "lexical_bm25_retrieval_only"
            if module.endswith("offline_rag_demo")
            else "paired_ascii_unicode_bm25_lexical_retrieval_only"
        )
        rows = {
            str(k): {
                "lexical_evidence_hit_rate": 0.5,
                "paired_mean_recall_delta_unicode_minus_ascii": bad_value,
            }
            for k in (1, 3, 5)
        }
        return {
            "method": method,
            "question_count": 1,
            "by_top_k": rows,
            "corpus_sha256": "a" * 64,
            "questions_sha256": "b" * 64,
        }

    monkeypatch.setattr(demo, "run_report", fake_report)
    with pytest.raises(RuntimeError, match="Invalid paired recall delta"):
        demo.main()


@pytest.mark.parametrize(
    "outcome, expected",
    [
        ("timeout", "timed out after 90 seconds"),
        ("failure", "failed (exit 7): invalid fixture"),
        ("missing", "did not produce its JSON report"),
        ("malformed", "produced an unreadable JSON report"),
        ("array", "produced a non-object JSON report"),
    ],
)
def test_demo_reports_subprocess_and_artifact_failures(monkeypatch, tmp_path, outcome, expected):
    """A broken installed benchmark must fail clearly, never silently pass."""
    def fake_run(command, **kwargs):
        assert kwargs["timeout"] == 90
        destination = tmp_path / "report.json"
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 90)
        if outcome == "failure":
            return subprocess.CompletedProcess(command, 7, "", "invalid fixture")
        if outcome == "malformed":
            destination.write_text("{broken", encoding="utf-8")
        if outcome == "array":
            destination.write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        demo.run_report("evalforge.offline_rag_demo", tmp_path / "report.json")
