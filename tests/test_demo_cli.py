"""Regression tests for the public, API-free installed demo entry point."""
from __future__ import annotations

import json
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


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), 1.01, -0.01, True, False])
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
        demo.main([])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), 1.01, -1.01, True, False])
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
        demo.main([])


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


@pytest.mark.parametrize(
    "module, payload, expected",
    [
        ("evalforge.offline_rag_demo", {}, "invalid benchmark cutoffs"),
        ("evalforge.offline_rag_demo", {"by_top_k": {"1": {}, "3": {}, "5": {}}}, "missing metric lexical_evidence_hit_rate"),
        ("evalforge.compare_lexical_baselines", {"by_top_k": {"1": {}, "3": {}, "5": {}}}, "missing metric paired_mean_recall_delta_unicode_minus_ascii"),
        ("evalforge.offline_rag_demo", {"by_top_k": {"1": [], "3": {}, "5": {}}}, "non-object metric row"),
    ],
)
def test_demo_rejects_invalid_report_schema(monkeypatch, tmp_path, module, payload, expected):
    """Malformed benchmark reports fail with actionable errors, not KeyError."""
    import json

    def fake_run(command, **kwargs):
        (tmp_path / "report.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        demo.run_report(module, tmp_path / "report.json")


@pytest.mark.parametrize("bad_count", [None, 0, -1, True, 1.5, "3"])
def test_demo_rejects_invalid_question_counts(monkeypatch, tmp_path, bad_count):
    """A successful benchmark must report a positive integer question count."""
    import json

    payload = {
        "question_count": bad_count,
        "by_top_k": {
            str(k): {"lexical_evidence_hit_rate": 0.5} for k in (1, 3, 5)
        },
        "corpus_sha256": "a" * 64,
        "questions_sha256": "b" * 64,
    }

    def fake_run(command, **kwargs):
        (tmp_path / "report.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="invalid question_count"):
        demo.run_report("evalforge.offline_rag_demo", tmp_path / "report.json")


def test_demo_does_not_print_partial_metrics_on_late_failure(monkeypatch, capsys):
    """Invalid later cutoffs must not leak a misleading partial success summary."""
    def fake_report(module, destination, *options):
        paired = module.endswith("compare_lexical_baselines")
        rows = {
            str(k): {
                "lexical_evidence_hit_rate": 0.5,
                "paired_mean_recall_delta_unicode_minus_ascii": (
                    float("nan") if paired and k == 5 else 0.0
                ),
            }
            for k in (1, 3, 5)
        }
        return {
            "method": ("paired_ascii_unicode_bm25_lexical_retrieval_only"
                       if paired else "lexical_bm25_retrieval_only"),
            "question_count": 3,
            "by_top_k": rows,
            "corpus_sha256": "a" * 64,
            "questions_sha256": "b" * 64,
        }

    monkeypatch.setattr(demo, "run_report", fake_report)
    with pytest.raises(RuntimeError, match="Invalid paired recall delta at k=5"):
        demo.main([])
    assert capsys.readouterr().out == ""


def test_demo_json_output_from_installed_module(tmp_path):
    """Machine-readable CLI output has stable keys and no human text."""
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.demo", "--json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["benchmark_type"] == "offline_lexical_retrieval_only"
    assert report["bm25"]["question_count"] == 14
    assert report["paired_tokenizer"]["question_count"] == 6
    assert set(report["bm25"]["lexical_evidence_hit_rate_by_k"]) == {"1", "3", "5"}
    assert set(report["paired_tokenizer"]["recall_delta_unicode_minus_ascii_by_k"]) == {"1", "3", "5"}
    for group in ("bm25", "paired_tokenizer"):
        for key in ("corpus_sha256", "questions_sha256"):
            assert len(report[group][key]) == 64
    assert "PASS" not in result.stdout
    assert "not LLM correctness" in report["limitations"]


def test_demo_json_never_prints_partial_output_on_invalid_metric(monkeypatch, capsys):
    """JSON consumers must never receive a misleading success document."""
    def fake_report(module, destination, *options):
        paired = module.endswith("compare_lexical_baselines")
        return {
            "method": ("paired_ascii_unicode_bm25_lexical_retrieval_only"
                       if paired else "lexical_bm25_retrieval_only"),
            "question_count": 1,
            "by_top_k": {
                str(k): {
                    "lexical_evidence_hit_rate": 0.5,
                    "paired_mean_recall_delta_unicode_minus_ascii": (
                        float("nan") if paired and k == 5 else 0.0
                    ),
                }
                for k in (1, 3, 5)
            },
            "corpus_sha256": "a" * 64,
            "questions_sha256": "b" * 64,
        }
    monkeypatch.setattr(demo, "run_report", fake_report)
    with pytest.raises(RuntimeError, match="Invalid paired recall delta at k=5"):
        demo.main(["--json"])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "-0.1", "1.1", "nope"])
def test_demo_rejects_invalid_quality_gate(bad):
    with pytest.raises(SystemExit) as error:
        demo.main(["--min-hit-rate", bad])
    assert error.value.code == 2


def test_demo_quality_gate_pass_and_fail(monkeypatch, capsys):
    def fake_report(module, destination, *options):
        paired = module.endswith("compare_lexical_baselines")
        return {
            "method": ("paired_ascii_unicode_bm25_lexical_retrieval_only"
                       if paired else "lexical_bm25_retrieval_only"),
            "question_count": 3,
            "corpus_sha256": "a" * 64,
            "questions_sha256": "b" * 64,
            "by_top_k": {
                str(k): {
                    "lexical_evidence_hit_rate": 0.5,
                    "paired_mean_recall_delta_unicode_minus_ascii": 0.0,
                } for k in (1, 3, 5)
            },
        }

    monkeypatch.setattr(demo, "run_report", fake_report)
    assert demo.main(["--json", "--min-hit-rate", "0.5"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1
    with pytest.raises(RuntimeError, match="below required 0.6"):
        demo.main(["--json", "--min-hit-rate", "0.6"])
    assert capsys.readouterr().out == ""


def test_demo_version_matches_installed_distribution(tmp_path):
    """CLI version and JSON provenance must agree with installed wheel metadata."""
    from importlib.metadata import version

    expected = version("evalforge-ai")
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.demo", "--version"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"demo.py {expected}"

    result = subprocess.run(
        [sys.executable, "-m", "evalforge.demo", "--json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["evalforge_version"] == expected
