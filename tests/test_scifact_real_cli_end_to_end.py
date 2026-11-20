"""End-to-end SciFact runner contract using real installed EvalForge CLIs.

This tiny synthetic BEIR-format fixture verifies orchestration and scoring.
It is not an independent SciFact benchmark or a published performance result.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_scifact.py"


def test_scifact_runner_with_installed_benchmark_and_comparator(tmp_path):
    if not shutil.which("evalforge-benchmark-beir") or not shutil.which("evalforge-compare-beir"):
        pytest.skip("Installed EvalForge CLIs required for end-to-end integration")
    dataset = tmp_path / "dataset"
    (dataset / "qrels").mkdir(parents=True)
    corpus = dataset / "corpus.jsonl"
    queries = dataset / "queries.jsonl"
    qrels = dataset / "qrels" / "test.tsv"
    corpus.write_text(
        "\n".join(json.dumps(row) for row in [
            {"_id": "d1", "title": "Solar energy", "text": "Solar panels produce electricity"},
            {"_id": "d2", "title": "Ocean tides", "text": "Tides move seawater"},
            {"_id": "d3", "title": "Battery storage", "text": "Batteries store electricity"},
        ]) + "\n", encoding="utf-8"
    )
    queries.write_text(
        "\n".join(json.dumps(row) for row in [
            {"_id": "q1", "text": "solar panels electricity"},
            {"_id": "q2", "text": "seawater tides"},
            {"_id": "q3", "text": "unlabeled question"},
        ]) + "\n", encoding="utf-8"
    )
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()
    command = [
        sys.executable, str(SCRIPT), "--dataset", str(dataset),
        "--output-dir", str(output), "--evalforge-revision", "ci-installed-package",
        "--source-url", "https://example.org/synthetic-beir-fixture",
        "--download-date", "2026-11-20",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=90)
    assert result.returncode in (0, 1), result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for key, path in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        assert manifest["input_sha256"][key] == hashlib.sha256(path.read_bytes()).hexdigest()
    for tokenizer in ("ascii", "unicode"):
        report = json.loads((output / f"scifact-{tokenizer}.json").read_text(encoding="utf-8"))
        assert report["schema_version"] == 1
        assert report["source"]["source_sha256"] == manifest["input_sha256"]
        assert report["source"]["labeled_queries"] == 2
        assert report["source"]["excluded_queries_without_positive_qrels"] == 1
        assert set(report["evaluation"]["by_top_k"]) == {"1", "3", "5", "10"}
    comparison = json.loads((output / "scifact-comparison.json").read_text(encoding="utf-8"))
    assert isinstance(comparison, dict)
    assert result.returncode in (0, 1)  # A quality-gate failure must retain all evidence.
