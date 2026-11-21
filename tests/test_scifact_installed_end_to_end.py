"""End-to-end SciFact-shaped BEIR fixture through the installed public CLI.

This is a hand-authored compatibility fixture, NOT the SciFact benchmark.
"""
import hashlib
import json
import subprocess
from pathlib import Path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_installed_scifact_pipeline_produces_reproducible_evidence(tmp_path):
    dataset = tmp_path / "synthetic-beir"
    (dataset / "qrels").mkdir(parents=True)
    (dataset / "corpus.jsonl").write_text(
        '{"_id":"d1","title":"Astronomy","text":"Mars has two moons Phobos Deimos"}\n'
        '{"_id":"d2","title":"Botany","text":"Plants use sunlight for photosynthesis"}\n',
        encoding="utf-8",
    )
    (dataset / "queries.jsonl").write_text(
        '{"_id":"q1","text":"Mars moons"}\n'
        '{"_id":"q2","text":"photosynthesis sunlight"}\n',
        encoding="utf-8",
    )
    (dataset / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence"
    output.mkdir()
    command = [
        "evalforge-run-scifact", "--dataset", str(dataset),
        "--output-dir", str(output), "--evalforge-revision", "fixture-test",
        "--source-url", "https://example.org/synthetic-fixture",
        "--download-date", "2026-11-21", "--max-drop", "0.02",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "BEIR SciFact"
    assert manifest["input_sha256"] == {
        "corpus": _digest(dataset / "corpus.jsonl"),
        "queries": _digest(dataset / "queries.jsonl"),
        "qrels": _digest(dataset / "qrels" / "test.tsv"),
    }
    for tokenizer in ("ascii", "unicode"):
        report = json.loads((output / f"scifact-{tokenizer}.json").read_text(encoding="utf-8"))
        assert report["source"]["source_sha256"] == manifest["input_sha256"]
        assert report["benchmark_scope"] == "offline_lexical_bm25_positive_qrels_only"
        assert set(report["evaluation"]["by_top_k"]) == {"1", "3", "5", "10"}
    comparison = json.loads((output / "scifact-comparison.json").read_text(encoding="utf-8"))
    assert comparison["passed"] is True
    assert set(comparison["by_top_k"]) == {"1", "3", "5", "10"}
    assert all(row["passed"] for row in comparison["by_top_k"].values())

    # Reusing an evidence directory must not silently overwrite a prior result.
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert repeated.returncode != 0
    assert "empty" in repeated.stderr.lower()
