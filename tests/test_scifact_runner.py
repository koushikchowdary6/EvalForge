"""Smoke tests for the standalone SciFact benchmark runner preflight."""
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_scifact.py"


def test_missing_dataset_rejected_before_writing(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--dataset", str(tmp_path / "missing"),
            "--output-dir", str(output), "--evalforge-revision", "test",
            "--source-url", "https://example.org/dataset",
            "--download-date", "2026-11-19",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "Missing SciFact input files" in result.stderr
    assert list(output.iterdir()) == []


def test_existing_artifacts_are_not_overwritten(tmp_path):
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "manifest.json"
    marker.write_text("original", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--dataset", str(dataset),
            "--output-dir", str(output), "--evalforge-revision", "test",
            "--source-url", "https://example.org/dataset",
            "--download-date", "2026-11-19",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "must be empty" in result.stderr
    assert marker.read_text(encoding="utf-8") == "original"


def test_noncanonical_and_impossible_download_dates_rejected(tmp_path):
    """Provenance dates must be real calendar dates in the documented format."""
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()

    for bad_date in ("20261120", "2026-11-31", "2026-1-02", "2026-W47-5"):
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT), "--dataset", str(dataset),
                "--output-dir", str(output), "--evalforge-revision", "test",
                "--source-url", "https://example.org/dataset",
                "--download-date", bad_date,
            ],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 2, (bad_date, result.stderr)
        assert "--download-date" in result.stderr
        assert list(output.iterdir()) == [], "Invalid dates must not create evidence"
