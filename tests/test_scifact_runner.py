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


def test_benchmark_subprocess_failure_uses_error_exit_code(tmp_path, monkeypatch, capsys):
    """A failed benchmark must never masquerade as a comparison gate failure."""
    from evalforge import scifact_runner

    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()

    def fail_benchmark(command, **kwargs):
        assert command[0] == "evalforge-benchmark-beir"
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(scifact_runner.subprocess, "run", fail_benchmark)
    result = scifact_runner.main([
        "--dataset", str(dataset), "--output-dir", str(output),
        "--evalforge-revision", "test", "--source-url", "https://example.org/dataset",
        "--download-date", "2026-11-20",
    ])
    assert result == 2
    assert "benchmark failed with exit 7" in capsys.readouterr().err
    assert (output / "manifest.json").is_file()
    assert not (output / "scifact-comparison.json").exists()


def test_benchmark_launch_oserror_is_reported(tmp_path, monkeypatch, capsys):
    """Permission and executable launch failures return 2 without a traceback."""
    from evalforge import scifact_runner

    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()

    def denied(command, **kwargs):
        raise PermissionError("executable not permitted")

    monkeypatch.setattr(scifact_runner.subprocess, "run", denied)
    result = scifact_runner.main([
        "--dataset", str(dataset), "--output-dir", str(output),
        "--evalforge-revision", "test", "--source-url", "https://example.org/dataset",
        "--download-date", "2026-11-20",
    ])
    assert result == 2
    assert "Could not launch ascii benchmark" in capsys.readouterr().err
    assert (output / "manifest.json").is_file()
