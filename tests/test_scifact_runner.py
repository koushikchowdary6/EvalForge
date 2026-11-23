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
    import json
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "output_sha256" not in manifest, "Failed runs must not claim complete evidence"


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


def test_unreadable_input_fingerprint_returns_error_without_evidence(tmp_path, monkeypatch, capsys):
    """An input read failure is an execution error, never a valid benchmark."""
    from evalforge import scifact_runner

    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()

    def unreadable(path):
        raise PermissionError("source file cannot be read")

    monkeypatch.setattr(scifact_runner, "sha256", unreadable)
    result = scifact_runner.main([
        "--dataset", str(dataset), "--output-dir", str(output),
        "--evalforge-revision", "test", "--source-url", "https://example.org/dataset",
        "--download-date", "2026-11-20",
    ])
    assert result == 2
    assert "Cannot fingerprint SciFact inputs" in capsys.readouterr().err
    assert list(output.iterdir()) == []


def test_dataset_mutation_during_benchmark_invalidates_evidence(tmp_path, monkeypatch, capsys):
    """A benchmark cannot certify results against inputs changed after fingerprinting."""
    import json
    from evalforge import scifact_runner

    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("original\\n", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()

    def mutate_input(command, **kwargs):
        assert command[0] == "evalforge-benchmark-beir"
        (dataset / "queries.jsonl").write_text("changed\\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(scifact_runner.subprocess, "run", mutate_input)
    status = scifact_runner.main([
        "--dataset", str(dataset), "--output-dir", str(output),
        "--evalforge-revision", "test", "--source-url", "https://example.org/dataset",
        "--download-date", "2026-11-20",
    ])
    assert status == 2
    assert "input changed during evaluation" in capsys.readouterr().err
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "output_sha256" not in manifest
    assert not (output / "scifact-comparison.json").exists()


def test_preflight_fingerprints_inputs_without_writing_evidence(tmp_path):
    """Preflight is a safe reproducibility check, not a benchmark result."""
    import hashlib
    import json

    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("fixture\\n", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--dataset", str(dataset),
            "--output-dir", str(output), "--evalforge-revision", "test",
            "--source-url", "https://example.org/dataset",
            "--download-date", "2026-11-22", "--preflight-only",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "inputs_fingerprinted_only"
    assert report["source_sha256"] == {
        key: hashlib.sha256((dataset / filename).read_bytes()).hexdigest()
        for key, filename in (
            ("corpus", "corpus.jsonl"),
            ("queries", "queries.jsonl"),
            ("qrels", "qrels/test.tsv"),
        )
    }
    assert list(output.iterdir()) == []


def test_preflight_rejects_missing_dataset_without_evidence(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--dataset", str(tmp_path / "missing"),
            "--output-dir", str(output), "--evalforge-revision", "test",
            "--source-url", "https://example.org/dataset",
            "--download-date", "2026-11-22", "--preflight-only",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "Missing SciFact input files" in result.stderr
    assert not list(output.iterdir())


def test_preflight_rejects_symlinked_inputs(tmp_path):
    """Dataset preflight must not follow links to unrelated local files."""
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"_id": "outside"}\\n', encoding="utf-8")
    (dataset / "corpus.jsonl").symlink_to(outside)
    (dataset / "queries.jsonl").write_text("", encoding="utf-8")
    (dataset / "qrels" / "test.tsv").write_text("", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(dataset), "--preflight-only"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "not links or special files" in result.stderr
    assert not result.stdout


def test_full_run_rejects_symlinked_inputs_without_artifacts(tmp_path):
    """A linked corpus must be rejected before the provenance manifest is written."""
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    target = tmp_path / "outside.jsonl"
    target.write_text("", encoding="utf-8")
    (dataset / "corpus.jsonl").symlink_to(target)
    (dataset / "queries.jsonl").write_text("", encoding="utf-8")
    (dataset / "qrels" / "test.tsv").write_text("", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(dataset),
         "--output-dir", str(output), "--evalforge-revision", "test",
         "--source-url", "https://example.org/dataset",
         "--download-date", "2026-11-22"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "not links or special files" in result.stderr
    assert not list(output.iterdir())



def test_preflight_rejects_symlinked_dataset_root(tmp_path):
    """A linked dataset directory must not bypass leaf-file checks."""
    actual = tmp_path / "actual"
    (actual / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (actual / filename).write_text("", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(linked), "--preflight-only"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "not links or special files" in result.stderr
    assert not result.stdout


def test_full_run_rejects_symlinked_qrels_directory(tmp_path):
    """Reject a linked subdirectory before writing any benchmark evidence."""
    dataset = tmp_path / "scifact"
    dataset.mkdir()
    external = tmp_path / "external_qrels"
    external.mkdir()
    (external / "test.tsv").write_text("", encoding="utf-8")
    (dataset / "qrels").symlink_to(external, target_is_directory=True)
    for filename in ("corpus.jsonl", "queries.jsonl"):
        (dataset / filename).write_text("", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(dataset),
         "--output-dir", str(output), "--evalforge-revision", "test",
         "--source-url", "https://example.org/dataset",
         "--download-date", "2026-11-23"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "not links or special files" in result.stderr
    assert not list(output.iterdir())



def test_full_run_rejects_symlinked_output_directory(tmp_path):
    """Never redirect benchmark artifacts through an empty output symlink."""
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("", encoding="utf-8")
    actual_output = tmp_path / "elsewhere"
    actual_output.mkdir()
    linked_output = tmp_path / "evidence"
    linked_output.symlink_to(actual_output, target_is_directory=True)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(dataset),
         "--output-dir", str(linked_output), "--evalforge-revision", "test",
         "--source-url", "https://example.org/dataset",
         "--download-date", "2026-11-23"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "symlink" in result.stderr
    assert not list(actual_output.iterdir())


def test_full_run_rejects_symlinked_output_ancestor(tmp_path):
    """A real leaf beneath a linked parent is also unsafe for evidence."""
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / filename).write_text("", encoding="utf-8")
    actual_parent = tmp_path / "actual"
    output = actual_parent / "evidence"
    output.mkdir(parents=True)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(dataset),
         "--output-dir", str(linked_parent / "evidence"),
         "--evalforge-revision", "test",
         "--source-url", "https://example.org/dataset",
         "--download-date", "2026-11-23"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "symlink" in result.stderr
    assert not list(output.iterdir())
