"""Guard the real-data workflow's output-directory contract.

The runner rejects a nonempty output directory. Acquisition diagnostics must
not be placed there before the benchmark starts.
"""
from pathlib import Path


def test_scifact_workflow_keeps_acquisition_logs_outside_evidence_directory():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "real-scifact-benchmark.yml"
    ).read_text(encoding="utf-8")

    assert 'diagnostics = Path("scifact-acquisition")' in workflow
    assert 'acquisition_log = diagnostics / "dataset-acquisition.txt"' in workflow
    assert 'evidence = Path("scifact-evidence")' not in workflow
    assert "mkdir -p scifact-evidence" in workflow
    assert "scifact-acquisition/" in workflow
    assert "scifact-evidence/" in workflow
