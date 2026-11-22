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


def test_manual_dataset_override_requires_pinned_checksum():
    """Keep manual mirror overrides opt-in, HTTPS-only, and content-pinned."""
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "real-scifact-benchmark.yml"
    ).read_text(encoding="utf-8")

    assert "dataset_url:" in workflow
    assert "dataset_sha256:" in workflow
    assert 'SCIFACT_SHA256: ${{ inputs.dataset_sha256 || \'\' }}' in workflow
    assert 'parsed.scheme != "https"' in workflow
    assert "url != official_url and not expected_sha" in workflow
    assert 're.fullmatch(r"[0-9a-f]{64}", expected_sha)' in workflow
    assert "hmac.compare_digest(digest.hexdigest(), expected_sha)" in workflow
    assert "archive.unlink(missing_ok=True)" in workflow
    assert '--source-url "$SCIFACT_URL"' in workflow
