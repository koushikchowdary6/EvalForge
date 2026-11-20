"""Packaging contract for the installable SciFact benchmark runner."""
import subprocess
import sys
from pathlib import Path


def test_scifact_module_has_working_help():
    completed = subprocess.run(
        [sys.executable, "-m", "evalforge.scifact_runner", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset" in completed.stdout
    assert "--timeout-seconds" in completed.stdout
    assert "--max-drop" in completed.stdout


def test_scifact_installed_entry_point_declared():
    config = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert 'evalforge-run-scifact = "evalforge.scifact_runner:main"' in config
