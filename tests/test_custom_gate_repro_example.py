"""Exercise the published consumer reproduction script against the installed CLI."""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reproducible_custom_gate_example():
    run = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "reproduce_custom_gate.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, (run.stdout, run.stderr)
    assert "PASS: deterministic installed custom gate" in run.stdout
    assert "no LLM quality/safety claim" in run.stdout
