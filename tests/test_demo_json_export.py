"""CLI artifact export integration tests using the bundled offline fixtures."""
import json
import subprocess
import sys


def run_demo(*args):
    return subprocess.run(
        [sys.executable, "-m", "evalforge.demo", *map(str, args)],
        capture_output=True, text=True, timeout=120,
    )


def test_json_output_creates_parseable_artifact(tmp_path):
    target = tmp_path / "evaluation.json"
    result = run_demo("--json", "--output", target)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark_type"] == "offline_lexical_retrieval_only"
    assert set(payload["bm25"]["lexical_evidence_hit_rate_by_k"]) == {"1", "3", "5"}


def test_json_output_never_overwrites_existing_artifact(tmp_path):
    target = tmp_path / "evaluation.json"
    target.write_text("preserve this", encoding="utf-8")
    result = run_demo("--json", "--output", target)
    assert result.returncode != 0
    assert target.read_text(encoding="utf-8") == "preserve this"
    assert result.stdout == ""


def test_output_requires_json(tmp_path):
    target = tmp_path / "evaluation.json"
    result = run_demo("--output", target)
    assert result.returncode != 0
    assert "--output requires --json" in result.stderr
    assert not target.exists()
