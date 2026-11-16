"""GitHub Actions step summaries are opt-in, auditable and preserve exit codes."""
import copy
import json

import pytest

from evalforge.compare_artifacts import main
from tests.test_compare_artifacts import fixture


def _artifacts(tmp_path, regressed=False):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    original = fixture()
    changed = copy.deepcopy(original)
    if regressed:
        changed["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.5
    baseline.write_text(json.dumps(original), encoding="utf-8")
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    return baseline, candidate


@pytest.mark.parametrize("regressed,expected", [(False, 0), (True, 1)])
def test_step_summary_includes_scores_and_gate(tmp_path, monkeypatch, capsys, regressed, expected):
    baseline, candidate = _artifacts(tmp_path, regressed)
    summary = tmp_path / "summary.md"
    summary.write_text("Existing job summary\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert main([str(baseline), str(candidate), "--github-step-summary"]) == expected
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("Existing job summary\n")
    assert "| k=3 | 0.750000 |" in text
    assert f"**Gate:** {'FAIL' if regressed else 'PASS'}" in text
    assert json.loads(capsys.readouterr().out)["bm25_regression_cutoffs"] == (["3"] if regressed else [])


def test_step_summary_requires_environment_variable(tmp_path, monkeypatch, capsys):
    baseline, candidate = _artifacts(tmp_path)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main([str(baseline), str(candidate), "--github-step-summary"])
    assert exc.value.code == 2
    assert "GITHUB_STEP_SUMMARY is not set" in capsys.readouterr().err


def test_default_does_not_touch_summary(tmp_path, monkeypatch):
    baseline, candidate = _artifacts(tmp_path)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert main([str(baseline), str(candidate)]) == 0
    assert not summary.exists()
