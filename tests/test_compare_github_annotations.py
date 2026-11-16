"""GitHub Actions annotation output remains opt-in and preserves JSON contracts."""
import copy
import json

from evalforge.compare_artifacts import main
from tests.test_compare_artifacts import fixture


def test_annotations_on_regression_keep_json_stdout(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(fixture()), encoding="utf-8")
    changed = copy.deepcopy(fixture())
    changed["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.5
    candidate.write_text(json.dumps(changed), encoding="utf-8")

    assert main([str(baseline), str(candidate), "--github-annotations"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["bm25_regression_cutoffs"] == ["3"]
    assert "::error title=EvalForge BM25 regression at k=3::" in captured.err
    assert "0.750000 to 0.500000" in captured.err


def test_annotations_do_not_emit_on_pass(tmp_path, capsys):
    artifact = tmp_path / "baseline.json"
    artifact.write_text(json.dumps(fixture()), encoding="utf-8")
    assert main([str(artifact), str(artifact), "--github-annotations"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["bm25_regression_cutoffs"] == []
    assert captured.err == ""


def test_default_regression_does_not_emit_annotations(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(fixture()), encoding="utf-8")
    changed = copy.deepcopy(fixture())
    changed["bm25"]["lexical_evidence_hit_rate_by_k"]["1"] = 0.0
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    assert main([str(baseline), str(candidate)]) == 1
    captured = capsys.readouterr()
    assert "::error" not in captured.err
