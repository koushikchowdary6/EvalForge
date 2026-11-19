"""Multi-cutoff BEIR comparison must not hide regressions at other k."""
import copy
import json

import pytest

from evalforge.beir_benchmark import benchmark
from evalforge.compare_beir import compare_all, main
from tests.test_beir_benchmark import sources


def _report(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    return benchmark(corpus, queries, qrels, cutoffs=(1, 3, 5))


def test_all_cutoffs_pass_for_identical_reports(tmp_path):
    baseline = _report(tmp_path)
    result = compare_all(baseline, copy.deepcopy(baseline))
    assert result["passed"] is True
    assert list(result["by_top_k"]) == ["1", "3", "5"]
    assert all(item["passed"] for item in result["by_top_k"].values())


def test_regression_at_one_cutoff_fails_entire_gate(tmp_path):
    baseline = _report(tmp_path)
    candidate = copy.deepcopy(baseline)
    candidate["evaluation"]["by_top_k"]["3"]["lexical_evidence_hit_lower_bound"] = 0
    result = compare_all(baseline, candidate)
    assert result["passed"] is False
    assert result["by_top_k"]["3"]["passed"] is False


def test_mismatched_cutoffs_are_rejected(tmp_path):
    baseline = _report(tmp_path)
    candidate = copy.deepcopy(baseline)
    del candidate["evaluation"]["by_top_k"]["5"]
    with pytest.raises(ValueError, match="identical"):
        compare_all(baseline, candidate)


@pytest.mark.parametrize("invalid", ["0", "01", "x"])
def test_invalid_cutoff_keys_are_rejected(tmp_path, invalid):
    baseline = _report(tmp_path)
    baseline["evaluation"]["by_top_k"][invalid] = baseline["evaluation"]["by_top_k"]["1"]
    with pytest.raises(ValueError, match="cutoff"):
        compare_all(baseline, baseline)


def test_cli_all_cutoffs_and_mutually_exclusive_flags(tmp_path, capsys):
    baseline = _report(tmp_path)
    candidate = copy.deepcopy(baseline)
    candidate["evaluation"]["by_top_k"]["3"]["lexical_evidence_hit_lower_bound"] = 0
    a, b = tmp_path / "baseline.json", tmp_path / "candidate.json"
    a.write_text(json.dumps(baseline), encoding="utf-8")
    b.write_text(json.dumps(candidate), encoding="utf-8")
    assert main([str(a), str(b), "--all-k"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is False
    assert set(result["by_top_k"]) == {"1", "3", "5"}
    assert main([str(a), str(a), "--all-k"]) == 0
