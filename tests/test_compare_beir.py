"""Regression checks for source-safe BEIR report comparison."""
import copy
import json

from evalforge.beir_benchmark import benchmark
from evalforge.compare_beir import compare, main
from tests.test_beir_benchmark import sources


def reports(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    return benchmark(corpus, queries, qrels, cutoffs=(1,))


def test_identical_reports_pass(tmp_path):
    baseline = reports(tmp_path)
    result = compare(baseline, copy.deepcopy(baseline), k=1)
    assert result["passed"] is True
    assert result["delta"] == 0


def test_regression_fails_and_tolerance_passes(tmp_path):
    baseline = reports(tmp_path)
    candidate = copy.deepcopy(baseline)
    candidate["evaluation"]["by_top_k"]["1"]["lexical_evidence_hit_lower_bound"] = 0.5
    assert not compare(baseline, candidate, k=1)["passed"]
    assert compare(baseline, candidate, k=1, max_drop=0.5)["passed"]


def test_dataset_drift_is_rejected_even_if_scores_match(tmp_path):
    import pytest
    baseline = reports(tmp_path)
    candidate = copy.deepcopy(baseline)
    candidate["source"]["source_sha256"]["queries"] = "a" * 64
    with pytest.raises(ValueError, match="source files differ"):
        compare(baseline, candidate, k=1)


def test_missing_cutoff_and_invalid_score_rejected(tmp_path):
    import pytest
    baseline = reports(tmp_path)
    with pytest.raises(ValueError, match="cutoff"):
        compare(baseline, baseline, k=5)
    invalid = copy.deepcopy(baseline)
    invalid["evaluation"]["by_top_k"]["1"]["lexical_evidence_hit_lower_bound"] = float("nan")
    with pytest.raises(ValueError, match="score"):
        compare(baseline, invalid, k=1)


def test_cli_exit_codes(tmp_path, capsys):
    baseline = reports(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(baseline), encoding="utf-8")
    candidate = copy.deepcopy(baseline)
    candidate["evaluation"]["by_top_k"]["1"]["lexical_evidence_hit_lower_bound"] = 0
    second.write_text(json.dumps(candidate), encoding="utf-8")
    args = [str(first), str(second), "--k", "1"]
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
    assert main(args + ["--max-drop", "1"]) == 0
    assert main(args + ["--max-drop", "-1"]) == 2

def test_rejects_inconsistent_counts_and_missing_tokenizer(tmp_path):
    import pytest

    baseline = reports(tmp_path)
    candidate = copy.deepcopy(baseline)
    candidate["source"]["labeled_queries"] += 1
    with pytest.raises(ValueError, match="counts do not reconcile"):
        compare(baseline, candidate, k=1)

    candidate = copy.deepcopy(baseline)
    candidate["evaluation"]["parameters"] = None
    with pytest.raises(ValueError, match="tokenizer configuration"):
        compare(baseline, candidate, k=1)

    candidate = copy.deepcopy(baseline)
    candidate["source"]["labeled_queries"] = True
    with pytest.raises(ValueError, match="source query count"):
        compare(baseline, candidate, k=1)
