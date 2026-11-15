"""Regression tests for cross-run lexical artifact comparisons."""
import copy
import pytest

from evalforge.compare_artifacts import compare


def fixture():
    section = {
        "question_count": 2,
        "corpus_sha256": "a" * 64,
        "questions_sha256": "b" * 64,
    }
    return {
        "schema_version": 1,
        "benchmark_type": "offline_lexical_retrieval_only",
        "limitations": "Lexical fixture only",
        "bm25": {**section, "lexical_evidence_hit_rate_by_k": {"1": 0.5, "3": 0.75, "5": 1.0}},
        "paired_tokenizer": {**section, "recall_delta_unicode_minus_ascii_by_k": {"1": 0.0, "3": -0.1, "5": 0.2}},
    }


def test_identical_artifacts_pass():
    assert compare(fixture(), fixture())["bm25_regression_cutoffs"] == []


def test_detects_regression_and_tolerance():
    candidate = copy.deepcopy(fixture())
    candidate["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.5
    assert compare(fixture(), candidate)["bm25_regression_cutoffs"] == ["3"]
    assert compare(fixture(), candidate, tolerance=0.25)["bm25_regression_cutoffs"] == []


def test_rejects_different_fixtures():
    candidate = copy.deepcopy(fixture())
    candidate["bm25"]["corpus_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="fixtures differ"):
        compare(fixture(), candidate)


def test_rejects_nonfinite_tolerance():
    with pytest.raises(ValueError, match="Tolerance"):
        compare(fixture(), fixture(), float("nan"))


def test_paired_delta_is_diagnostic_only():
    candidate = copy.deepcopy(fixture())
    candidate["paired_tokenizer"]["recall_delta_unicode_minus_ascii_by_k"]["3"] = -0.2
    result = compare(fixture(), candidate)
    assert result["bm25_regression_cutoffs"] == []
    assert result["metric_deltas"]["paired_tokenizer"]["3"] == pytest.approx(-0.1)
