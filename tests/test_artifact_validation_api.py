"""Regression tests for Python callers of the artifact quality gate."""
import math

import pytest

from evalforge.artifact_validation import validate


@pytest.fixture
def valid_report():
    section = {
        "question_count": 2,
        "corpus_sha256": "a" * 64,
        "questions_sha256": "b" * 64,
    }
    return {
        "schema_version": 1,
        "benchmark_type": "offline_lexical_retrieval_only",
        "limitations": "Small lexical fixture; not model safety.",
        "bm25": {
            **section,
            "lexical_evidence_hit_rate_by_k": {"1": 0.5, "3": 0.75, "5": 1.0},
        },
        "paired_tokenizer": {
            **section,
            "recall_delta_unicode_minus_ascii_by_k": {"1": 0.0, "3": -0.2, "5": 0.1},
        },
    }


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1, True, "0.5", None])
def test_python_api_rejects_invalid_thresholds(valid_report, threshold):
    with pytest.raises(ValueError, match="Minimum hit rate"):
        validate(valid_report, threshold)


def test_python_api_accepts_valid_threshold(valid_report):
    validate(valid_report, 0.75)


def test_python_api_enforces_quality_gate(valid_report):
    with pytest.raises(ValueError, match="below"):
        validate(valid_report, 0.76)


@pytest.mark.parametrize("rates", [
    {"1": 0.8, "3": 0.6, "5": 0.9},
    {"1": 0.1, "3": 0.9, "5": 0.7},
])
def test_python_api_rejects_decreasing_hit_rate(valid_report, rates):
    valid_report["bm25"]["lexical_evidence_hit_rate_by_k"] = rates
    with pytest.raises(ValueError, match="must not decrease"):
        validate(valid_report, 0.0)


def test_python_api_allows_equal_hit_rates(valid_report):
    valid_report["bm25"]["lexical_evidence_hit_rate_by_k"] = {
        "1": 0.5, "3": 0.5, "5": 0.5,
    }
    validate(valid_report, 0.5)
