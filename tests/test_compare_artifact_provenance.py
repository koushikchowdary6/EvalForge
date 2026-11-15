"""Regression coverage for auditable offline comparison decisions."""
from copy import deepcopy

from evalforge.compare_artifacts import compare


def _report(version="0.1.0"):
    digest_a, digest_b = "a" * 64, "b" * 64
    return {
        "schema_version": 1,
        "benchmark_type": "offline_lexical_retrieval_only",
        "evalforge_version": version,
        "limitations": "Offline lexical fixture only; not LLM correctness.",
        "bm25": {
            "question_count": 14,
            "corpus_sha256": digest_a,
            "questions_sha256": digest_b,
            "lexical_evidence_hit_rate_by_k": {"1": 0.5, "3": 0.75, "5": 0.8},
        },
        "paired_tokenizer": {
            "question_count": 6,
            "corpus_sha256": digest_b,
            "questions_sha256": digest_a,
            "recall_delta_unicode_minus_ascii_by_k": {"1": 0.1, "3": 0.2, "5": 0.3},
        },
    }


def test_comparison_includes_versions_and_both_dataset_fingerprints():
    baseline = _report("0.1.0")
    candidate = _report("0.2.0")
    result = compare(baseline, candidate)
    provenance = result["provenance"]
    assert provenance["baseline_version"] == "0.1.0"
    assert provenance["candidate_version"] == "0.2.0"
    for section in ("bm25", "paired_tokenizer"):
        assert provenance["datasets"][section] == {
            field: baseline[section][field]
            for field in ("question_count", "corpus_sha256", "questions_sha256")
        }
    assert result["bm25_regression_cutoffs"] == []


def test_provenance_preserved_when_comparison_detects_regression():
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.25
    result = compare(baseline, candidate)
    assert result["bm25_regression_cutoffs"] == ["3"]
    assert result["provenance"]["datasets"]["bm25"]["corpus_sha256"] == "a" * 64
