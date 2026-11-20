"""Regression tests for opt-in tokenizer parity in BEIR comparisons."""
import copy
import json

from evalforge.compare_beir import _require_same_tokenizer, main


def _report(tokenizer="ascii_alphanumeric"):
    return {
        "schema_version": 1,
        "benchmark_scope": "offline_lexical_bm25_positive_qrels_only",
        "source": {
            "source_sha256": {"corpus": "a" * 64, "queries": "b" * 64, "qrels": "c" * 64},
            "source_queries": 1,
            "labeled_queries": 1,
            "excluded_queries_without_positive_qrels": 0,
        },
        "evaluation": {
            "method": "lexical_bm25_retrieval_only",
            "corpus_size": 2,
            "question_count": 1,
            "parameters": {"tokenizer": tokenizer},
            "by_top_k": {"1": {"lexical_evidence_hit_lower_bound": 0.5}},
        },
    }


def test_tokenizer_parity_accepts_identical_configurations():
    _require_same_tokenizer(_report(), copy.deepcopy(_report()))


def test_tokenizer_parity_rejects_drift():
    import pytest
    with pytest.raises(ValueError, match="tokenizer configurations differ"):
        _require_same_tokenizer(_report(), _report("unicode_alphanumeric"))


def test_tokenizer_parity_rejects_missing_configuration():
    import pytest
    candidate = _report()
    del candidate["evaluation"]["parameters"]["tokenizer"]
    with pytest.raises(ValueError, match="missing tokenizer"):
        _require_same_tokenizer(_report(), candidate)


def test_cli_optional_parity_gate_preserves_deliberate_experiment(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_report()), encoding="utf-8")
    candidate.write_text(json.dumps(_report("unicode_alphanumeric")), encoding="utf-8")
    args = [str(baseline), str(candidate), "--all-k"]
    assert main(args) == 0
    assert main(args + ["--require-same-tokenizer"]) == 2


def test_cli_same_tokenizer_parity_passes(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_report()), encoding="utf-8")
    candidate.write_text(json.dumps(_report()), encoding="utf-8")
    assert main([str(baseline), str(candidate), "--k", "1", "--require-same-tokenizer"]) == 0
