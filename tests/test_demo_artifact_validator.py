"""Offline artifact validator regression tests (no provider credentials)."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "examples" / "validate_demo_artifact.py"
spec = importlib.util.spec_from_file_location("validate_demo_artifact", MODULE)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def sample():
    return {
        "schema_version": 1,
        "evalforge_version": "0.1.0",
        "benchmark_type": "offline_lexical_retrieval_only",
        "limitations": "Not a model quality benchmark",
        "bm25": {
            "question_count": 14,
            "corpus_sha256": "a" * 64,
            "questions_sha256": "b" * 64,
            "lexical_evidence_hit_rate_by_k": {"1": 0.4, "3": 0.6, "5": 0.8},
        },
        "paired_tokenizer": {
            "question_count": 6,
            "corpus_sha256": "c" * 64,
            "questions_sha256": "d" * 64,
            "recall_delta_unicode_minus_ascii_by_k": {"1": 0.1, "3": 0, "5": -0.2},
        },
    }


def test_valid_report_and_threshold():
    validator.validate(sample(), 0.6, "0.1.0")


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(schema_version=True),
    lambda r: r.update(benchmark_type="provider_llm_safety"),
    lambda r: r["bm25"].update(corpus_sha256="bad"),
    lambda r: r["bm25"].update(question_count=True),
    lambda r: r["bm25"]["lexical_evidence_hit_rate_by_k"].update({"3": True}),
    lambda r: r["bm25"]["lexical_evidence_hit_rate_by_k"].update({"3": float("nan")}),
    lambda r: r["paired_tokenizer"]["recall_delta_unicode_minus_ascii_by_k"].update({"3": 2}),
    lambda r: r["bm25"]["lexical_evidence_hit_rate_by_k"].pop("5"),
])
def test_reject_invalid_artifact(mutation):
    report = sample()
    mutation(report)
    with pytest.raises(ValueError):
        validator.validate(report, 0.0)


def test_reject_below_gate_and_version_mismatch():
    with pytest.raises(ValueError, match="below"):
        validator.validate(sample(), 0.7)
    with pytest.raises(ValueError, match="version"):
        validator.validate(sample(), 0.0, "9.9.9")


def test_duplicate_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        json.loads('{"bm25": {"question_count": 14, "question_count": 15}}',
                   object_pairs_hook=validator._unique_keys)
