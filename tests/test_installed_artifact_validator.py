"""Regression tests for the installed artifact-validation entry point."""
import json
from pathlib import Path
from evalforge.artifact_validation import main


def _artifact():
    return {
        "schema_version": 1,
        "evalforge_version": "0.1.0",
        "benchmark_type": "offline_lexical_retrieval_only",
        "limitations": "Lexical retrieval only",
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
            "recall_delta_unicode_minus_ascii_by_k": {"1": 0.1, "3": 0.0, "5": -0.2},
        },
    }


def test_installed_validator_accepts_valid_report(tmp_path, capsys):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    assert main([str(path), "--min-hit-rate", "0.6"]) == 0
    assert "VALID" in capsys.readouterr().out


def test_installed_validator_rejects_failed_gate(tmp_path, capsys):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    import pytest
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--min-hit-rate", "0.7"])
    assert exc.value.code == 1
    assert "below" in capsys.readouterr().err


def test_installed_validator_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    import pytest
    with pytest.raises(SystemExit) as exc:
        main([str(path)])
    assert exc.value.code == 1
