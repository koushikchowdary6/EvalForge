"""Contract tests for the installed Markdown summary renderer."""
import json
from pathlib import Path

from evalforge.report_markdown import main, render


def sample():
    digest = "a" * 64
    return {
        "schema_version": 1,
        "benchmark_type": "offline_lexical_retrieval_only",
        "evalforge_version": "0.1.0",
        "limitations": "Synthetic lexical fixture only",
        "bm25": {
            "question_count": 5, "corpus_sha256": digest, "questions_sha256": digest,
            "lexical_evidence_hit_rate_by_k": {"1": 0.5, "3": 0.75, "5": None},
        },
        "paired_tokenizer": {
            "question_count": 4, "corpus_sha256": digest, "questions_sha256": digest,
            "recall_delta_unicode_minus_ascii_by_k": {"1": -0.25, "3": 0.0, "5": None},
        },
    }


def test_render_includes_provenance_and_explicit_scope():
    output = render(sample())
    assert "| 1 | 0.500 | -0.250 |" in output
    assert "| 5 | undefined | undefined |" in output
    assert "a" * 64 in output
    assert "not model safety" in output


def test_cli_refuses_overwrite_and_rejects_invalid_artifact(tmp_path: Path, capsys):
    source = tmp_path / "artifact.json"
    destination = tmp_path / "summary.md"
    source.write_text(json.dumps(sample()), encoding="utf-8")
    assert main([str(source), "--output", str(destination)]) == 0
    original = destination.read_text(encoding="utf-8")
    assert main([str(source), "--output", str(destination)]) == 1
    assert destination.read_text(encoding="utf-8") == original
    source.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    assert main([str(source)]) == 1
    assert "Duplicate JSON key" in capsys.readouterr().err


def test_rejects_bad_metric_without_output(tmp_path: Path):
    source = tmp_path / "invalid.json"
    destination = tmp_path / "summary.md"
    payload = sample()
    payload["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 1.5
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert main([str(source), "--output", str(destination)]) == 1
    assert not destination.exists()
