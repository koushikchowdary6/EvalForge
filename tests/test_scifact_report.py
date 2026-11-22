"""Product-facing Markdown reporting for verified SciFact evidence."""
import json

import pytest

from evalforge import scifact_report


def test_render_uses_verified_comparison_scores(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comparison = {
        "max_allowed_drop": 0.02,
        "passed": False,
        "by_top_k": {
            str(k): {
                "baseline_hit_lower_bound": 0.7,
                "candidate_hit_lower_bound": 0.6,
                "delta": -0.1,
                "passed": False,
            }
            for k in (1, 3, 5, 10)
        },
    }
    (evidence / "scifact-comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    manifest = {
        "source_url": "https://example.org/scifact",
        "evalforge_revision": "abc123",
        "download_date": "2026-11-21",
        "input_sha256": {"corpus": "a" * 64, "queries": "b" * 64, "qrels": "c" * 64},
    }
    calls = []
    def fake_verify(path, dataset):
        calls.append((path, dataset))
        return manifest
    monkeypatch.setattr(scifact_report, "verify", fake_verify)
    text = scifact_report.render(evidence)
    assert len(calls) == 1
    assert "| 1 | 0.7000 | 0.6000 | -0.1000 | FAIL |" in text
    assert "Overall gate: FAIL" in text
    assert "| `corpus` | `" + "a" * 64 + "` |" in text
    assert "| `queries` | `" + "b" * 64 + "` |" in text
    assert "| `qrels` | `" + "c" * 64 + "` |" in text
    assert "do not authenticate the dataset publisher" in text
    assert "not generated-answer quality" in text


def test_invalid_evidence_never_generates_output(tmp_path):
    output = tmp_path / "report.md"
    assert scifact_report.main([str(tmp_path), "--output", str(output)]) == 2
    assert not output.exists()


def test_refuses_to_overwrite_existing_report(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "scifact-comparison.json").write_text(json.dumps({
        "max_allowed_drop": 0.0, "passed": True,
        "by_top_k": {
            str(k): {"baseline_hit_lower_bound": 1.0, "candidate_hit_lower_bound": 1.0,
                     "delta": 0.0, "passed": True}
            for k in (1, 3, 5, 10)
        },
    }), encoding="utf-8")
    monkeypatch.setattr(scifact_report, "verify", lambda path, dataset: {
        "source_url": "https://example.org", "evalforge_revision": "test",
        "download_date": "2026-11-21",
    })
    output = tmp_path / "report.md"
    output.write_text("do not overwrite", encoding="utf-8")
    assert scifact_report.main([str(evidence), "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "do not overwrite"
