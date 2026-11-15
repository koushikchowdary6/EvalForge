"""Atomic CLI export contract: complete artifact or no destination."""
import json

import pytest

from evalforge import demo


def _report(paired):
    metric = ("paired_mean_recall_delta_unicode_minus_ascii" if paired
              else "lexical_evidence_hit_rate")
    return {
        "method": ("paired_ascii_unicode_bm25_lexical_retrieval_only" if paired
                   else "lexical_bm25_retrieval_only"),
        "question_count": 2,
        "corpus_sha256": "a" * 64,
        "questions_sha256": "b" * 64,
        "by_top_k": {str(k): {metric: 0.5} for k in (1, 3, 5)},
    }


def test_json_export_is_parseable_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        demo, "run_report",
        lambda module, destination, *options: _report(
            module.endswith("compare_lexical_baselines")
        ),
    )
    target = tmp_path / "evaluation.json"
    assert demo.main(["--json", "--output", str(target)]) == 0
    assert json.loads(target.read_text())["schema_version"] == 1
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        demo.main(["--json", "--output", str(target)])
    assert target.read_bytes() == original
    assert list(tmp_path.glob(".evalforge-*.tmp")) == []


def test_json_export_link_failure_leaves_no_partial_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        demo, "run_report",
        lambda module, destination, *options: _report(
            module.endswith("compare_lexical_baselines")
        ),
    )
    target = tmp_path / "evaluation.json"

    def fail_link(source, destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(demo.os, "link", fail_link)
    with pytest.raises(OSError, match="simulated publish failure"):
        demo.main(["--json", "--output", str(target)])
    assert not target.exists()
    assert list(tmp_path.glob(".evalforge-*.tmp")) == []
