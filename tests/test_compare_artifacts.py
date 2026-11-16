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

def test_installed_cli_reports_regression_exit_code(tmp_path, capsys):
    """Exercise the public command's JSON output and CI failure contract."""
    import json
    from evalforge.compare_artifacts import main

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(fixture()), encoding="utf-8")
    changed = copy.deepcopy(fixture())
    changed["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.5
    candidate.write_text(json.dumps(changed), encoding="utf-8")

    assert main([str(baseline), str(candidate), "--tolerance", "0.01"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["bm25_regression_cutoffs"] == ["3"]
    assert result["metric_deltas"]["bm25"]["3"] == pytest.approx(-0.25)
    assert main([str(baseline), str(candidate), "--tolerance", "0.25"]) == 0
    assert json.loads(capsys.readouterr().out)["bm25_regression_cutoffs"] == []


def test_cli_rejects_duplicate_json_keys(tmp_path, capsys):
    """A duplicate metric must not silently override evidence."""
    import json
    from evalforge.compare_artifacts import main

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(fixture()), encoding="utf-8")
    candidate.write_text(
        json.dumps(fixture()).replace('"schema_version": 1',
                                      '"schema_version": 1, "schema_version": 1', 1),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main([str(baseline), str(candidate)])
    assert exc.value.code == 2
    assert "duplicate" in capsys.readouterr().err.lower()


def test_cli_rejects_incomparable_dataset(tmp_path, capsys):
    """Cross-dataset comparisons must fail instead of producing false deltas."""
    import json
    from evalforge.compare_artifacts import main

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(fixture()), encoding="utf-8")
    changed = copy.deepcopy(fixture())
    changed["paired_tokenizer"]["questions_sha256"] = "c" * 64
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([str(baseline), str(candidate)])
    assert exc.value.code == 2
    assert "fixtures differ" in capsys.readouterr().err


def test_comparison_exposes_absolute_scores_and_gate_tolerance():
    """A downstream CI report must explain the gate using actual scores."""
    candidate = copy.deepcopy(fixture())
    candidate["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.5
    report = compare(fixture(), candidate, tolerance=0.1)
    assert report["tolerance"] == pytest.approx(0.1)
    assert report["absolute_scores"]["bm25"]["baseline"]["3"] == pytest.approx(0.75)
    assert report["absolute_scores"]["bm25"]["candidate"]["3"] == pytest.approx(0.5)
    assert report["metric_deltas"]["bm25"]["3"] == pytest.approx(-0.25)
    assert report["bm25_regression_cutoffs"] == ["3"]
    assert report["absolute_scores"]["paired_tokenizer"]["baseline"]["3"] == pytest.approx(-0.1)
    assert report["absolute_scores"]["paired_tokenizer"]["candidate"]["3"] == pytest.approx(-0.1)


def test_cli_writes_comparison_artifact_even_when_gate_fails(tmp_path, capsys):
    """CI should retain a machine-readable failed gate for later inspection."""
    import json
    from evalforge.compare_artifacts import main

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(fixture()), encoding="utf-8")
    changed = copy.deepcopy(fixture())
    changed["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.5
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    assert main([str(baseline), str(candidate), "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["bm25_regression_cutoffs"] == ["3"]
    assert capsys.readouterr().out == ""


def test_cli_never_overwrites_existing_comparison_artifact(tmp_path, capsys):
    """Existing CI evidence must not be silently replaced."""
    import json
    from evalforge.compare_artifacts import main

    baseline = tmp_path / "baseline.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(fixture()), encoding="utf-8")
    output.write_text("previous evidence", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([str(baseline), str(baseline), "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_text(encoding="utf-8") == "previous evidence"
    assert "output rejected" in capsys.readouterr().err.lower()


def test_cli_failed_publish_leaves_no_partial_report_or_tempfile(tmp_path, monkeypatch, capsys):
    """An interrupted publish must leave no incomplete report for CI to consume."""
    import json
    import os
    from evalforge.compare_artifacts import main

    baseline = tmp_path / "baseline.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(fixture()), encoding="utf-8")

    def fail_publish(source, destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(os, "link", fail_publish)
    with pytest.raises(SystemExit) as exc:
        main([str(baseline), str(baseline), "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["baseline.json"]
    assert "output rejected" in capsys.readouterr().err.lower()


def test_absolute_floor_fails_without_relative_regression():
    """An unchanged weak model must not pass solely because baseline is weak."""
    weak = fixture()
    weak["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.25
    report = compare(weak, copy.deepcopy(weak), min_hit_rate=0.5)
    assert report["bm25_regression_cutoffs"] == []
    assert report["minimum_hit_rate_failed"] is True
    assert report["min_hit_rate"] == 0.5
    assert compare(weak, weak, min_hit_rate=0.25)["minimum_hit_rate_failed"] is False


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_absolute_floor_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="Minimum hit rate"):
        compare(fixture(), fixture(), min_hit_rate=invalid)


def test_absolute_floor_cli_exit_and_report(tmp_path, capsys):
    """Quality floor is a gate failure (exit 1), not an invalid report (exit 2)."""
    import json
    from evalforge.compare_artifacts import main

    weak = fixture()
    weak["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.25
    path = tmp_path / "weak.json"
    path.write_text(json.dumps(weak), encoding="utf-8")
    assert main([str(path), str(path), "--min-hit-rate", "0.5"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["minimum_hit_rate_failed"] is True
    assert report["bm25_regression_cutoffs"] == []
    assert main([str(path), str(path), "--min-hit-rate", "0.25"]) == 0
    assert json.loads(capsys.readouterr().out)["minimum_hit_rate_failed"] is False


def test_github_annotations_report_absolute_floor_failure(tmp_path, capsys):
    """A failing absolute quality floor must appear in Actions, even without a regression."""
    import json
    from evalforge.compare_artifacts import main

    weak = fixture()
    weak["bm25"]["lexical_evidence_hit_rate_by_k"]["3"] = 0.25
    path = tmp_path / "weak.json"
    path.write_text(json.dumps(weak), encoding="utf-8")
    assert main([str(path), str(path), "--min-hit-rate", "0.5", "--github-annotations"]) == 1
    output = capsys.readouterr()
    assert "BM25 minimum hit rate at k=3" in output.err
    assert "0.250000" in output.err
    assert "0.500000" in output.err
    assert "regression at" not in output.err
    assert json.loads(output.out)["minimum_hit_rate_failed"] is True


def test_github_annotations_omit_absolute_floor_on_pass(tmp_path, capsys):
    """Passing quality floors must not emit misleading error annotations."""
    import json
    from evalforge.compare_artifacts import main

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(fixture()), encoding="utf-8")
    assert main([str(path), str(path), "--min-hit-rate", "0.5", "--github-annotations"]) == 0
    assert "::error" not in capsys.readouterr().err
