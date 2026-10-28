"""General benchmark artifacts must be complete, standard JSON on disk."""

import json

import pytest

from evalforge import artifacts
from evalforge.runner import EvalForgeRunner


def _runner(tmp_path):
    runner = EvalForgeRunner.__new__(EvalForgeRunner)
    runner.output_dir = tmp_path
    runner.results = [{"case_id": "original", "model": "gpt"}]
    runner.report = {"total_evaluations": 1}
    return runner


def test_general_save_replaces_complete_results_and_report(tmp_path):
    runner = _runner(tmp_path)
    runner.save_results()
    runner.results = [{"case_id": "new", "model": "claude"}]
    runner.report = {"total_evaluations": 2}
    runner.save_results()

    assert json.loads((tmp_path / "results.json").read_text()) == runner.results
    assert json.loads((tmp_path / "report.json").read_text()) == runner.report
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("invalid", [object(), float("nan")], ids=["object", "nan"])
def test_invalid_results_preserve_previous_complete_artifact(tmp_path, invalid):
    runner = _runner(tmp_path)
    runner.save_results()
    original = (tmp_path / "results.json").read_bytes()
    runner.results = [{"case_id": "bad", "value": invalid}]

    with pytest.raises((TypeError, ValueError)):
        runner.save_results()

    assert (tmp_path / "results.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("invalid", [object(), float("inf")], ids=["object", "infinity"])
def test_invalid_report_preserves_previous_report(tmp_path, invalid):
    runner = _runner(tmp_path)
    runner.save_results()
    original = (tmp_path / "report.json").read_bytes()
    runner.report = {"bad": invalid}

    with pytest.raises((TypeError, ValueError)):
        runner.save_results()

    assert (tmp_path / "report.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_replace_preserves_previous_results(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.save_results()
    original = (tmp_path / "results.json").read_bytes()
    runner.results = [{"case_id": "new"}]

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        runner.save_results()

    assert (tmp_path / "results.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_aborted_run_clears_stale_report_without_partial_json(tmp_path):
    runner = _runner(tmp_path)
    runner.save_results()
    runner.results = []
    runner.report = None
    runner.save_results()

    assert json.loads((tmp_path / "results.json").read_text()) == []
    assert not (tmp_path / "report.json").exists()
    assert not list(tmp_path.glob(".*.tmp"))
