"""Agent benchmark artifacts must never be replaced by partial or invalid JSON."""

import json

import pytest

import evalforge.artifacts as artifacts
from evalforge.agent_runner import AgentEvalRunner


def _runner(tmp_path):
    runner = AgentEvalRunner.__new__(AgentEvalRunner)
    runner.output_dir = tmp_path
    runner.results = [{"case_id": "original", "grade": {"success": True}}]
    runner.report = {"total_tasks": 1}
    return runner


def test_agent_save_publishes_complete_results_and_report(tmp_path):
    runner = _runner(tmp_path)
    runner.save()
    runner.results = [{"case_id": "new", "grade": {"success": False}}]
    runner.report = {"total_tasks": 2}
    runner.save()

    assert json.loads((tmp_path / "agent_results.json").read_text()) == runner.results
    assert json.loads((tmp_path / "agent_report.json").read_text()) == runner.report
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("bad_value", [object(), float("nan")])
def test_bad_results_do_not_truncate_previous_artifact(tmp_path, bad_value):
    runner = _runner(tmp_path)
    runner.save()
    before = (tmp_path / "agent_results.json").read_bytes()
    runner.results = [{"case_id": "bad", "value": bad_value}]

    with pytest.raises((TypeError, ValueError)):
        runner.save()

    assert (tmp_path / "agent_results.json").read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_bad_report_preserves_previous_report(tmp_path):
    runner = _runner(tmp_path)
    runner.save()
    before = (tmp_path / "agent_report.json").read_bytes()
    runner.report = {"bad": object()}

    with pytest.raises(TypeError):
        runner.save()

    assert (tmp_path / "agent_report.json").read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_atomic_replace_preserves_previous_agent_results(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.save()
    before = (tmp_path / "agent_results.json").read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    runner.results = [{"case_id": "new"}]
    with pytest.raises(OSError, match="simulated disk failure"):
        runner.save()

    assert (tmp_path / "agent_results.json").read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
