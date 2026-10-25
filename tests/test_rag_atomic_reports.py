"""RAG evaluation artifacts survive failed writes without partial JSON."""

import json

import pytest

import evalforge.rag_runner as rag_runner


def _runner(tmp_path):
    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.output_dir = tmp_path
    runner.results = [{"case_id": "original", "top_k": 1}]
    runner.report = {"total_evaluations": 1}
    return runner


def test_atomic_save_replaces_both_complete_artifacts(tmp_path):
    runner = _runner(tmp_path)
    runner.save()
    runner.results = [{"case_id": "new", "top_k": 3}]
    runner.report = {"total_evaluations": 2}
    runner.save()

    assert json.loads((tmp_path / "rag_results.json").read_text()) == runner.results
    assert json.loads((tmp_path / "rag_report.json").read_text()) == runner.report
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_result_serialization_preserves_previous_file(tmp_path):
    runner = _runner(tmp_path)
    runner.save()
    original = (tmp_path / "rag_results.json").read_bytes()
    runner.results = [{"case_id": "invalid", "value": object()}]

    with pytest.raises(TypeError):
        runner.save()

    assert (tmp_path / "rag_results.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_report_serialization_preserves_previous_report(tmp_path):
    runner = _runner(tmp_path)
    runner.save()
    original = (tmp_path / "rag_report.json").read_bytes()
    runner.report = {"bad_value": object()}

    with pytest.raises(TypeError):
        runner.save()

    assert (tmp_path / "rag_report.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_atomic_replace_cleans_temp_and_preserves_previous_json(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.save()
    original = (tmp_path / "rag_results.json").read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(rag_runner.os, "replace", fail_replace)
    runner.results = [{"case_id": "new", "top_k": 5}]
    with pytest.raises(OSError, match="simulated disk failure"):
        runner.save()

    assert (tmp_path / "rag_results.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))
