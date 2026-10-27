"""Agent benchmark reports must identify the exact validated task inputs."""

import json

import pytest

import evalforge.agent_runner as module
from evalforge.agent_runner import AgentEvalRunner, _agent_dataset_sha256


def task(prompt="Return done.", case_id="case-1"):
    return {
        "id": case_id, "category": "single_tool", "prompt": prompt,
        "expected_tools": [], "forbidden_tools": [],
        "answer_must_contain": ["done"],
    }


class StubAgent:
    def __init__(self, **kwargs):
        pass

    def run(self, prompt, mock_responses=None):
        return {"tool_calls": [], "final_answer": "done", "turns": 1,
                "latency_ms": 1, "tokens": {}, "error": None}


def test_canonical_hash_preserves_case_order_and_labels():
    left = [task(), task(case_id="case-2")]
    reordered_keys = [dict(reversed(list(item.items()))) for item in left]
    assert _agent_dataset_sha256(left) == _agent_dataset_sha256(reordered_keys)
    assert _agent_dataset_sha256(left) != _agent_dataset_sha256(list(reversed(left)))
    assert _agent_dataset_sha256(left) != _agent_dataset_sha256([task("Different"), left[1]])
    assert len(_agent_dataset_sha256({"metadata": "\\ud800"})) == 64


def test_report_uses_validated_tasks_and_refreshes_between_runs(tmp_path, monkeypatch):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([task()]), encoding="utf-8")
    monkeypatch.setattr(module, "ToolCallingAgent", StubAgent)
    runner = AgentEvalRunner(path, output_dir=tmp_path)
    runner.load()
    initial = runner.tasks_sha256
    assert initial == _agent_dataset_sha256([task()])
    runner.run()
    report = runner.generate_report()
    assert report["benchmark_provenance"]["agent_tasks_sha256"] == initial
    runner.save()
    saved = json.loads((tmp_path / "agent_report.json").read_text(encoding="utf-8"))
    assert saved["benchmark_provenance"]["agent_tasks_sha256"] == initial

    runner.tasks[0]["prompt"] = "Return done again."
    runner.run()
    assert runner.tasks_sha256 != initial
    assert runner.generate_report()["benchmark_provenance"]["agent_tasks_sha256"] == runner.tasks_sha256


def test_failed_reload_and_invalid_mutation_cannot_retain_stale_digest(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([task()]), encoding="utf-8")
    runner = AgentEvalRunner(path, output_dir=tmp_path)
    runner.load()
    assert runner.tasks_sha256

    path.write_text(json.dumps([dict(task(), prompt=" ")]), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        runner.load()
    assert runner.tasks_sha256 is None
    assert runner.results == []
    assert runner.report is None

    runner.tasks = [task()]
    runner.tasks_sha256 = "stale"
    runner.tasks[0]["expected_tools"] = "not-a-list"
    with pytest.raises(ValueError, match="expected_tools"):
        runner.run()
    assert runner.tasks_sha256 is None
