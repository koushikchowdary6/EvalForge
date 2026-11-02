"""Reject malformed Unicode benchmark evidence before paid agent calls.

JSON permits escaped lone surrogates, but provider requests and UTF-8 artifacts
require Unicode scalar values. Valid multilingual text remains supported.
"""

import json

import pytest

import evalforge.agent_runner as runner_module
from evalforge.agent_runner import AgentEvalRunner, validate_agent_tasks


def _task():
    return {
        "id": "unicode-evidence",
        "category": "no_tool",
        "prompt": "Summarize this weather report.",
        "expected_tools": ["get_weather"],
        "answer_must_contain": ["sunny"],
        "mock_responses": {"get_weather": "Sunny skies."},
    }


@pytest.mark.parametrize("field", [
    "prompt", "category", "answer_must_contain", "mock_responses",
])
@pytest.mark.parametrize("invalid", ["\\ud800", "\\udfff", "text\\ud800"])
def test_lone_surrogates_in_agent_benchmark_evidence_are_rejected(field, invalid):
    task = _task()
    if field == "answer_must_contain":
        task[field] = [invalid]
    elif field == "mock_responses":
        task[field]["get_weather"] = invalid
    else:
        task[field] = invalid
    with pytest.raises(ValueError, match=field):
        validate_agent_tasks([task])


def test_escaped_surrogate_in_json_is_rejected_before_agent_initialization(
    tmp_path, monkeypatch,
):
    task = _task()
    task["mock_responses"]["get_weather"] = "Sunny\\ud800"
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([task]), encoding="utf-8")
    runner = AgentEvalRunner(dataset, output_dir=tmp_path)

    def never_initialize(*args, **kwargs):
        raise AssertionError("invalid benchmark reached model initialization")

    monkeypatch.setattr(runner_module, "ToolCallingAgent", never_initialize)
    with pytest.raises(ValueError, match="mock_responses"):
        runner.load()
    assert runner.tasks == []
    assert runner.tasks_sha256 is None


def test_run_revalidates_mutated_unicode_evidence_before_paid_agent(
    tmp_path, monkeypatch,
):
    runner = AgentEvalRunner(tmp_path / "unused.json", output_dir=tmp_path)
    task = _task()
    runner.tasks = [task]
    runner.results = [{"stale": True}]
    runner.report = {"stale": True}
    task["prompt"] += "\\udfff"

    def never_initialize(*args, **kwargs):
        raise AssertionError("invalid benchmark reached model initialization")

    monkeypatch.setattr(runner_module, "ToolCallingAgent", never_initialize)
    with pytest.raises(ValueError, match="prompt"):
        runner.run()
    assert runner.results == []
    assert runner.report is None
    assert runner.tasks_sha256 is None


def test_multilingual_and_astral_unicode_benchmark_evidence_remains_valid():
    task = _task()
    task["prompt"] = "Résumé 東京 🌍"
    task["category"] = "évaluation"
    task["answer_must_contain"] = ["café", "東京", "🌍"]
    task["mock_responses"]["get_weather"] = "Café\\u0301 🌤️"
    validate_agent_tasks([task])
