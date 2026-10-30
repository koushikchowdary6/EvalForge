"""Resource bounds for untrusted agent benchmark prompts and mock tool evidence."""

import json

import pytest

import evalforge.agent_runner as runner_module
from evalforge.agent_runner import (
    AgentEvalRunner, MAX_AGENT_MOCK_RESPONSE_CHARS,
    MAX_AGENT_PROMPT_CHARS, validate_agent_tasks,
)


def _task():
    return {
        "id": "bounded-agent-case",
        "prompt": "Summarize the retrieved article.",
        "expected_tools": ["search_knowledge_base"],
        "mock_responses": {"search_knowledge_base": "A brief document."},
    }


@pytest.mark.parametrize("field,limit", [
    ("prompt", MAX_AGENT_PROMPT_CHARS),
    ("mock_responses", MAX_AGENT_MOCK_RESPONSE_CHARS),
])
def test_exact_limit_accepted_and_one_character_over_rejected(field, limit):
    task = _task()
    if field == "prompt":
        task["prompt"] = "A" * limit
    else:
        task["mock_responses"]["search_knowledge_base"] = "A" * limit
    validate_agent_tasks([task])

    if field == "prompt":
        task["prompt"] += "B"
    else:
        task["mock_responses"]["search_knowledge_base"] += "B"
    with pytest.raises(ValueError, match=field):
        validate_agent_tasks([task])


def test_invalid_mock_evidence_is_rejected_before_paid_agent(tmp_path, monkeypatch):
    runner = AgentEvalRunner(tmp_path / "unused.json", output_dir=tmp_path)
    task = _task()
    task["mock_responses"]["search_knowledge_base"] = "X" * (MAX_AGENT_MOCK_RESPONSE_CHARS + 1)
    runner.tasks = [task]
    runner.results = [{"stale": True}]
    runner.report = {"stale": True}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid benchmark reached provider initialization")

    monkeypatch.setattr(runner_module, "ToolCallingAgent", fail_if_called)
    with pytest.raises(ValueError, match="mock_responses"):
        runner.run()
    assert runner.results == []
    assert runner.report is None


def test_invalid_file_reload_does_not_retain_previous_tasks(tmp_path):
    path = tmp_path / "agent.json"
    path.write_text(json.dumps([_task()]), encoding="utf-8")
    runner = AgentEvalRunner(path, output_dir=tmp_path)
    runner.load()
    assert len(runner.tasks) == 1

    task = _task()
    task["prompt"] = "P" * (MAX_AGENT_PROMPT_CHARS + 1)
    path.write_text(json.dumps([task]), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt exceeds"):
        runner.load()
    assert runner.tasks == []
    assert runner.tasks_sha256 is None
