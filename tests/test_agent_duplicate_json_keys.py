"""Duplicate JSON labels must never silently redefine agent benchmark ground truth."""

import json

import pytest

from evalforge.agent_runner import AgentEvalRunner


def _task():
    return {
        "id": "case-1",
        "prompt": "Look up user U-1.",
        "expected_tools": ["lookup_user"],
        "expected_args": {"lookup_user": {"user_id": "U-1"}},
        "mock_responses": {"lookup_user": "User U-1 found"},
        "forbidden_tools": [],
    }


@pytest.mark.parametrize("original, duplicated", [
    ('"id": "case-1"', '"id": "case-1", "id": "case-2"'),
    ('"user_id": "U-1"', '"user_id": "U-1", "user_id": "U-2"'),
    ('"lookup_user": "User U-1 found"',
     '"lookup_user": "User U-1 found", "lookup_user": "User U-2 found"'),
])
def test_duplicate_keys_at_any_depth_fail_closed_and_clear_stale_results(
    tmp_path, original, duplicated
):
    dataset = tmp_path / "tasks.json"
    valid_json = json.dumps([_task()])
    assert original in valid_json
    dataset.write_text(valid_json, encoding="utf-8")
    runner = AgentEvalRunner(dataset, output_dir=tmp_path)
    assert runner.load()[0]["id"] == "case-1"
    runner.results = [{"stale": True}]
    runner.report = {"stale": True}

    ambiguous_json = valid_json.replace(original, duplicated, 1)
    assert json.loads(ambiguous_json)  # A default decoder would silently accept it.
    dataset.write_text(ambiguous_json, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        runner.load()
    assert runner.tasks == []
    assert runner.results == []
    assert runner.report is None


def test_unique_nested_keys_still_load(tmp_path):
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([_task()]), encoding="utf-8")
    runner = AgentEvalRunner(dataset, output_dir=tmp_path)
    assert runner.load()[0]["expected_args"]["lookup_user"]["user_id"] == "U-1"
