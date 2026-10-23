"""Regression tests: agent graders must not mis-credit the wrong action target."""

from evalforge.agent_graders import argument_accuracy, grade_trajectory


def test_identity_fields_preserve_case_whitespace_and_type():
    expected = {"send_email": {"recipient": "Team+Ops@example.org", "subject": "Quarterly Report"}}
    assert argument_accuracy(
        [{"tool": "send_email", "args": {
            "recipient": "team+ops@example.org", "subject": "QuarterlyReport",
        }}],
        expected,
    ) == 0.0
    assert argument_accuracy(
        [{"tool": "send_email", "args": {
            "recipient": "Team+Ops@example.org", "subject": "Quarterly Report",
        }}],
        expected,
    ) == 1.0
    assert argument_accuracy(
        [{"tool": "lookup_user", "args": {"user_id": 4471}}],
        {"lookup_user": {"user_id": "4471"}},
    ) == 0.0


def test_file_paths_must_not_be_case_folded_or_collapsed():
    expected = {"delete_file": {"path": "/var/data/My File.db"}}
    for wrong_path in ("/var/data/my file.db", "/var/data/MyFile.db"):
        assert argument_accuracy(
            [{"tool": "delete_file", "args": {"path": wrong_path}}],
            expected,
        ) == 0.0


def test_expression_whitespace_remains_ignorable():
    expected = {"calculate": {"expression": "47 * 23"}}
    assert argument_accuracy(
        [{"tool": "calculate", "args": {"expression": "47*23"}}],
        expected,
    ) == 1.0


def test_wrong_recipient_cannot_be_reported_as_agent_success():
    task = {
        "id": "identity-boundary",
        "category": "single_tool",
        "expected_tools": ["send_email"],
        "expected_args": {"send_email": {"recipient": "Team+Ops@example.org"}},
        "forbidden_tools": [],
        "answer_must_contain": ["sent"],
    }
    trajectory = {
        "tool_calls": [{"tool": "send_email", "args": {
            "recipient": "team+ops@example.org",
        }}],
        "final_answer": "Sent.",
    }
    result = grade_trajectory(task, trajectory)
    assert result["tool_selection_score"] == 1.0
    assert result["argument_accuracy"] == 0.0
    assert result["success"] is False
    assert result["failure_category"] == "bad_arguments"
