"""Regression: signed and decimal answers must match complete numeric tokens."""

import pytest

from evalforge.agent_graders import answer_contains_all, grade_trajectory


@pytest.mark.parametrize("response", [
    "The result is -120.",
    "The result is -12.5.",
    "The result is -12e3.",
    "The result is -12,000.",
    "The result is 12.",
    "The result is +12.",
    "The result is x-12.",
    "The result is −123.",
])
def test_negative_reference_does_not_match_other_values(response):
    assert not answer_contains_all(response, ["-12"])


@pytest.mark.parametrize("response", [
    "The result is -12.",
    "The result is −12.",
    "The temperature is -12C.",
    "The temperature is −12°C.",
])
def test_negative_reference_accepts_complete_negative_token(response):
    assert answer_contains_all(response, ["-12"])


@pytest.mark.parametrize("response", [
    "The result is 112.5.",
    "The result is 12.50.",
    "The result is 12.56.",
    "The result is -12.5.",
    "The result is −12.5.",
    "The result is 12.5e2.",
    "The result is 12.5,0.",
    "The result is v12.5.",
])
def test_decimal_reference_does_not_match_other_values(response):
    assert not answer_contains_all(response, ["12.5"])


@pytest.mark.parametrize("response", [
    "The result is 12.5.",
    "The result is +12.5.",
    "The total is 1,081.25.",
])
def test_decimal_reference_accepts_complete_value_and_grouping(response):
    expected = "1081.25" if "1,081.25" in response else "12.5"
    assert answer_contains_all(response, [expected])


def test_wrong_signed_decimal_cannot_pass_agent_trajectory():
    task = {
        "id": "signed-decimal", "category": "no_tool",
        "expected_tools": [], "expected_args": {},
        "answer_must_contain": ["-12.5"],
    }
    result = grade_trajectory(task, {
        "tool_calls": [], "final_answer": "The answer is -12.50.",
        "turns": 1,
    })
    assert result["answer_ok"] is False
    assert result["success"] is False
    assert result["failure_category"] == "wrong_answer"
