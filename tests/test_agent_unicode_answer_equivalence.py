"""Agent answer grading should treat Unicode-equivalent labels consistently."""

import pytest

from evalforge.agent_graders import answer_contains_all, grade_trajectory


@pytest.mark.parametrize(("response", "required"), [
    ("The result was STRASSE.", "Straße"),
    ("The location is Cafe\u0301.", "Café"),
    ("The result was Straße.", "STRASSE"),
    ("The answer is CAFE\u0301 STRASSE.", "Café Straße"),
])
def test_agent_answer_accepts_canonical_and_casefold_equivalence(response, required):
    assert answer_contains_all(response, [required])


@pytest.mark.parametrize(("response", "required"), [
    ("The result was STRASSEN.", "Straße"),
    ("The location is Caféine.", "Café"),
    ("The result was 120.", "12"),
    ("The result was 1,2.", "12"),
])
def test_equivalence_does_not_credit_partial_words_or_numbers(response, required):
    assert not answer_contains_all(response, [required])


def test_unicode_equivalent_answer_can_complete_agent_task():
    task = {
        "id": "unicode-answer",
        "category": "no_tool",
        "expected_tools": [],
        "expected_args": {},
        "answer_must_contain": ["Straße"],
    }
    result = grade_trajectory(task, {
        "tool_calls": [], "final_answer": "The answer is STRASSE.", "turns": 1,
    })
    assert result["answer_ok"] is True
    assert result["success"] is True
    assert result["failure_category"] is None
