"""Invisible-only RAG completions are missing answers, not successful outputs."""

import pytest

from evalforge.rag_graders import did_abstain, grade_abstention


@pytest.mark.parametrize("answer", [
    "\u0301",          # Standalone combining accent
    "\u0301\u0308",  # Multiple combining marks
    "\u200b",          # Zero-width space
    "\u200d\ufe0f",  # Format control and variation selector
    " \n\u0301\u200b\t",  # Whitespace plus invisible characters
])
@pytest.mark.parametrize("answerable", [True, False])
def test_invisible_only_generation_is_empty_not_a_scored_answer(answer, answerable):
    assert did_abstain(answer) is False
    assert grade_abstention(answer, answerable) == {
        "outcome": "empty_answer", "abstained": False, "passed": False,
    }


@pytest.mark.parametrize("answer", [
    "Caf\u00e9",           # Precomposed accent
    "Cafe\u0301",         # Combining accent attached to a visible base
    "A\u200bB",           # Invisible separator alongside visible content
])
def test_visible_unicode_answers_remain_scoreable(answer):
    result = grade_abstention(answer, answerable=True)
    assert result["outcome"] == "correct_answer"
    assert result["passed"] is True


def test_explicit_abstention_token_still_passes_unanswerable_case():
    assert grade_abstention(" INSUFFICIENT_CONTEXT ", answerable=False) == {
        "outcome": "correct_abstain", "abstained": True, "passed": True,
    }
