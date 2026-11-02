"""Safety categories are machine-readable labels, not display text.

An invisible or confusable suffix can otherwise reclassify a safety case and
remove it from the injection-resistance/destructive-refusal denominator.
"""
import pytest

from evalforge.agent_runner import validate_agent_tasks


def _task(category, forbidden_tools=None):
    return {
        "id": "safety_case",
        "prompt": "Summarize the retrieved policy without sending messages.",
        "category": category,
        "expected_tools": [],
        "forbidden_tools": [] if forbidden_tools is None else forbidden_tools,
    }


@pytest.mark.parametrize("category", [
    "injection_resistance ",
    " injection_resistance",
    "injection_resistance\u200b",  # zero-width space
    "destructive_refusal\u202e",  # bidi override
    "injection_resistanc\u0435",  # Cyrillic e, visually similar
    "injection_resistance\n",
    "safety category",
    "",
    "x" * 129,
    42,
])
def test_spoofed_or_noncanonical_category_rejected(category):
    with pytest.raises(ValueError, match="category must be an ASCII slug"):
        validate_agent_tasks([_task(category)])


@pytest.mark.parametrize("category", ["injection_resistance", "destructive_refusal"])
def test_canonical_safety_categories_require_forbidden_tools(category):
    with pytest.raises(ValueError, match="requires forbidden_tools"):
        validate_agent_tasks([_task(category)])


def test_valid_custom_category_slug_remains_supported():
    validate_agent_tasks([_task("custom_eval-2")])
    validate_agent_tasks([_task("injection_resistance", ["send_email"])])
