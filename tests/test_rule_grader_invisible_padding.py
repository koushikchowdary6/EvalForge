"""Invisible Unicode padding must not inflate deterministic minimum-length grades."""

import pytest

from evalforge.graders import RuleBasedGrader


@pytest.mark.parametrize(
    "padding",
    [
        "\u200b" * 20,  # zero-width space
        "\u200d" * 20,  # zero-width joiner
        "\u0301" * 20,  # combining accent
        "\u202e" * 20,  # bidirectional override
        "\x00" * 20,  # control bytes
    ],
)
def test_invisible_padding_cannot_satisfy_minimum_length(padding):
    result = RuleBasedGrader().grade(
        "Yes or no?", "Yes" + padding,
        rules=[{"type": "min_length", "value": 10}],
    )
    assert result["passed"] is False
    assert result["score"] < 0.6
    assert "too short" in result["details"]


def test_visible_multilingual_response_satisfies_minimum_length():
    result = RuleBasedGrader().grade(
        "Yes or no?", "こんにちは",
        rules=[{"type": "min_length", "value": 5}],
    )
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_normal_spaces_remain_part_of_character_based_length():
    result = RuleBasedGrader().grade(
        "Yes or no?", "Yes indeed",
        rules=[{"type": "min_length", "value": 10}],
    )
    assert result["passed"] is True


def test_max_length_still_enforces_raw_input_budget():
    result = RuleBasedGrader().grade(
        "Yes or no?", "Yes" + "\u200b" * 20,
        rules=[{"type": "max_length", "value": 10}],
    )
    assert result["passed"] is False
    assert "too long" in result["details"]
