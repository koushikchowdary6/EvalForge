"""Regression tests for exact, bounded arithmetic in mock agent tools."""

import pytest

from evalforge.agent_tools import execute_tool


@pytest.mark.parametrize(("expression", "expected"), [
    ("9007199254740993 + 1", "9007199254740994"),
    ("9007199254740993 * 3", "27021597764222979"),
    ("0.1 + 0.2", "0.3"),
    ("1.25 * 0.4", "0.5"),
    ("2.5 - 0.5", "2"),
    ("10 / 4", "2.5"),
    ("7 / 0", "[calculate] Division by zero"),
])
def test_mock_calculator_does_not_round_reference_results(expression, expected):
    assert execute_tool("calculate", {"expression": expression}) == expected


def test_mock_calculator_rejects_excessively_large_input():
    expression = "9" * 513 + " + 1"
    assert "Unsupported expression" in execute_tool("calculate", {"expression": expression})


def test_mock_calculator_never_evaluates_python():
    expression = "__import__('os').system('echo unsafe')"
    assert "Unsupported expression" in execute_tool("calculate", {"expression": expression})
