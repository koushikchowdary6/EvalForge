"""Regression tests for the conservative paired lexical CI gate."""
import json
from unittest.mock import patch

import pytest

from evalforge.compare_lexical_baselines import main


def _report(upper):
    return {
        "method": "paired_ascii_unicode_bm25_lexical_retrieval_only",
        "by_top_k": {
            "1": {
                "paired_mean_recall_delta_upper_bound_unicode_minus_ascii": upper,
            },
        },
        "cases": [{"case_id": "example", "top_k": 1}],
    }


@pytest.mark.parametrize(
    ("upper", "expected_code"),
    [(-0.25, 1), (0.0, 0), (0.2, 0), (None, 0), (-1e-14, 0)],
)
def test_certain_regression_gate(tmp_path, capsys, upper, expected_code):
    output = tmp_path / "paired.json"
    with (
        patch("evalforge.compare_lexical_baselines.load_benchmark_json", return_value=[]),
        patch("evalforge.compare_lexical_baselines.compare_tokenizers", return_value=_report(upper)),
    ):
        if expected_code:
            with pytest.raises(SystemExit) as error:
                main(["--fail-on-certain-regression", "--summary-only", "--output", str(output)])
            assert error.value.code == expected_code
        else:
            main(["--fail-on-certain-regression", "--summary-only", "--output", str(output)])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "cases" not in data
    assert data["by_top_k"]["1"]["paired_mean_recall_delta_upper_bound_unicode_minus_ascii"] == upper
    if expected_code:
        assert "Certain Unicode lexical recall regression" in capsys.readouterr().err


def test_gate_opt_in_does_not_break_default_stdout(capsys):
    with (
        patch("evalforge.compare_lexical_baselines.load_benchmark_json", return_value=[]),
        patch("evalforge.compare_lexical_baselines.compare_tokenizers", return_value=_report(-0.5)),
    ):
        main(["--summary-only"])
    assert json.loads(capsys.readouterr().out)["by_top_k"]["1"][
        "paired_mean_recall_delta_upper_bound_unicode_minus_ascii"
    ] == -0.5
