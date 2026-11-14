"""CLI regression tests for the opt-in unanswerable lexical-match gate.

A lexical match on an unanswerable item is a diagnostic, not hallucination.
"""
import json
from unittest.mock import patch

import pytest

from evalforge.compare_lexical_baselines import main


def _report(delta):
    return {
        "method": "paired_ascii_unicode_bm25_lexical_retrieval_only",
        "by_top_k": {
            "1": {
                "paired_mean_recall_delta_upper_bound_unicode_minus_ascii": None,
                "paired_unanswerable_match_rate_delta_unicode_minus_ascii": delta,
            },
        },
        "cases": [{"case_id": "unanswerable", "top_k": 1}],
    }


@pytest.mark.parametrize(
    ("delta", "should_fail"),
    [(0.5, True), (1e-13, False), (0.0, False), (-0.5, False), (None, False)],
)
def test_unanswerable_gate_is_opt_in_and_preserves_diagnostics(
    tmp_path, capsys, delta, should_fail
):
    report_path = tmp_path / "report.json"
    with (
        patch("evalforge.compare_lexical_baselines.load_benchmark_json", return_value=[]),
        patch("evalforge.compare_lexical_baselines.compare_tokenizers", return_value=_report(delta)),
    ):
        args = [
            "--fail-on-unanswerable-match-regression",
            "--summary-only",
            "--output", str(report_path),
        ]
        if should_fail:
            with pytest.raises(SystemExit) as error:
                main(args)
            assert error.value.code == 1
            assert "unanswerable lexical match-rate regression" in capsys.readouterr().err
        else:
            main(args)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "cases" not in report
    assert report["by_top_k"]["1"][
        "paired_unanswerable_match_rate_delta_unicode_minus_ascii"
    ] == delta


def test_unanswerable_gate_does_not_change_default_cli(capsys):
    with (
        patch("evalforge.compare_lexical_baselines.load_benchmark_json", return_value=[]),
        patch("evalforge.compare_lexical_baselines.compare_tokenizers", return_value=_report(0.5)),
    ):
        main(["--summary-only"])
    report = json.loads(capsys.readouterr().out)
    assert report["by_top_k"]["1"][
        "paired_unanswerable_match_rate_delta_unicode_minus_ascii"
    ] == 0.5


def test_unanswerable_gate_checks_every_cutoff(tmp_path):
    report = _report(0.0)
    report["by_top_k"]["3"] = {
        "paired_mean_recall_delta_upper_bound_unicode_minus_ascii": None,
        "paired_unanswerable_match_rate_delta_unicode_minus_ascii": 0.25,
    }
    output = tmp_path / "all-cutoffs.json"
    with (
        patch("evalforge.compare_lexical_baselines.load_benchmark_json", return_value=[]),
        patch("evalforge.compare_lexical_baselines.compare_tokenizers", return_value=report),
    ):
        with pytest.raises(SystemExit) as error:
            main(["--fail-on-unanswerable-match-regression", "--output", str(output)])
    assert error.value.code == 1
    assert set(json.loads(output.read_text())["by_top_k"]) == {"1", "3"}
