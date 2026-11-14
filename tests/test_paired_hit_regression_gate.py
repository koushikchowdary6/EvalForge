"""CLI regression gates must use worst-case positive-score tie bounds."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evalforge import compare_lexical_baselines as cli


def report(hit_upper, recall_upper=0.0):
    return {
        "by_top_k": {
            "1": {
                "paired_hit_rate_delta_upper_bound_unicode_minus_ascii": hit_upper,
                "paired_mean_recall_delta_upper_bound_unicode_minus_ascii": recall_upper,
            }
        },
        "cases": [{"case_id": "fixture", "top_k": 1}],
    }


class CertainHitRegressionGateTests(unittest.TestCase):
    def run_cli(self, hit_upper, *flags, recall_upper=0.0):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "load_benchmark_json", return_value=[]), \
             patch.object(cli, "compare_tokenizers", return_value=report(hit_upper, recall_upper)), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                cli.main(["--k", "1", *flags])
            except SystemExit as exc:
                return exc.code, stdout.getvalue(), stderr.getvalue()
        return 0, stdout.getvalue(), stderr.getvalue()

    def test_certain_hit_regression_fails_when_enabled(self):
        code, output, error = self.run_cli(-0.25, "--fail-on-certain-hit-regression")
        self.assertEqual(code, 1)
        self.assertIn("k=1", error)
        self.assertEqual(json.loads(output)["by_top_k"]["1"][
            "paired_hit_rate_delta_upper_bound_unicode_minus_ascii"], -0.25)

    def test_uncertain_or_undefined_hit_delta_does_not_fail(self):
        for upper in (0.0, 0.5, None):
            with self.subTest(upper=upper):
                code, _, _ = self.run_cli(upper, "--fail-on-certain-hit-regression")
                self.assertEqual(code, 0)

    def test_default_does_not_gate_and_recall_gate_remains_independent(self):
        self.assertEqual(self.run_cli(-0.5)[0], 0)
        self.assertEqual(self.run_cli(-0.5, "--fail-on-certain-regression")[0], 0)
        self.assertEqual(self.run_cli(0.5, "--fail-on-certain-regression",
                                      recall_upper=-0.25)[0], 1)

    def test_compact_output_still_reports_aggregate(self):
        code, output, _ = self.run_cli(0.0, "--summary-only",
                                       "--fail-on-certain-hit-regression")
        self.assertEqual(code, 0)
        self.assertNotIn("cases", json.loads(output))
        self.assertIn("by_top_k", json.loads(output))

    def test_failed_gate_writes_report_for_ci_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            code, _, _ = self.run_cli(-0.5, "--fail-on-certain-hit-regression",
                                      "--output", str(destination))
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(destination.read_text())["by_top_k"]["1"][
                "paired_hit_rate_delta_upper_bound_unicode_minus_ascii"], -0.5)


if __name__ == "__main__":
    unittest.main()
