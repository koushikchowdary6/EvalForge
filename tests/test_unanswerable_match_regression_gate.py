"""Paired unanswerable lexical-match regression gate tests (not hallucination claims)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evalforge import compare_lexical_baselines as cli


def fixture(delta):
    return {
        "by_top_k": {
            "1": {
                "paired_mean_recall_delta_upper_bound_unicode_minus_ascii": None,
                "paired_unanswerable_match_rate_delta_unicode_minus_ascii": delta,
            }
        },
        "cases": [{"case_id": "unanswerable", "top_k": 1}],
    }


class UnanswerableMatchGateTests(unittest.TestCase):
    def invoke(self, delta, *flags):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "load_benchmark_json", return_value=[]), \
             patch.object(cli, "compare_tokenizers", return_value=fixture(delta)), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                cli.main(["--k", "1", *flags])
            except SystemExit as exc:
                return exc.code, stdout.getvalue(), stderr.getvalue()
        return 0, stdout.getvalue(), stderr.getvalue()

    def test_positive_delta_fails_only_when_opted_in(self):
        self.assertEqual(self.invoke(0.5)[0], 0)
        code, output, error = self.invoke(
            0.5, "--fail-on-unanswerable-match-regression")
        self.assertEqual(code, 1)
        self.assertIn("k=1", error)
        self.assertIn("unanswerable lexical", error)
        self.assertEqual(json.loads(output)["by_top_k"]["1"][
            "paired_unanswerable_match_rate_delta_unicode_minus_ascii"], 0.5)

    def test_zero_negative_and_undefined_delta_pass(self):
        for delta in (0, -0.5, None):
            with self.subTest(delta=delta):
                self.assertEqual(self.invoke(
                    delta, "--fail-on-unanswerable-match-regression")[0], 0)

    def test_compact_failure_preserves_export(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            code, output, _ = self.invoke(
                0.25, "--summary-only", "--fail-on-unanswerable-match-regression",
                "--output", str(destination))
            self.assertEqual(code, 1)
            self.assertEqual(output, "")
            report = json.loads(destination.read_text(encoding="utf-8"))
            self.assertNotIn("cases", report)
            self.assertEqual(report["by_top_k"]["1"][
                "paired_unanswerable_match_rate_delta_unicode_minus_ascii"], 0.25)


if __name__ == "__main__":
    unittest.main()
