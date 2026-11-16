#!/usr/bin/env python3
"""Reproduce the installed EvalForge offline evaluation contract without API keys.

Run after installing the package: python examples/reproduce_offline_benchmark.py
This is a lexical retrieval smoke benchmark, not an LLM safety evaluation.
"""
import json
import subprocess
import tempfile
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True, text=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="evalforge-reproduce-") as directory:
        root = Path(directory)
        baseline = root / "baseline.json"
        candidate = root / "candidate.json"
        comparison = root / "comparison.json"
        run("evalforge-demo", "--json", "--output", str(baseline))
        run("evalforge-demo", "--json", "--output", str(candidate))
        run("evalforge-validate", str(baseline))
        run("evalforge-validate", str(candidate))
        run("evalforge-compare-artifacts", str(baseline), str(candidate),
            "--output", str(comparison))
        first = json.loads(baseline.read_text(encoding="utf-8"))
        second = json.loads(candidate.read_text(encoding="utf-8"))
        result = json.loads(comparison.read_text(encoding="utf-8"))
        assert first == second, "Offline benchmark was not deterministic"
        assert result["bm25_regression_cutoffs"] == [], "Identical inputs regressed"
        for section in ("bm25", "paired_tokenizer"):
            for key in ("corpus_sha256", "questions_sha256"):
                assert len(first[section][key]) == 64, (section, key)
        # Also prove the installed gate rejects a degraded candidate and retains
        # machine-readable failure evidence for a downstream CI artifact upload.
        degraded = root / "degraded.json"
        failure_report = root / "regression.json"
        altered = json.loads(candidate.read_text(encoding="utf-8"))
        hit_rates = altered["bm25"]["lexical_evidence_hit_rate_by_k"]
        assert hit_rates["3"] is not None and hit_rates["3"] > 0, (
            "Bundled fixture no longer has a positive hit rate at k=3"
        )
        hit_rates["3"] = 0.0
        degraded.write_text(json.dumps(altered), encoding="utf-8")
        failure = subprocess.run(
            ["evalforge-compare-artifacts", str(baseline), str(degraded),
             "--output", str(failure_report)],
            capture_output=True, text=True, check=False,
        )
        assert failure.returncode == 1, (
            f"Expected regression exit code 1, got {failure.returncode}: "
            f"{failure.stderr}"
        )
        failed_evidence = json.loads(failure_report.read_text(encoding="utf-8"))
        assert "3" in failed_evidence["bm25_regression_cutoffs"], (
            "Regression report omitted degraded k=3 metric"
        )
        assert failed_evidence["metric_deltas"]["bm25"]["3"] < 0
        print("PASS: installed regression gate rejects degraded k=3 evidence "
              "and preserves a JSON report")
        print("PASS: reproducible offline benchmark, validated evidence, no regression")
        print("Scope: bundled lexical fixtures only; no model calls or safety claims")


if __name__ == "__main__":
    main()
