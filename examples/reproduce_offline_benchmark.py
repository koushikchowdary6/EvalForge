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
        print("PASS: reproducible offline benchmark, validated evidence, no regression")
        print("Scope: bundled lexical fixtures only; no model calls or safety claims")


if __name__ == "__main__":
    main()
