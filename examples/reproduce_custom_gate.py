#!/usr/bin/env python3
"""Reproduce a consumer-style custom retrieval gate and a controlled failing gate.

Run from a source checkout after installing EvalForge:
    python examples/reproduce_custom_gate.py

The bundled customer-support examples are synthetic lexical retrieval fixtures,
not evidence of production RAG quality, generation accuracy, or LLM safety.
"""
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "customer_support_corpus.json"
QUESTIONS = ROOT / "examples" / "customer_support_questions.json"


def invoke(output: Path, *, min_answerable: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "evalforge-custom-gate",
            "--corpus", str(CORPUS),
            "--questions", str(QUESTIONS),
            "--k", "1",
            "--min-hit-rate", "0.5",
            "--min-answerable-questions", str(min_answerable),
            "--min-unanswerable-questions", "1",
            "--min-distinct-relevant-docs", "4",
            "--include-cases",
            "--output", str(output),
        ],
        text=True, capture_output=True, check=False,
    )


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise RuntimeError(detail)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="evalforge-custom-repro-") as directory:
        root = Path(directory)
        first, second, failed = (root / name for name in
                                 ("pass-one.json", "pass-two.json", "negative-control.json"))
        for path in (first, second):
            run = invoke(path, min_answerable=4)
            require(run.returncode == 0, f"Expected passing gate: {run.stderr}")
            require(path.exists(), "Passing gate did not save evidence")
        require(first.read_bytes() == second.read_bytes(),
                "Identical inputs did not produce byte-identical JSON evidence")
        passed = json.loads(first.read_text(encoding="utf-8"))
        require(passed["schema"] == "evalforge.custom_retrieval_gate.v1",
                "Unexpected gate report schema")
        require(passed["passed"] is True and passed["failure_reasons"] == [],
                "Passing gate has inconsistent diagnostics")
        require(passed["observed_unanswerable_questions"] == 1,
                "Fixture negative-question coverage changed")
        require(passed["observed_distinct_relevant_documents"] == 4,
                "Fixture relevance diversity changed")
        require(len(passed["cases"]) == 5,
                "Per-question evidence missing")

        # Impossible coverage requirement: must fail without hiding evidence.
        negative = invoke(failed, min_answerable=5)
        require(negative.returncode == 1,
                f"Expected quality gate exit 1, got {negative.returncode}: {negative.stderr}")
        require(failed.exists(), "Failed quality gate did not save evidence")
        rejected = json.loads(failed.read_text(encoding="utf-8"))
        require(rejected["passed"] is False and
                "insufficient_answerable_questions" in rejected["failure_reasons"],
                "Negative control did not identify insufficient coverage")
        require(rejected["observed_lexical_evidence_hit_rate"] ==
                passed["observed_lexical_evidence_hit_rate"],
                "Coverage-only negative control unexpectedly changed retrieval score")

        # The installed summary CLI must preserve the failed gate's diagnostics.
        summary = root / "negative-control.md"
        summary_cmd = ["evalforge-custom-summary", str(failed), "--output", str(summary)]
        rendered = subprocess.run(summary_cmd, text=True, capture_output=True, check=False)
        require(rendered.returncode == 0 and summary.exists(),
                f"Failed gate could not be rendered: {rendered.stderr}")
        markdown = summary.read_text(encoding="utf-8")
        require("**Result:** FAIL" in markdown and
                "Too few answerable questions" in markdown,
                "Markdown summary omitted the failing coverage diagnostic")
        require(passed["dataset_audit"]["corpus_sha256"] in markdown,
                "Markdown summary omitted dataset provenance")
        summary_before = summary.read_bytes()
        repeated_summary = subprocess.run(summary_cmd, text=True, capture_output=True,
                                          check=False)
        require(repeated_summary.returncode == 2 and
                summary.read_bytes() == summary_before,
                "Summary CLI overwrote existing diagnostic evidence")

        # Evidence must be immutable by default, including on a failing gate.
        before = failed.read_bytes()
        repeat = invoke(failed, min_answerable=5)
        require(repeat.returncode == 2 and failed.read_bytes() == before,
                "Gate overwrote an existing evaluation artifact")
    print("PASS: deterministic installed custom gate, labeled negative coverage,")
    print("      explicit failing control, readable failure summary, no-overwrite contract")
    print("Scope: synthetic lexical retrieval only; no LLM quality/safety claim")


if __name__ == "__main__":
    main()
