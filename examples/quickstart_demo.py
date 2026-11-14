"""Reproducible, API-free EvalForge quickstart for recruiters and developers.

Run from a checkout with: python examples/quickstart_demo.py
Requires the package installed (python -m pip install .). No provider keys.
Reports are written to a temporary directory, never to tracked files.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_report(module: str, destination: Path, *options: str) -> dict:
    command = [
        sys.executable, "-m", module, "--k", "1", "3", "5",
        "--summary-only", "--output", str(destination), *options,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(destination.read_text(encoding="utf-8"))
    if "cases" in report:
        raise RuntimeError("Compact report unexpectedly includes individual cases")
    if set(report["by_top_k"]) != {"1", "3", "5"}:
        raise RuntimeError("Benchmark cutoffs differ from requested values")
    for key in ("corpus_sha256", "questions_sha256"):
        digest = report[key]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"Invalid dataset fingerprint: {key}")
    return report


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="evalforge-demo-") as directory:
        root = Path(directory)
        bm25 = run_report("evalforge.offline_rag_demo", root / "bm25.json")
        paired = run_report(
            "evalforge.compare_lexical_baselines", root / "paired.json"
        )
        if bm25["method"] != "lexical_bm25_retrieval_only":
            raise RuntimeError("Unexpected offline benchmark method")
        if paired["method"] != "paired_ascii_unicode_bm25_lexical_retrieval_only":
            raise RuntimeError("Unexpected paired benchmark method")
        print("EvalForge offline quickstart: PASS")
        for name, report in (("BM25", bm25), ("Paired ASCII/Unicode", paired)):
            print(
                f"{name}: {report['question_count']} questions; "
                f"cutoffs {', '.join(report['by_top_k'])}; "
                f"corpus SHA-256 {report['corpus_sha256'][:12]}…; "
                f"questions SHA-256 {report['questions_sha256'][:12]}…"
            )
        # Display actual measurements, not only successful execution and hashes.
        # Values come from the bundled fixtures; no accuracy claims about LLMs.
        for cutoff in ("1", "3", "5"):
            bm25_row = bm25["by_top_k"][cutoff]
            paired_row = paired["by_top_k"][cutoff]
            hit = bm25_row["lexical_evidence_hit_rate"]
            delta = paired_row["paired_mean_recall_delta_unicode_minus_ascii"]
            if hit is not None and not 0 <= hit <= 1:
                raise RuntimeError(f"Invalid lexical evidence hit rate at k={cutoff}")
            if delta is not None and not -1 <= delta <= 1:
                raise RuntimeError(f"Invalid paired recall delta at k={cutoff}")
            hit_text = "undefined" if hit is None else f"{hit:.3f}"
            delta_text = "undefined" if delta is None else f"{delta:+.3f}"
            print(
                f"k={cutoff}: BM25 lexical evidence hit rate={hit_text}; "
                f"Unicode-minus-ASCII paired recall delta={delta_text}"
            )
        print(
            "These are deterministic lexical retrieval diagnostics, "
            "not LLM correctness or hallucination benchmarks."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
