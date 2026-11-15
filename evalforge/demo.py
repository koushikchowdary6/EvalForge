"""Installed, API-free EvalForge quickstart for developers.

Run with: evalforge-demo
Requires the package installed. No provider keys.
Reports are written to a temporary directory, never to tracked files.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def run_report(module: str, destination: Path, *options: str) -> dict:
    command = [
        sys.executable, "-m", module, "--k", "1", "3", "5",
        "--summary-only", "--output", str(destination), *options,
    ]
    # Bound execution so the public quickstart cannot hang indefinitely.
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=90
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{module} timed out after 90 seconds") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"{module} failed (exit {completed.returncode}): {detail[-2000:]}"
        )
    if not destination.is_file():
        raise RuntimeError(f"{module} did not produce its JSON report")
    try:
        report = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{module} produced an unreadable JSON report") from exc
    if not isinstance(report, dict):
        raise RuntimeError(f"{module} produced a non-object JSON report")
    if "cases" in report:
        raise RuntimeError("Compact report unexpectedly includes individual cases")
    cutoffs = report.get("by_top_k")
    if not isinstance(cutoffs, dict) or set(cutoffs) != {"1", "3", "5"}:
        raise RuntimeError(f"{module} returned invalid benchmark cutoffs")
    for cutoff, row in cutoffs.items():
        if not isinstance(row, dict):
            raise RuntimeError(f"{module} returned a non-object metric row at k={cutoff}")
        metric = ("paired_mean_recall_delta_unicode_minus_ascii"
                  if module.endswith("compare_lexical_baselines")
                  else "lexical_evidence_hit_rate")
        if metric not in row:
            raise RuntimeError(f"{module} missing metric {metric} at k={cutoff}")
    count = report.get("question_count")
    if type(count) is not int or count <= 0:
        raise RuntimeError(f"{module} returned invalid question_count")
    for key in ("corpus_sha256", "questions_sha256"):
        digest = report.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"Invalid dataset fingerprint: {key}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run two API-free retrieval benchmarks")
    parser.add_argument(
        "--json", action="store_true",
        help="Emit one machine-readable JSON summary instead of human-readable output",
    )
    parser.add_argument(
        "--min-hit-rate", type=float, default=None, metavar="RATE",
        help="Fail if BM25 lexical evidence hit rate at k=3 is below RATE (0..1)",
    )
    parser.add_argument("--output", type=Path, default=None, metavar="PATH",
                        help="Write --json report to a new file without overwriting")
    args = parser.parse_args(argv)
    if args.output is not None and not args.json:
        parser.error("--output requires --json")
    if args.min_hit_rate is not None and (
        not math.isfinite(args.min_hit_rate) or not 0 <= args.min_hit_rate <= 1
    ):
        parser.error("--min-hit-rate must be a finite number between 0 and 1")
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
        # Validate every cutoff before emitting any success-looking metrics.
        for cutoff in ("1", "3", "5"):
            bm25_row = bm25["by_top_k"][cutoff]
            paired_row = paired["by_top_k"][cutoff]
            hit = bm25_row["lexical_evidence_hit_rate"]
            delta = paired_row["paired_mean_recall_delta_unicode_minus_ascii"]
            if hit is not None and (type(hit) not in (int, float) or not math.isfinite(hit) or not 0 <= hit <= 1):
                raise RuntimeError(f"Invalid lexical evidence hit rate at k={cutoff}")
            if delta is not None and (type(delta) not in (int, float) or not math.isfinite(delta) or not -1 <= delta <= 1):
                raise RuntimeError(f"Invalid paired recall delta at k={cutoff}")
            hit_text = "undefined" if hit is None else f"{hit:.3f}"
            delta_text = "undefined" if delta is None else f"{delta:+.3f}"
        if args.min_hit_rate is not None:
            observed = bm25["by_top_k"]["3"]["lexical_evidence_hit_rate"]
            if observed is None or observed < args.min_hit_rate:
                raise RuntimeError(
                    f"BM25 lexical evidence hit rate at k=3 "
                    f"({observed}) is below required {args.min_hit_rate}"
                )
        if args.json:
            # Only validated values reach stdout. This output is suitable for
            # CI quality gates and can be parsed without scraping human text.
            payload = {
                "schema_version": 1,
                "benchmark_type": "offline_lexical_retrieval_only",
                "bm25": {
                    "question_count": bm25["question_count"],
                    "corpus_sha256": bm25["corpus_sha256"],
                    "questions_sha256": bm25["questions_sha256"],
                    "lexical_evidence_hit_rate_by_k": {
                        k: bm25["by_top_k"][k]["lexical_evidence_hit_rate"]
                        for k in ("1", "3", "5")
                    },
                },
                "paired_tokenizer": {
                    "question_count": paired["question_count"],
                    "corpus_sha256": paired["corpus_sha256"],
                    "questions_sha256": paired["questions_sha256"],
                    "recall_delta_unicode_minus_ascii_by_k": {
                        k: paired["by_top_k"][k]["paired_mean_recall_delta_unicode_minus_ascii"]
                        for k in ("1", "3", "5")
                    },
                },
                "limitations": "Lexical retrieval diagnostics; not LLM correctness or hallucination resistance",
            }
            serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
            if args.output is None:
                print(serialized)
            else:
                # Exclusive creation prevents overwriting existing benchmark evidence.
                with args.output.open("x", encoding="utf-8") as output:
                    output.write(serialized + "\n")
        else:
            for name, report in (("BM25", bm25), ("Paired ASCII/Unicode", paired)):
                print(
                    f"{name}: {report['question_count']} questions; "
                    f"cutoffs {', '.join(report['by_top_k'])}; "
                    f"corpus SHA-256 {report['corpus_sha256'][:12]}…; "
                    f"questions SHA-256 {report['questions_sha256'][:12]}…"
                )
            for cutoff in ("1", "3", "5"):
                bm25_row = bm25["by_top_k"][cutoff]
                paired_row = paired["by_top_k"][cutoff]
                hit = bm25_row["lexical_evidence_hit_rate"]
                delta = paired_row["paired_mean_recall_delta_unicode_minus_ascii"]
                hit_text = "undefined" if hit is None else f"{hit:.3f}"
                delta_text = "undefined" if delta is None else f"{delta:+.3f}"
                print(
                    f"k={cutoff}: BM25 lexical evidence hit rate={hit_text}; "
                    f"Unicode-minus-ASCII paired recall delta={delta_text}"
                )
            print("EvalForge offline quickstart: PASS")
            print(
                "These are deterministic lexical retrieval diagnostics, "
                "not LLM correctness or hallucination benchmarks."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
