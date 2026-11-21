"""Render a verified SciFact lexical retrieval run as a shareable Markdown table.

This reports observed scores only; it never generates or estimates benchmarks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evalforge.scifact_evidence_verify import verify


def render(directory: Path, dataset: Path | None = None) -> str:
    manifest = verify(directory, dataset)
    comparison = json.loads((directory / "scifact-comparison.json").read_text(encoding="utf-8"))
    rows = []
    for k in (1, 3, 5, 10):
        item = comparison["by_top_k"][str(k)]
        rows.append(
            f"| {k} | {item['baseline_hit_lower_bound']:.4f} | "
            f"{item['candidate_hit_lower_bound']:.4f} | {item['delta']:+.4f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    source = str(manifest["source_url"]).replace("|", "%7C").replace("\n", " ")
    revision = str(manifest["evalforge_revision"]).replace("|", "%7C").replace("\n", " ")
    return "\n".join([
        "# SciFact BM25 lexical retrieval comparison",
        "",
        "**Scope:** Positive-qrels-only lexical evidence hit-rate lower bound; "
        "not generated-answer quality, hallucination resistance, or agent safety.",
        "",
        f"- Dataset declared by operator: BEIR SciFact, test split",
        f"- Source declared by operator: {source}",
        f"- Acquisition date declared by operator: {manifest['download_date']}",
        f"- EvalForge revision declared by operator: `{revision}`",
        f"- Original dataset files re-hashed: {'yes' if dataset is not None else 'no'}",
        "",
        "| k | ASCII BM25 | Unicode BM25 | Unicode minus ASCII | Gate |",
        "| ---: | ---: | ---: | ---: | :---: |",
        *rows,
        "",
        f"Illustrative allowed absolute drop: {comparison['max_allowed_drop']:.4f}. "
        f"Overall gate: {'PASS' if comparison['passed'] else 'FAIL'}.",
        "",
        "The evidence directory passed local SHA-256 and report-consistency verification. "
        "This does not authenticate the dataset source or independently establish benchmark quality.",
        "",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Completed SciFact evidence directory")
    parser.add_argument("--dataset", type=Path, help="Re-hash original dataset files")
    parser.add_argument("--output", type=Path, help="New Markdown file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = render(args.directory, args.dataset)
        if args.output:
            with args.output.open("x", encoding="utf-8") as target:
                target.write(report)
        else:
            print(report, end="")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot render SciFact evidence: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
