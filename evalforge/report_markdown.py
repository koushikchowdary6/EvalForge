"""Render a validated EvalForge offline benchmark artifact as a Markdown summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evalforge.artifact_validation import _read_report, _unique_keys, validate


def render(report: dict) -> str:
    """Render lexical metrics without interpreting them as LLM safety scores."""
    validate(report, 0.0)
    bm25 = report["bm25"]
    paired = report["paired_tokenizer"]
    lines = [
        "# EvalForge offline retrieval report", "",
        "**Scope:** deterministic lexical retrieval diagnostics only; not LLM answer accuracy or safety.", "",
        f"- BM25 questions: {bm25['question_count']}",
        f"- Paired-tokenizer questions: {paired['question_count']}", "",
        "| k | BM25 lexical evidence hit rate | Unicode minus ASCII recall delta |",
        "| ---: | ---: | ---: |",
    ]
    for k in ("1", "3", "5"):
        hit = bm25["lexical_evidence_hit_rate_by_k"][k]
        delta = paired["recall_delta_unicode_minus_ascii_by_k"][k]
        h = "undefined" if hit is None else f"{hit:.3f}"
        d = "undefined" if delta is None else f"{delta:+.3f}"
        lines.append(f"| {k} | {h} | {d} |")
    lines.extend(["", "## Dataset provenance", "",
                  "| Benchmark | Corpus SHA-256 | Questions SHA-256 |",
                  "| --- | --- | --- |"])
    for label, part in (("BM25", bm25), ("Paired tokenizer", paired)):
        lines.append(f"| {label} | `{part['corpus_sha256']}` | `{part['questions_sha256']}` |")
    lines.extend(["", "Metrics describe small labeled lexical fixtures, not model safety.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Existing evalforge-demo --json artifact")
    parser.add_argument("--output", type=Path, help="New Markdown file; never overwrite")
    args = parser.parse_args(argv)
    try:
        report = json.loads(_read_report(args.report), object_pairs_hook=_unique_keys,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {value}")))
        markdown = render(report)
        if args.output is None:
            print(markdown, end="")
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(markdown)
    except (ValueError, KeyError, TypeError, OSError, UnicodeError) as exc:
        print(f"EvalForge Markdown report rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
