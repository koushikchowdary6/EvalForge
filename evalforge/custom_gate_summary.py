"""Render a custom retrieval gate JSON artifact as a human-readable CI summary.

Only reports offline lexical retrieval evidence; never LLM answer quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from .artifact_validation import _read_report, _unique_keys


def _number(value: object, *, fraction: bool = False) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric metric")
    if not math.isfinite(value) or (fraction and not 0 <= value <= 1):
        raise ValueError("Invalid metric value")
    return f"{value:.3f}" if fraction else str(value)


def render(report: dict) -> str:
    """Reject incompatible reports and produce safe, bounded Markdown."""
    if not isinstance(report, dict) or report.get("schema") != "evalforge.custom_retrieval_gate.v1":
        raise ValueError("Not an EvalForge custom retrieval gate v1 artifact")
    if report.get("scope") != "offline_lexical_retrieval_only_not_llm_quality_or_safety":
        raise ValueError("Unexpected evaluation scope")
    if type(report.get("passed")) is not bool:
        raise ValueError("Missing boolean gate result")
    reasons = report.get("failure_reasons")
    if not isinstance(reasons, list) or any(not isinstance(x, str) for x in reasons):
        raise ValueError("Invalid failure reasons")
    if report["passed"] == bool(reasons):
        raise ValueError("Inconsistent gate outcome and failure reasons")
    k = report.get("k")
    if type(k) is not int or k < 1:
        raise ValueError("Invalid retrieval cutoff")
    metric = _number(report.get("observed_lexical_evidence_hit_rate"), fraction=True)
    threshold = _number(report.get("minimum_lexical_evidence_hit_rate"), fraction=True)
    def count(key: str) -> str:
        value = report.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid count: {key}")
        return str(value)
    # Avoid raw user-controlled strings becoming Markdown or GitHub workflow commands.
    known_reasons = {
        "dataset_preflight_warnings": "Dataset preflight warnings",
        "insufficient_answerable_questions": "Too few answerable questions",
        "insufficient_unanswerable_questions": "Too few unanswerable questions",
        "insufficient_distinct_relevant_documents": "Too few distinct relevant documents",
        "undefined_lexical_evidence_hit_rate": "Undefined lexical hit rate",
        "lexical_evidence_hit_rate_below_minimum": "Lexical hit rate below threshold",
        "ambiguous_relevance_cutoff_ties": "Ambiguous relevance cutoff ties",
    }
    if any(reason not in known_reasons for reason in reasons):
        raise ValueError("Unknown failure reason")
    lines = [
        "# EvalForge custom retrieval gate", "",
        f"**Result:** {'PASS' if report['passed'] else 'FAIL'}",
        "**Scope:** offline BM25 lexical retrieval only; not generated-answer accuracy or LLM safety.", "",
        "| Check | Observed | Required |",
        "| --- | ---: | ---: |",
        f"| Lexical evidence hit rate at k={k} | {metric} | {threshold} |",
        f"| Answerable questions | {_answerable(report)} | {count('minimum_answerable_questions')} |",
        f"| Unanswerable questions | {count('observed_unanswerable_questions')} | {count('minimum_unanswerable_questions')} |",
        f"| Distinct relevant documents | {count('observed_distinct_relevant_documents')} | {count('minimum_distinct_relevant_documents')} |",
        f"| Mixed-relevance cutoff ties | {count('ambiguous_relevance_cutoff_tie_count')} | {'0 (strict)' if report.get('fail_on_ambiguous_ties') is True else 'Not gated'} |",
        "",
    ]
    if reasons:
        lines.extend(["## Gate failures", ""])
        lines.extend(f"- {known_reasons[reason]}" for reason in reasons)
        lines.append("")
    audit = report.get("dataset_audit")
    benchmark = report.get("benchmark")
    if not isinstance(audit, dict) or not isinstance(benchmark, dict):
        raise ValueError("Missing dataset provenance")
    for label, value in (("Corpus", audit.get("corpus_sha256")),
                         ("Questions", audit.get("questions_sha256"))):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid {label} SHA-256")
        lines.append(f"- {label} SHA-256: `{value}`")
    lines.extend(["", "These scores are dataset-specific lexical diagnostics, not production RAG or AI-safety benchmarks.", ""])
    return "\n".join(lines)


def _answerable(report: dict) -> str:
    audit = report.get("dataset_audit")
    if not isinstance(audit, dict):
        raise ValueError("Missing dataset audit")
    value = audit.get("answerable_questions")
    if type(value) is not int or value < 0:
        raise ValueError("Invalid answerable question count")
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Existing evalforge-custom-gate JSON report")
    parser.add_argument("--output", type=Path, help="New Markdown file; refuses overwrite")
    args = parser.parse_args(argv)
    try:
        data = json.loads(_read_report(args.report), object_pairs_hook=_unique_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {value}")))
        markdown = render(data)
        if args.output is None:
            print(markdown, end="")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(markdown)
    except (ValueError, TypeError, OSError, UnicodeError) as exc:
        print(f"EvalForge custom summary rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
