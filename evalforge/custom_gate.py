"""Gate a custom labeled retrieval benchmark with a deterministic lexical baseline.

This is not an LLM correctness or safety acceptance test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from .dataset_audit import audit
from .offline_rag_demo import evaluate, load_benchmark_json


def gate(docs: list, questions: list, *, k: int = 3,
         min_hit_rate: float = 0.5, tokenizer: str = "ascii",
         include_cases: bool = False, fail_on_ambiguous_ties: bool = False,
         min_answerable_questions: int = 0, min_distinct_relevant_docs: int = 0,
         min_unanswerable_questions: int = 0) -> dict:
    """Return evidence and pass/fail; reject invalid or unreachable labels."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(min_hit_rate, (int, float)) or isinstance(min_hit_rate, bool) or not math.isfinite(min_hit_rate) or not 0 <= min_hit_rate <= 1:
        raise ValueError("min_hit_rate must be finite and between 0 and 1")
    if type(include_cases) is not bool:
        raise ValueError("include_cases must be a boolean")
    if type(fail_on_ambiguous_ties) is not bool:
        raise ValueError("fail_on_ambiguous_ties must be a boolean")
    if type(min_answerable_questions) is not int or min_answerable_questions < 0:
        raise ValueError("min_answerable_questions must be a nonnegative integer")
    if type(min_distinct_relevant_docs) is not int or min_distinct_relevant_docs < 0:
        raise ValueError("min_distinct_relevant_docs must be a nonnegative integer")
    if type(min_unanswerable_questions) is not int or min_unanswerable_questions < 0:
        raise ValueError("min_unanswerable_questions must be a nonnegative integer")
    preflight = audit(docs, questions, tokenizer)
    # Count unique relevance labels, not repeated questions about one document.
    distinct_relevant_docs = len({doc_id for question in questions if question["answerable"]
                                  for doc_id in question["relevant_doc_ids"]})
    warnings = {name: preflight[name] for name in (
        "unindexable_document_ids",
        "answerable_question_ids_without_corpus_token_overlap",
        "question_ids_with_unindexable_relevance_labels",
        "answerable_question_ids_without_relevant_doc_token_overlap",
    ) if preflight[name]}
    result = evaluate(docs, questions, [k], tokenizer_mode=tokenizer)
    metric = result["by_top_k"][str(k)]["lexical_evidence_hit_rate"]
    ambiguous_ties = result["by_top_k"][str(k)]["answerable_mixed_relevance_cutoff_tie_count"]
    insufficient_questions = preflight["answerable_questions"] < min_answerable_questions
    observed_unanswerable = sum(not q["answerable"] for q in questions)
    insufficient_unanswerable = observed_unanswerable < min_unanswerable_questions
    insufficient_relevant_docs = distinct_relevant_docs < min_distinct_relevant_docs
    passed = (not warnings and not insufficient_questions and not insufficient_unanswerable and not insufficient_relevant_docs and metric is not None and metric >= min_hit_rate
              and (not fail_on_ambiguous_ties or ambiguous_ties == 0))
    # Stable, machine-readable reasons let CI consumers distinguish dataset
    # integrity problems from metric regressions without parsing annotations.
    failure_reasons = []
    if warnings:
        failure_reasons.append("dataset_preflight_warnings")
    if insufficient_questions:
        failure_reasons.append("insufficient_answerable_questions")
    if insufficient_unanswerable:
        failure_reasons.append("insufficient_unanswerable_questions")
    if insufficient_relevant_docs:
        failure_reasons.append("insufficient_distinct_relevant_documents")
    if metric is None:
        failure_reasons.append("undefined_lexical_evidence_hit_rate")
    elif metric < min_hit_rate:
        failure_reasons.append("lexical_evidence_hit_rate_below_minimum")
    if fail_on_ambiguous_ties and ambiguous_ties:
        failure_reasons.append("ambiguous_relevance_cutoff_ties")
    report = {
        "schema": "evalforge.custom_retrieval_gate.v1",
        "scope": "offline_lexical_retrieval_only_not_llm_quality_or_safety",
        "passed": passed,
        "failure_reasons": failure_reasons,
        "k": k,
        "minimum_answerable_questions": min_answerable_questions,
        "minimum_unanswerable_questions": min_unanswerable_questions,
        "observed_unanswerable_questions": observed_unanswerable,
        "minimum_distinct_relevant_documents": min_distinct_relevant_docs,
        "observed_distinct_relevant_documents": distinct_relevant_docs,
        "minimum_lexical_evidence_hit_rate": min_hit_rate,
        "observed_lexical_evidence_hit_rate": metric,
        "preflight_warnings": warnings,
        "ambiguous_relevance_cutoff_tie_count": ambiguous_ties,
        "fail_on_ambiguous_ties": fail_on_ambiguous_ties,
        "dataset_audit": preflight,
        "benchmark": {key: value for key, value in result.items() if key != "cases"},
    }
    if include_cases:
        report["cases"] = result["cases"]
    return report


def github_annotations(report: dict) -> list[str]:
    """Emit workflow-command-safe, actionable CI diagnostics for failed gates."""
    if report["passed"]:
        return []
    def safe(value: object) -> str:
        return (str(value).replace("%", "%25").replace("\r", "%0D")
                .replace("\n", "%0A").replace(":", "%3A"))
    messages = []
    for name, ids in sorted(report["preflight_warnings"].items()):
        messages.append(f"::error title=EvalForge dataset preflight::{safe(name)}: {safe(', '.join(map(str, ids)))}")
    observed = report["observed_lexical_evidence_hit_rate"]
    minimum = report["minimum_lexical_evidence_hit_rate"]
    if observed is None or observed < minimum:
        messages.append(f"::error title=EvalForge lexical retrieval gate::hit rate {safe(observed)} below minimum {safe(minimum)}")
    if "insufficient_answerable_questions" in report.get("failure_reasons", []):
        messages.append(f"::error title=EvalForge dataset size::answerable questions {report['dataset_audit']['answerable_questions']} below required {report['minimum_answerable_questions']}")
    if "insufficient_unanswerable_questions" in report.get("failure_reasons", []):
        messages.append(f"::error title=EvalForge negative coverage::unanswerable questions {report['observed_unanswerable_questions']} below required {report['minimum_unanswerable_questions']}")
    if "insufficient_distinct_relevant_documents" in report.get("failure_reasons", []):
        messages.append(f"::error title=EvalForge relevance diversity::distinct relevant documents {report['observed_distinct_relevant_documents']} below required {report['minimum_distinct_relevant_documents']}")
    if report.get("fail_on_ambiguous_ties") and report.get("ambiguous_relevance_cutoff_tie_count", 0):
        messages.append("::error title=EvalForge ambiguous BM25 cutoff::mixed-relevance positive-score ties make the selected top-k dependent on tie breaking")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, default=0.5)
    parser.add_argument("--min-answerable-questions", type=int, default=0,
                        help="Fail if fewer than this many labeled answerable questions exist")
    parser.add_argument("--min-unanswerable-questions", type=int, default=0,
                        help="Require labeled unanswerable questions to exercise negative-query coverage")
    parser.add_argument("--min-distinct-relevant-docs", type=int, default=0,
                        help="Require this many unique labeled relevant documents across answerable questions")
    parser.add_argument("--tokenizer", choices=("ascii", "unicode"), default="ascii")
    parser.add_argument("--output", type=Path, help="New JSON report; refuses overwrite")
    parser.add_argument("--include-cases", action="store_true",
                        help="Include per-question retrieval diagnostics in the JSON evidence")
    parser.add_argument("--github-annotations", action="store_true", help="Emit actionable GitHub Actions errors to stderr on failed gates")
    parser.add_argument("--fail-on-ambiguous-ties", action="store_true",
                        help="Fail when a positive-score tie at k contains both relevant and irrelevant documents")
    args = parser.parse_args(argv)
    try:
        result = gate(load_benchmark_json(args.corpus),
                      load_benchmark_json(args.questions), k=args.k,
                      min_hit_rate=args.min_hit_rate, tokenizer=args.tokenizer,
                      include_cases=args.include_cases,
                      fail_on_ambiguous_ties=args.fail_on_ambiguous_ties,
                      min_answerable_questions=args.min_answerable_questions,
                      min_distinct_relevant_docs=args.min_distinct_relevant_docs,
                      min_unanswerable_questions=args.min_unanswerable_questions)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            print(rendered, end="")
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        print(f"EvalForge custom gate rejected: {exc}", file=sys.stderr)
        return 2
    if args.github_annotations:
        for annotation in github_annotations(result):
            print(annotation, file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
