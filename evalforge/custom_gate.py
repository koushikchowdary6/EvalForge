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
         min_hit_rate: float = 0.5, tokenizer: str = "ascii") -> dict:
    """Return evidence and pass/fail; reject invalid or unreachable labels."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(min_hit_rate, (int, float)) or isinstance(min_hit_rate, bool) or not math.isfinite(min_hit_rate) or not 0 <= min_hit_rate <= 1:
        raise ValueError("min_hit_rate must be finite and between 0 and 1")
    preflight = audit(docs, questions, tokenizer)
    warnings = {name: preflight[name] for name in (
        "unindexable_document_ids",
        "answerable_question_ids_without_corpus_token_overlap",
        "question_ids_with_unindexable_relevance_labels",
        "answerable_question_ids_without_relevant_doc_token_overlap",
    ) if preflight[name]}
    result = evaluate(docs, questions, [k], tokenizer_mode=tokenizer)
    metric = result["by_top_k"][str(k)]["lexical_evidence_hit_rate"]
    passed = not warnings and metric is not None and metric >= min_hit_rate
    return {
        "schema": "evalforge.custom_retrieval_gate.v1",
        "scope": "offline_lexical_retrieval_only_not_llm_quality_or_safety",
        "passed": passed,
        "k": k,
        "minimum_lexical_evidence_hit_rate": min_hit_rate,
        "observed_lexical_evidence_hit_rate": metric,
        "preflight_warnings": warnings,
        "dataset_audit": preflight,
        "benchmark": {key: value for key, value in result.items() if key != "cases"},
    }


def github_annotations(report: dict) -> list[str]:
    """Emit workflow-command-safe, actionable CI diagnostics for failed gates."""
    if report["passed"]:
        return []
    def safe(value: object) -> str:
        return (str(value).replace("%", "%25").replace("\\r", "%0D")
                .replace("\\n", "%0A").replace(":", "%3A"))
    messages = []
    for name, ids in sorted(report["preflight_warnings"].items()):
        messages.append(f"::error title=EvalForge dataset preflight::{safe(name)}: {safe(', '.join(map(str, ids)))}")
    observed = report["observed_lexical_evidence_hit_rate"]
    minimum = report["minimum_lexical_evidence_hit_rate"]
    if observed is None or observed < minimum:
        messages.append(f"::error title=EvalForge lexical retrieval gate::hit rate {safe(observed)} below minimum {safe(minimum)}")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, default=0.5)
    parser.add_argument("--tokenizer", choices=("ascii", "unicode"), default="ascii")
    parser.add_argument("--output", type=Path, help="New JSON report; refuses overwrite")
    parser.add_argument("--github-annotations", action="store_true", help="Emit actionable GitHub Actions errors to stderr on failed gates")\n    args = parser.parse_args(argv)
    try:
        result = gate(load_benchmark_json(args.corpus),
                      load_benchmark_json(args.questions), k=args.k,
                      min_hit_rate=args.min_hit_rate, tokenizer=args.tokenizer)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            print(rendered, end="")
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        print(f"EvalForge custom gate rejected: {exc}", file=sys.stderr)
        return 2
    if args.github_annotations:\n        for annotation in github_annotations(result):\n            print(annotation, file=sys.stderr)\n    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
