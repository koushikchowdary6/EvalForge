"""Compare two BEIR reports on identical source files, with an optional CI floor.

This compares conservative BM25 lexical evidence, not LLM answer quality.
Different tokenizer configurations are allowed; dataset drift is not.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def compare(baseline: dict, candidate: dict, *, k: int, max_drop: float = 0.0) -> dict:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not math.isfinite(max_drop) or not 0 <= max_drop <= 1:
        raise ValueError("max-drop must be a finite number in [0, 1]")
    for report in (baseline, candidate):
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise ValueError("expected version 1 BEIR benchmark reports")
        if report.get("benchmark_scope") != "offline_lexical_bm25_positive_qrels_only":
            raise ValueError("incompatible benchmark scope")
        source = report.get("source")
        evaluation = report.get("evaluation")
        if not isinstance(source, dict) or not isinstance(evaluation, dict):
            raise ValueError("missing source provenance or evaluation")
        fingerprints = source.get("source_sha256")
        if not isinstance(fingerprints, dict) or set(fingerprints) != {"corpus", "queries", "qrels"}:
            raise ValueError("missing source SHA-256 fingerprints")
        if any(not isinstance(v, str) or len(v) != 64 or
               any(ch not in "0123456789abcdef" for ch in v)
               for v in fingerprints.values()):
            raise ValueError("invalid source SHA-256 fingerprint")
        if evaluation.get("method") != "lexical_bm25_retrieval_only":
            raise ValueError("incompatible retrieval method")
        by_k = evaluation.get("by_top_k")
        if not isinstance(by_k, dict) or str(k) not in by_k:
            raise ValueError(f"both reports must contain cutoff k={k}")
    if baseline["source"]["source_sha256"] != candidate["source"]["source_sha256"]:
        raise ValueError("source files differ; comparison would conflate dataset drift")
    for field in ("labeled_queries", "excluded_queries_without_positive_qrels"):
        if baseline["source"].get(field) != candidate["source"].get(field):
            raise ValueError("query inclusion counts differ")
    for field in ("corpus_size", "question_count"):
        if baseline["evaluation"].get(field) != candidate["evaluation"].get(field):
            raise ValueError("evaluation sizes differ")
    scores = []
    for report in (baseline, candidate):
        value = report["evaluation"]["by_top_k"][str(k)]
        if not isinstance(value, dict):
            raise ValueError("invalid cutoff summary")
        score = value.get("lexical_evidence_hit_lower_bound")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("missing or invalid conservative hit-rate score")
        scores.append(score)
    baseline_score, candidate_score = scores
    drop = baseline_score - candidate_score
    return {
        "schema_version": 1,
        "comparison_scope": "same_source_beir_lexical_evidence_only",
        "k": k,
        "source_sha256": baseline["source"]["source_sha256"],
        "baseline_tokenizer": baseline["evaluation"].get("parameters", {}).get("tokenizer"),
        "candidate_tokenizer": candidate["evaluation"].get("parameters", {}).get("tokenizer"),
        "baseline_hit_lower_bound": baseline_score,
        "candidate_hit_lower_bound": candidate_score,
        "delta": candidate_score - baseline_score,
        "max_allowed_drop": max_drop,
        "passed": drop <= max_drop + 1e-12,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--max-drop", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        result = compare(baseline, candidate, k=args.k, max_drop=args.max_drop)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["passed"] else 1
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError, RecursionError) as exc:
        print(f"BEIR comparison rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
