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


def _require_same_tokenizer(baseline: dict, candidate: dict) -> None:
    """Reject configuration drift when comparing reproducibility baselines."""
    try:
        left = baseline["evaluation"]["parameters"]["tokenizer"]
        right = candidate["evaluation"]["parameters"]["tokenizer"]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing tokenizer configuration") from exc
    if not isinstance(left, str) or not left or not isinstance(right, str) or not right:
        raise ValueError("missing tokenizer configuration")
    if left != right:
        raise ValueError("tokenizer configurations differ; disable --require-same-tokenizer only for deliberate tokenizer experiments")


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
        for field in ("labeled_queries", "excluded_queries_without_positive_qrels"):
            count = source.get(field)
            if type(count) is not int or count < 0:
                raise ValueError(f"invalid source query count: {field}")
        for field in ("corpus_size", "question_count"):
            count = evaluation.get(field)
            if type(count) is not int or count < 0:
                raise ValueError(f"invalid evaluation size: {field}")
        source_queries = source.get("source_queries")
        if type(source_queries) is not int or source_queries < 0:
            raise ValueError("invalid source query count: source_queries")
        if source["labeled_queries"] != evaluation["question_count"]:
            raise ValueError("labeled query count does not match evaluation")
        if (source["labeled_queries"] + source["excluded_queries_without_positive_qrels"]
                != source_queries):
            raise ValueError("included and excluded query counts do not reconcile")
        parameters = evaluation.get("parameters")
        if not isinstance(parameters, dict) or not isinstance(parameters.get("tokenizer"), str):
            raise ValueError("missing tokenizer configuration")
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


def compare_all(baseline: dict, candidate: dict, *, max_drop: float = 0.0) -> dict:
    """Gate every reported cutoff, rejecting missing or mismatched cutoffs."""
    try:
        left = baseline["evaluation"]["by_top_k"]
        right = candidate["evaluation"]["by_top_k"]
    except (TypeError, KeyError) as exc:
        raise ValueError("missing BEIR cutoff summaries") from exc
    if not isinstance(left, dict) or not isinstance(right, dict) or not left or set(left) != set(right):
        raise ValueError("reports must contain identical nonempty cutoff sets")
    cutoffs = []
    for key in left:
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key or int(key) < 1:
            raise ValueError("invalid cutoff key")
        cutoffs.append(int(key))
    comparisons = [compare(baseline, candidate, k=k, max_drop=max_drop)
                   for k in sorted(cutoffs)]
    return {
        "schema_version": 1,
        "comparison_scope": "same_source_beir_lexical_evidence_all_cutoffs",
        "max_allowed_drop": max_drop,
        "passed": all(item["passed"] for item in comparisons),
        "by_top_k": {str(item["k"]): item for item in comparisons},
    }



def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_report(path: Path) -> dict:
    """Read strict JSON so duplicated provenance or scores cannot be hidden."""
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--k", type=int, help="Compare one retrieval cutoff")
    group.add_argument("--all-k", action="store_true", help="Gate every identical cutoff in both reports")
    parser.add_argument("--max-drop", type=float, default=0.0)
    parser.add_argument("--require-same-tokenizer", action="store_true", help="Reject tokenizer drift between baseline and candidate")
    parser.add_argument("--output", type=Path, help="Write JSON evidence to a new file (never overwrite)")
    args = parser.parse_args(argv)
    try:
        baseline = _load_report(args.baseline)
        candidate = _load_report(args.candidate)
        if args.require_same_tokenizer:
            _require_same_tokenizer(baseline, candidate)
        result = (compare_all(baseline, candidate, max_drop=args.max_drop)
                  if args.all_k else compare(baseline, candidate, k=args.k, max_drop=args.max_drop))
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            # Exclusive creation prevents overwriting prior CI evidence.
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        return 0 if result["passed"] else 1
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError, RecursionError) as exc:
        print(f"BEIR comparison rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
