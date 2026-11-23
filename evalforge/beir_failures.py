"""Inspect missed positive-qrel queries in an offline BEIR BM25 baseline.

This diagnostic shows *which* labeled queries fail and which documents BM25
ranked instead. It does not infer answer quality, semantic relevance, or safety.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .beir_import import convert
from .offline_rag_demo import (
    _build_bm25_index, _rank_with_bm25_index, _select_tokenizer,
)


def analyze(corpus: Path, queries: Path, qrels: Path, *, k: int = 5,
            tokenizer: str = "ascii", limit: int = 20):
    """Return reproducible, positive-score-only retrieval failure examples."""
    if type(k) is not int or k < 1 or type(limit) is not int or limit < 1:
        raise ValueError("k and limit must be positive integers")
    tokenize = _select_tokenizer(tokenizer)
    docs, cases, source = convert(corpus, queries, qrels)
    index = _build_bm25_index(docs, tokenize)
    failures = []
    successes = 0
    reason_counts = {name: 0 for name in (
        "no_query_tokens", "all_query_terms_oov",
        "no_positive_relevant_overlap", "relevant_docs_outside_top_k",
    )}
    for case in cases:
        ranked = _rank_with_bm25_index(case["question"], docs, index, tokenize)
        # Zero BM25 score is not evidence of lexical retrieval.
        retrieved = [(doc_id, score) for doc_id, score in ranked[:k] if score > 0]
        relevant = set(case["relevant_doc_ids"])
        hits = sum(doc_id in relevant for doc_id, _ in retrieved)
        if hits:
            successes += 1
            continue
        terms = set(tokenize(case["question"]))
        vocabulary = set(index[3])
        if not terms:
            reason = "no_query_tokens"
        elif not terms.intersection(vocabulary):
            reason = "all_query_terms_oov"
        elif not any(doc_id in relevant and score > 0 for doc_id, score in ranked):
            reason = "no_positive_relevant_overlap"
        else:
            reason = "relevant_docs_outside_top_k"
        reason_counts[reason] += 1
        first_relevant = next(
            ((rank, score) for rank, (doc_id, score) in enumerate(ranked, 1)
             if doc_id in relevant and score > 0), None
        )
        failures.append({
            "query_id": case["id"],
            "query": case["question"],
            "relevant_doc_ids": sorted(relevant),
            "top_k_positive_results": [
                {"doc_id": doc_id, "score": score, "relevant": doc_id in relevant}
                for doc_id, score in retrieved
            ],
            "reason": reason,
            "first_positive_relevant_rank": first_relevant[0] if first_relevant else None,
            "first_positive_relevant_score": first_relevant[1] if first_relevant else None,
        })
    failures.sort(key=lambda row: row["query_id"])
    return {
        "schema_version": 1,
        "scope": "offline_bm25_positive_qrels_only_diagnostic",
        "source": source,
        "tokenizer": tokenizer,
        "k": k,
        "labeled_queries": len(cases),
        "queries_with_positive_relevant_hit": successes,
        "queries_without_positive_relevant_hit": len(failures),
        "failure_reason_counts": reason_counts,
        "shown_failures": failures[:limit],
        "truncated": len(failures) > limit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tokenizer", choices=("ascii", "unicode"), default="ascii")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-hit-rate", type=float, help="Fail with exit code 1 when the positive-qrel hit@k rate is below this 0..1 floor; still emit the JSON report")
    parser.add_argument("--output", type=Path, help="New JSON report; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.k < 1 or args.limit < 1:
            raise ValueError("k and limit must be positive")
        if args.min_hit_rate is not None and not (0.0 <= args.min_hit_rate <= 1.0):
            raise ValueError("min-hit-rate must be finite and between 0 and 1")
        if args.output and args.output.resolve() in {
            args.corpus.resolve(), args.queries.resolve(), args.qrels.resolve()
        }:
            raise ValueError("output cannot overwrite input")
        if args.output and (args.output.exists() or args.output.is_symlink()):
            raise ValueError("output already exists or is a symlink")
        report = analyze(args.corpus, args.queries, args.qrels,
                         k=args.k, tokenizer=args.tokenizer, limit=args.limit)
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            created = False
            try:
                with args.output.open("x", encoding="utf-8") as stream:
                    created = True
                    stream.write(payload)
            except OSError:
                if created:
                    args.output.unlink(missing_ok=True)
                raise
        else:
            sys.stdout.write(payload)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(f"BEIR failure analysis rejected: {exc}", file=sys.stderr)
        return 2
    if args.min_hit_rate is not None:
        if report["labeled_queries"] == 0:
            print("BEIR failure analysis gate rejected: no labeled queries", file=sys.stderr)
            return 2
        hit_rate = report["queries_with_positive_relevant_hit"] / report["labeled_queries"]
        if hit_rate < args.min_hit_rate:
            print(f"BEIR hit@{args.k} gate failed: {hit_rate:.6f} < {args.min_hit_rate:.6f}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
