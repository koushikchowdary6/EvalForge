"""Compare query-level BEIR BM25 failures across two tokenizers.

This is a paired lexical retrieval diagnostic, not an LLM quality benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .beir_failures import analyze


def compare(corpus: Path, queries: Path, qrels: Path, *, k: int = 5, limit: int = 20):
    """Identify labeled queries recovered or lost by Unicode tokenization."""
    if type(k) is not int or k < 1 or type(limit) is not int or limit < 1:
        raise ValueError("k and limit must be positive integers")
    # Analyze every failure, not only the first N; the display limit applies afterward.
    ascii_report = analyze(corpus, queries, qrels, k=k, tokenizer="ascii", limit=2**31 - 1)
    unicode_report = analyze(corpus, queries, qrels, k=k, tokenizer="unicode", limit=2**31 - 1)
    if ascii_report["source"] != unicode_report["source"]:
        raise ValueError("source fingerprints differ between tokenizer runs")
    ascii_failed = {row["query_id"]: row for row in ascii_report["shown_failures"]}
    unicode_failed = {row["query_id"]: row for row in unicode_report["shown_failures"]}
    recovered = sorted(set(ascii_failed) - set(unicode_failed))
    regressed = sorted(set(unicode_failed) - set(ascii_failed))
    shared = sorted(set(ascii_failed) & set(unicode_failed))
    changes = [
        {"query_id": query_id, "query": (ascii_failed.get(query_id) or unicode_failed[query_id])["query"],
         "change": "recovered_by_unicode" if query_id in recovered else "lost_by_unicode",
         "ascii_failure_reason": ascii_failed[query_id]["reason"] if query_id in ascii_failed else None,
         "unicode_failure_reason": unicode_failed[query_id]["reason"] if query_id in unicode_failed else None}
        for query_id in sorted(set(recovered) | set(regressed))
    ]
    return {
        "schema_version": 1,
        "scope": "paired_bm25_ascii_unicode_positive_qrels_only",
        "source": ascii_report["source"],
        "k": k,
        "labeled_queries": ascii_report["labeled_queries"],
        "ascii_positive_hit_queries": ascii_report["queries_with_positive_relevant_hit"],
        "unicode_positive_hit_queries": unicode_report["queries_with_positive_relevant_hit"],
        "recovered_by_unicode": len(recovered),
        "lost_by_unicode": len(regressed),
        "failed_by_both": len(shared),
        "unchanged_successes": ascii_report["labeled_queries"] - len(recovered) - len(regressed) - len(shared),
        "shown_changes": changes[:limit],
        "truncated": len(changes) > limit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, help="New JSON report; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.output and (args.output.exists() or args.output.resolve() in {
            args.corpus.resolve(), args.queries.resolve(), args.qrels.resolve()
        }):
            raise ValueError("output exists or matches an input file")
        result = compare(args.corpus, args.queries, args.qrels, k=args.k, limit=args.limit)
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
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
    except (OSError, ValueError, UnicodeError, TypeError, RecursionError) as exc:
        print(f"BEIR tokenizer comparison rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
