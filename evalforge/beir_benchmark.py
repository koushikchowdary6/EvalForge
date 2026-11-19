"""Evaluate locally obtained BEIR-style retrieval data without intermediate files.

This is an offline lexical BM25 baseline, not a generative-LLM benchmark.
Only queries with positive qrels are scored; missing qrels do not imply
unanswerable questions. Source files are never downloaded or redistributed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .beir_import import convert
from .offline_rag_demo import evaluate


def benchmark(corpus: Path, queries: Path, qrels: Path, *, cutoffs=(1, 3, 5),
              tokenizer="ascii"):
    documents, questions, provenance = convert(corpus, queries, qrels)
    result = evaluate(documents, questions, cutoffs=cutoffs, tokenizer_mode=tokenizer)
    return {
        "schema_version": 1,
        "benchmark_scope": "offline_lexical_bm25_positive_qrels_only",
        "source": provenance,
        "evaluation": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="BEIR corpus.jsonl")
    parser.add_argument("--queries", required=True, type=Path, help="BEIR queries.jsonl")
    parser.add_argument("--qrels", required=True, type=Path, help="BEIR qrels TSV")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--tokenizer", choices=("ascii", "unicode"), default="ascii")
    parser.add_argument("--output", type=Path, help="New JSON report path; refuses overwrite")
    args = parser.parse_args(argv)
    try:
        # Reject invalid cutoffs before parsing potentially large external datasets.
        # Keep the same positive-integer contract as the retrieval evaluator.
        if any(k <= 0 for k in args.k):
            raise ValueError("retrieval cutoffs must be positive integers")
        if args.output and args.output.resolve() in {
            args.corpus.resolve(), args.queries.resolve(), args.qrels.resolve()
        }:
            raise ValueError("report path cannot overwrite an input")
        if args.output and args.output.exists():
            raise ValueError("report already exists; refusing to overwrite")
        result = benchmark(args.corpus, args.queries, args.qrels,
                           cutoffs=args.k, tokenizer=args.tokenizer)
        serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            created = False
            try:
                with args.output.open("x", encoding="utf-8") as stream:
                    created = True
                    stream.write(serialized)
            except OSError:
                # Do not leave a truncated report if writing fails (e.g. full disk).
                # Never delete a pre-existing file rejected by exclusive creation.
                if created:
                    args.output.unlink(missing_ok=True)
                raise
        else:
            sys.stdout.write(serialized)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        print(f"BEIR benchmark rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
