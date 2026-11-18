"""Preflight labeled retrieval datasets before running EvalForge benchmarks.

Reports dataset integrity and lexical coverage; never calls an LLM or grades answers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .offline_rag_demo import (
    _fingerprint, load_benchmark_json, tokenize, tokenize_unicode, validate,
)


def audit(docs: list, questions: list, tokenizer: str = "ascii") -> dict:
    """Return reproducible corpus/question diagnostics after strict validation."""
    validate(docs, questions)
    if tokenizer not in ("ascii", "unicode"):
        raise ValueError("tokenizer must be ascii or unicode")
    split = tokenize if tokenizer == "ascii" else tokenize_unicode
    vocabulary = set()
    unindexable = set()
    document_terms = {}
    for doc in docs:
        terms = set(split(doc["title"] + " " + doc["text"]))
        vocabulary.update(terms)
        document_terms[doc["doc_id"]] = terms
        if not terms:
            unindexable.add(doc["doc_id"])
    no_overlap = []
    unindexable_labels = []
    no_relevant_overlap = []
    answerable = 0
    unanswerable_with_overlap = []
    for case in questions:
        query_terms = set(split(case["question"]))
        if not case["answerable"]:
            if query_terms.intersection(vocabulary):
                unanswerable_with_overlap.append(case["id"])
            continue
        answerable += 1
        if not query_terms.intersection(vocabulary):
            no_overlap.append(case["id"])
        relevant_vocabulary = set().union(*(document_terms[doc_id] for doc_id in case["relevant_doc_ids"]))
        if not query_terms.intersection(relevant_vocabulary):
            no_relevant_overlap.append(case["id"])
        if set(case["relevant_doc_ids"]).intersection(unindexable):
            unindexable_labels.append(case["id"])
    return {
        "schema": "evalforge.dataset_audit.v1",
        "scope": "lexical_preflight_not_retrieval_accuracy_or_llm_safety",
        "tokenizer": tokenizer,
        "documents": len(docs),
        "questions": len(questions),
        "answerable_questions": answerable,
        "unanswerable_questions": len(questions) - answerable,
        # Lexical overlap is a warning, not evidence of hallucination.
        "unanswerable_question_ids_with_corpus_token_overlap": sorted(unanswerable_with_overlap),
        "vocabulary_size": len(vocabulary),
        "unindexable_document_ids": sorted(unindexable),
        "answerable_question_ids_without_corpus_token_overlap": sorted(no_overlap),
        "question_ids_with_unindexable_relevance_labels": sorted(unindexable_labels),
        "answerable_question_ids_without_relevant_doc_token_overlap": sorted(no_relevant_overlap),
        "corpus_sha256": _fingerprint(docs),
        "questions_sha256": _fingerprint(questions),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--tokenizer", choices=("ascii", "unicode"), default="ascii")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on unreachable lexical labels or no query/corpus token overlap")
    args = parser.parse_args(argv)
    try:
        report = audit(load_benchmark_json(args.corpus),
                       load_benchmark_json(args.questions), args.tokenizer)
    except (ValueError, OSError, TypeError, RecursionError) as exc:
        print(f"EvalForge dataset preflight rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if args.strict and (report["unindexable_document_ids"]
                        or report["answerable_question_ids_without_corpus_token_overlap"]
                        or report["question_ids_with_unindexable_relevance_labels"]
                        or report["answerable_question_ids_without_relevant_doc_token_overlap"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
