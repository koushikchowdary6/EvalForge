"""Paired, API-free ASCII versus Unicode BM25 retrieval audit.

This compares *the same* documents, questions, labels, and top-k cutoffs.
It measures lexical evidence only, not multilingual semantic understanding,
model accuracy, statistical significance, or generated-answer quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalforge.offline_rag_demo import DATA_DIR, evaluate, load_benchmark_json


def compare_tokenizers(docs, questions, cutoffs=(1, 3, 5)):
    """Report both gains and regressions without cherry-picking a tokenizer."""
    # Duplicate or nonpositive cutoffs can silently collapse by_top_k keys or
    # produce misleading benchmark comparisons. Reject them before evaluation.
    cutoffs = tuple(cutoffs)
    if not cutoffs or any(type(k) is not int or k <= 0 for k in cutoffs):
        raise ValueError("top-k cutoffs must be nonempty positive integers")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("top-k cutoffs must be unique")
    ascii_report = evaluate(docs, questions, cutoffs, tokenizer_mode="ascii")
    unicode_report = evaluate(docs, questions, cutoffs, tokenizer_mode="unicode")
    assert ascii_report["corpus_sha256"] == unicode_report["corpus_sha256"]
    assert ascii_report["questions_sha256"] == unicode_report["questions_sha256"]

    ascii_rows = {(row["case_id"], row["top_k"]): row
                  for row in ascii_report["cases"]}
    unicode_rows = {(row["case_id"], row["top_k"]): row
                    for row in unicode_report["cases"]}
    by_top_k = {}
    cases = []
    for k in cutoffs:
        both = ascii_only = unicode_only = neither = 0
        ascii_recall_total = unicode_recall_total = 0.0
        paired_delta_lower_total = paired_delta_upper_total = 0.0
        paired_hit_lower_total = paired_hit_upper_total = 0
        tie_ambiguous_case_count = hit_tie_ambiguous_case_count = 0
        answerable_count = unanswerable_count = 0
        ascii_unanswerable_match = unicode_unanswerable_match = 0
        unanswerable_both = unanswerable_ascii_only = 0
        unanswerable_unicode_only = unanswerable_neither = 0
        for question in questions:
            key = (question["id"], k)
            a, u = ascii_rows[key], unicode_rows[key]
            if question["answerable"]:
                answerable_count += 1
                a_metrics, u_metrics = (
                    a["lexical_evidence_metrics"],
                    u["lexical_evidence_metrics"],
                )
                a_hit = bool(a_metrics["hit_rate"])
                u_hit = bool(u_metrics["hit_rate"])
                if a_hit and u_hit:
                    both += 1
                elif a_hit:
                    ascii_only += 1
                elif u_hit:
                    unicode_only += 1
                else:
                    neither += 1
                a_recall = a_metrics["recall_at_k"]
                u_recall = u_metrics["recall_at_k"]
                ascii_recall_total += a_recall
                unicode_recall_total += u_recall
                # Tied positive BM25 scores have no meaningful ID ordering.
                # The paired difference can vary even when the reported
                # deterministic (ID-tiebroken) point estimate is unchanged.
                delta_lower = (
                    u["lexical_evidence_recall_lower_bound"]
                    - a["lexical_evidence_recall_upper_bound"]
                )
                delta_upper = (
                    u["lexical_evidence_recall_upper_bound"]
                    - a["lexical_evidence_recall_lower_bound"]
                )
                paired_delta_lower_total += delta_lower
                paired_delta_upper_total += delta_upper
                tie_ambiguous_case_count += int(delta_lower != delta_upper)
                # A positive-score cutoff tie can also reverse hit@k,
                # even when the deterministic ID tie-break reports no change.
                # Recall bounds are fractions of labeled relevant documents;
                # any positive recall is a hit, including at the upper bound.
                hit_delta_lower = (
                    int(u["lexical_evidence_recall_lower_bound"] > 0)
                    - int(a["lexical_evidence_recall_upper_bound"] > 0)
                )
                hit_delta_upper = (
                    int(u["lexical_evidence_recall_upper_bound"] > 0)
                    - int(a["lexical_evidence_recall_lower_bound"] > 0)
                )
                paired_hit_lower_total += hit_delta_lower
                paired_hit_upper_total += hit_delta_upper
                hit_tie_ambiguous_case_count += int(hit_delta_lower != hit_delta_upper)
                a_unanswerable = u_unanswerable = None
            else:
                unanswerable_count += 1
                a_hit = u_hit = a_recall = u_recall = None
                delta_lower = delta_upper = None
                hit_delta_lower = hit_delta_upper = None
                a_unanswerable = bool(a["unanswerable_positive_lexical_match"])
                u_unanswerable = bool(u["unanswerable_positive_lexical_match"])
                ascii_unanswerable_match += int(a_unanswerable)
                unicode_unanswerable_match += int(u_unanswerable)
                if a_unanswerable and u_unanswerable:
                    unanswerable_both += 1
                elif a_unanswerable:
                    unanswerable_ascii_only += 1
                elif u_unanswerable:
                    unanswerable_unicode_only += 1
                else:
                    unanswerable_neither += 1

            cases.append({
                "case_id": question["id"],
                "top_k": k,
                "answerable": question["answerable"],
                "ascii_positive_score_retrieved_ids": a["positive_score_retrieved_ids"],
                "unicode_positive_score_retrieved_ids": u["positive_score_retrieved_ids"],
                "ascii_evidence_hit": a_hit,
                "unicode_evidence_hit": u_hit,
                "ascii_evidence_recall": a_recall,
                "unicode_evidence_recall": u_recall,
                "paired_recall_delta_unicode_minus_ascii": (
                    u_recall - a_recall if question["answerable"] else None
                ),
                "paired_recall_delta_lower_bound_unicode_minus_ascii": delta_lower,
                "paired_recall_delta_upper_bound_unicode_minus_ascii": delta_upper,
                "paired_hit_delta_lower_bound_unicode_minus_ascii": hit_delta_lower,
                "paired_hit_delta_upper_bound_unicode_minus_ascii": hit_delta_upper,
                "ascii_unanswerable_positive_lexical_match": a_unanswerable,
                "unicode_unanswerable_positive_lexical_match": u_unanswerable,
                "paired_unanswerable_match_delta_unicode_minus_ascii": (
                    int(u_unanswerable) - int(a_unanswerable)
                    if not question["answerable"] else None
                ),
            })

        a_hit_rate = (both + ascii_only) / answerable_count if answerable_count else None
        u_hit_rate = (both + unicode_only) / answerable_count if answerable_count else None
        a_mean_recall = ascii_recall_total / answerable_count if answerable_count else None
        u_mean_recall = unicode_recall_total / answerable_count if answerable_count else None
        by_top_k[str(k)] = {
            "answerable_count": answerable_count,
            "both_hit": both,
            "ascii_only_hit": ascii_only,
            "unicode_only_hit": unicode_only,
            "neither_hit": neither,
            "ascii_evidence_hit_rate": a_hit_rate,
            "unicode_evidence_hit_rate": u_hit_rate,
            "paired_hit_rate_delta_unicode_minus_ascii": (
                u_hit_rate - a_hit_rate if answerable_count else None
            ),
            "paired_hit_rate_delta_lower_bound_unicode_minus_ascii": (
                paired_hit_lower_total / answerable_count if answerable_count else None
            ),
            "paired_hit_rate_delta_upper_bound_unicode_minus_ascii": (
                paired_hit_upper_total / answerable_count if answerable_count else None
            ),
            "paired_hit_delta_tie_ambiguous_cases": hit_tie_ambiguous_case_count,
            "ascii_mean_evidence_recall": a_mean_recall,
            "unicode_mean_evidence_recall": u_mean_recall,
            "paired_mean_recall_delta_unicode_minus_ascii": (
                u_mean_recall - a_mean_recall if answerable_count else None
            ),
            "paired_mean_recall_delta_lower_bound_unicode_minus_ascii": (
                paired_delta_lower_total / answerable_count if answerable_count else None
            ),
            "paired_mean_recall_delta_upper_bound_unicode_minus_ascii": (
                paired_delta_upper_total / answerable_count if answerable_count else None
            ),
            "paired_recall_delta_tie_ambiguous_cases": tie_ambiguous_case_count,
            "unanswerable_count": unanswerable_count,
            "ascii_unanswerable_positive_lexical_match_count": ascii_unanswerable_match,
            "unicode_unanswerable_positive_lexical_match_count": unicode_unanswerable_match,
            # An unanswerable lexical match is *not* a generated hallucination.
            # Report paired discordance rather than only separate totals:
            # aggregate rates alone can conceal offsetting regressions.
            "unanswerable_both_match": unanswerable_both,
            "unanswerable_ascii_only_match": unanswerable_ascii_only,
            "unanswerable_unicode_only_match": unanswerable_unicode_only,
            "unanswerable_neither_match": unanswerable_neither,
            "ascii_unanswerable_lexical_match_rate": (
                ascii_unanswerable_match / unanswerable_count
                if unanswerable_count else None
            ),
            "unicode_unanswerable_lexical_match_rate": (
                unicode_unanswerable_match / unanswerable_count
                if unanswerable_count else None
            ),
            "paired_unanswerable_match_rate_delta_unicode_minus_ascii": (
                (unanswerable_unicode_only - unanswerable_ascii_only) / unanswerable_count
                if unanswerable_count else None
            ),
        }

    return {
        "method": "paired_ascii_unicode_bm25_lexical_retrieval_only",
        "corpus_sha256": ascii_report["corpus_sha256"],
        "questions_sha256": ascii_report["questions_sha256"],
        "cutoffs": list(cutoffs),
        "question_count": len(questions),
        "scope": (
            "paired_descriptive_lexical_overlap_on_supplied_labels; "
            "hit_and_recall_delta_bounds_are_worst_best_cases_over_independent_positive_score_tie_orders; "
            "not_confidence_intervals_semantic_accuracy_or_statistical_significance"
        ),
        "by_top_k": by_top_k,
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=DATA_DIR / "multilingual_lexical_corpus.json"
    )
    parser.add_argument(
        "--questions", type=Path, default=DATA_DIR / "multilingual_lexical_questions.json"
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Omit per-question cases while retaining paired metrics and dataset fingerprints",
    )
    args = parser.parse_args(argv)
    try:
        docs = load_benchmark_json(args.corpus)
        questions = load_benchmark_json(args.questions)
        report = compare_tokenizers(docs, questions, args.k)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        parser.error(str(exc))
    if args.summary_only:
        report = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
