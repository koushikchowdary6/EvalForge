"""API-free, deterministic BM25 retrieval baseline for the bundled RAG benchmark.

This is *retrieval only*: it never generates answers or measures groundedness,
hallucinations, or LLM performance. It uses the same retrieval graders as the
embedding-backed RAG evaluator for an honest baseline comparison.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from evalforge.identifiers import valid_benchmark_id
from evalforge.rag_graders import (
    average_precision_at_k, hit_rate, ndcg_at_k, precision_at_k,
    recall_at_k, reciprocal_rank,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets"
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text):
    """Transparent lexical tokenizer, not an embedding or language model."""
    return _TOKEN.findall(text.casefold())


def validate(docs, questions):
    """Reject ambiguous benchmark IDs/labels before computing any scores."""
    if not isinstance(docs, list) or not docs:
        raise ValueError("corpus must be a nonempty list")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    ids = set()
    for doc in docs:
        if (not isinstance(doc, dict)
                or not valid_benchmark_id(doc.get("doc_id"))
                or not isinstance(doc.get("title"), str)
                or not isinstance(doc.get("text"), str)
                or not (doc["title"].strip() or doc["text"].strip())):
            raise ValueError("invalid corpus document")
        if doc["doc_id"] in ids:
            raise ValueError("duplicate corpus document ID")
        ids.add(doc["doc_id"])
    case_ids = set()
    for case in questions:
        if (not isinstance(case, dict)
                or not valid_benchmark_id(case.get("id"))
                or not isinstance(case.get("question"), str)
                or not case["question"].strip()
                or type(case.get("answerable")) is not bool
                or not isinstance(case.get("relevant_doc_ids"), list)):
            raise ValueError("invalid benchmark question")
        if case["id"] in case_ids:
            raise ValueError("duplicate question ID")
        case_ids.add(case["id"])
        labels = case["relevant_doc_ids"]
        if (any(not isinstance(doc_id, str) or doc_id not in ids for doc_id in labels)
                or len(set(labels)) != len(labels)
                or (case["answerable"] and not labels)
                or (not case["answerable"] and labels)):
            raise ValueError("invalid relevance labels")


def bm25_rank_with_scores(question, docs):
    """Return (document ID, BM25 score) pairs, ordered by score then ID."""
    tokens = [tokenize(doc["title"] + " " + doc["text"]) for doc in docs]
    lengths = [len(words) for words in tokens]
    average_length = max(1.0, sum(lengths) / len(lengths))
    frequencies = [Counter(words) for words in tokens]
    document_frequency = Counter()
    for freq in frequencies:
        document_frequency.update(freq.keys())
    query_terms = set(tokenize(question))
    scores = []
    for doc, freq, length in zip(docs, frequencies, lengths):
        score = 0.0
        for term in query_terms:
            count = freq[term]
            if not count:
                continue
            df = document_frequency[term]
            idf = math.log1p((len(docs) - df + 0.5) / (df + 0.5))
            score += idf * count * 2.2 / (
                count + 1.2 * (0.25 + 0.75 * length / average_length)
            )
        scores.append((doc["doc_id"], score))
    return sorted(scores, key=lambda row: (-row[1], row[0]))


def bm25_rank(question, docs):
    """Rank corpus by BM25 (k1=1.2, b=0.75), breaking ties by doc ID."""
    return [doc_id for doc_id, _ in bm25_rank_with_scores(question, docs)]


def _unique_json_keys(pairs):
    """Reject ambiguous duplicate keys anywhere in benchmark JSON."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in benchmark input")
        result[key] = value
    return result


def _reject_nonfinite_json(value):
    """NaN and Infinity are not JSON and cannot be benchmark provenance."""
    raise ValueError("non-finite JSON constant in benchmark input")


def load_benchmark_json(path):
    """Read unambiguous, standards-compliant JSON before ranking documents."""
    with path.open(encoding="utf-8") as stream:
        return json.load(
            stream, object_pairs_hook=_unique_json_keys,
            parse_constant=_reject_nonfinite_json,
        )


def _fingerprint(value):
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate(docs, questions, cutoffs=(1, 3, 5)):
    validate(docs, questions)
    if (not isinstance(cutoffs, (list, tuple)) or not cutoffs
            or any(type(k) is not int or k < 1 for k in cutoffs)
            or len(set(cutoffs)) != len(cutoffs)):
        raise ValueError("cutoffs must be unique positive integers")
    scored_rankings = {
        case["id"]: bm25_rank_with_scores(case["question"], docs)
        for case in questions
    }
    # The baseline intentionally uses an ASCII-only lexical tokenizer. Record
    # where that tokenizer drops an entire query or encounters non-ASCII
    # alphanumeric text, so tie-only retrieval is not mistaken for coverage.
    query_token_counts = {
        case["id"]: len(tokenize(case["question"])) for case in questions
    }
    query_has_non_ascii_alphanumeric = {
        case["id"]: any(ch.isalnum() and not ch.isascii() for ch in case["question"])
        for case in questions
    }
    rankings = {
        case_id: [doc_id for doc_id, _ in scored]
        for case_id, scored in scored_rankings.items()
    }
    lexical_match_counts = {
        case_id: sum(score > 0 for _, score in scored)
        for case_id, scored in scored_rankings.items()
    }
    # A lexical match on an unanswerable question is a useful abstention
    # diagnostic, not evidence that a generated answer was hallucinated.
    unanswerable_count = sum(not case["answerable"] for case in questions)
    unanswerable_with_lexical_matches = sum(
        not case["answerable"] and lexical_match_counts[case["id"]] > 0
        for case in questions
    )
    cases = []
    summaries = {}
    for k in cutoffs:
        rows = []
        evidence_rows = []
        for case in questions:
            ranked = rankings[case["id"]][:k]
            # Zero-score BM25 results are arbitrary ID-order tie breakers,
            # not evidence that the query matched a document.
            evidence_ranked = [
                doc_id for doc_id, score in scored_rankings[case["id"]]
                if score > 0
            ][:k]
            relevant = case["relevant_doc_ids"]
            row = {
                "case_id": case["id"],
                "top_k": k,
                "answerable": case["answerable"],
                "retrieved_ids": ranked,
                "positive_score_retrieved_ids": evidence_ranked,
                "relevant_ids": relevant,
                "lexical_positive_document_count": lexical_match_counts[case["id"]],
                "all_zero_score_ranking": lexical_match_counts[case["id"]] == 0,
                "query_ascii_token_count": query_token_counts[case["id"]],
                "query_has_no_ascii_tokens": query_token_counts[case["id"]] == 0,
                "query_contains_non_ascii_alphanumeric": (
                    query_has_non_ascii_alphanumeric[case["id"]]
                ),
            }
            if case["answerable"]:
                row["metrics"] = {
                    "precision_at_k": precision_at_k(ranked, relevant, k=k),
                    "recall_at_k": recall_at_k(ranked, relevant, k=k),
                    "ndcg_at_k": ndcg_at_k(ranked, relevant, k=k),
                    "average_precision_at_k": average_precision_at_k(ranked, relevant, k=k),
                    "reciprocal_rank": reciprocal_rank(ranked, relevant),
                    "hit_rate": hit_rate(ranked, relevant),
                }
                rows.append(row["metrics"])
                # Keep legacy rank metrics for comparisons, but report
                # evidence-backed quality separately so zero-score ID ties
                # cannot masquerade as successful lexical retrieval.
                row["lexical_evidence_metrics"] = {
                    "hit_rate": hit_rate(evidence_ranked, relevant),
                    "recall_at_k": recall_at_k(evidence_ranked, relevant, k=k),
                    "reciprocal_rank": reciprocal_rank(evidence_ranked, relevant),
                }
                evidence_rows.append(row["lexical_evidence_metrics"])
            else:
                # Unanswerable questions have no relevant documents. Their
                # retrieval scores are undefined for this answerable-only report.
                row["metrics"] = None
                row["lexical_evidence_metrics"] = None
                row["unanswerable_positive_lexical_match"] = (
                    lexical_match_counts[case["id"]] > 0
                )
            cases.append(row)
        summaries[str(k)] = {
            "answerable_count": len(rows),
            "answerable_without_ascii_query_tokens": sum(
                case["answerable"] and query_token_counts[case["id"]] == 0
                for case in questions
            ),
            "unanswerable_without_ascii_query_tokens": sum(
                not case["answerable"] and query_token_counts[case["id"]] == 0
                for case in questions
            ),
            "questions_with_non_ascii_alphanumeric": sum(
                query_has_non_ascii_alphanumeric.values()
            ),
            "unanswerable_excluded": len(questions) - len(rows),
            "unanswerable_positive_lexical_match_count": unanswerable_with_lexical_matches,
            "unanswerable_positive_lexical_match_rate": (
                unanswerable_with_lexical_matches / unanswerable_count
                if unanswerable_count else None
            ),
            "answerable_zero_lexical_overlap_count": sum(
                case["answerable"] and lexical_match_counts[case["id"]] == 0
                for case in questions
            ),
            "unanswerable_zero_lexical_overlap_count": sum(
                not case["answerable"] and lexical_match_counts[case["id"]] == 0
                for case in questions
            ),
            # An unanswerable-only benchmark has no relevant-document
            # denominator. Preserve its coverage diagnostics, but report
            # undefined retrieval quality as null instead of dividing by zero.
            "lexical_evidence_hit_rate": (
                sum(row["hit_rate"] for row in evidence_rows) / len(evidence_rows)
                if evidence_rows else None
            ),
            "lexical_evidence_recall_at_k": (
                sum(row["recall_at_k"] for row in evidence_rows) / len(evidence_rows)
                if evidence_rows else None
            ),
            "lexical_evidence_mrr": (
                sum(row["reciprocal_rank"] for row in evidence_rows) / len(evidence_rows)
                if evidence_rows else None
            ),
            "tie_only_hit_count": sum(
                legacy["hit_rate"] > 0 and evidence["hit_rate"] == 0
                for legacy, evidence in zip(rows, evidence_rows)
            ),
            **{name: (sum(row[name] for row in rows) / len(rows) if rows else None)
               for name in ("precision_at_k", "recall_at_k", "ndcg_at_k",
                            "average_precision_at_k", "reciprocal_rank", "hit_rate")},
        }
    return {
        "method": "lexical_bm25_retrieval_only",
        "parameters": {"k1": 1.2, "b": 0.75, "tokenizer": "ascii_alphanumeric"},
        "corpus_sha256": _fingerprint(docs),
        "questions_sha256": _fingerprint(questions),
        "corpus_size": len(docs),
        "question_count": len(questions),
        "tokenizer_diagnostics": {
            "questions_without_ascii_tokens": sum(count == 0 for count in query_token_counts.values()),
            "questions_with_non_ascii_alphanumeric": sum(
                query_has_non_ascii_alphanumeric.values()
            ),
        },
        "by_top_k": summaries,
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "rag_corpus.json")
    parser.add_argument("--questions", type=Path, default=DATA_DIR / "rag_questions.json")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args(argv)
    try:
        docs = load_benchmark_json(args.corpus)
        questions = load_benchmark_json(args.questions)
        report = evaluate(docs, questions, args.k)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
