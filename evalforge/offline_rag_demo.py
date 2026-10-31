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
import os
from pathlib import Path
import re
import stat
import unicodedata

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


def tokenize_unicode(text):
    """Unicode word baseline, retaining combining marks within words.

    Python's word regex excludes combining vowel signs and accents. Splitting
    on them turns one Devanagari word into unrelated fragments and can create
    misleading lexical matches. Preserve marks attached to a preceding word
    character, but do not treat isolated marks or punctuation as terms.
    This remains lexical only: no stemming, translation, or CJK segmentation.
    """
    normalized = unicodedata.normalize("NFC", text.casefold())
    tokens, current = [], []
    for char in normalized:
        if char.isalnum() or (
            current and unicodedata.category(char).startswith("M")
        ):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens

def _select_tokenizer(mode):
    if mode == "ascii":
        return tokenize
    if mode == "unicode":
        return tokenize_unicode
    raise ValueError("tokenizer_mode must be 'ascii' or 'unicode'")


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


def _build_bm25_index(docs, tokenizer=None):
    """Compute immutable-per-evaluation corpus statistics once, not per query."""
    if tokenizer is None:
        tokenizer = tokenize
    tokens = [tokenizer(doc["title"] + " " + doc["text"]) for doc in docs]
    lengths = [len(words) for words in tokens]
    average_length = max(1.0, sum(lengths) / len(lengths))
    frequencies = [Counter(words) for words in tokens]
    document_frequency = Counter()
    for freq in frequencies:
        document_frequency.update(freq.keys())
    return frequencies, lengths, average_length, document_frequency


def _rank_with_bm25_index(question, docs, index, tokenizer=None):
    """Score a query against the same corpus snapshot as every other query."""
    if tokenizer is None:
        tokenizer = tokenize
    frequencies, lengths, average_length, document_frequency = index
    query_terms = set(tokenizer(question))
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


def _lexical_equivalence_diagnostics(docs, questions, frequencies, tokenizer_mode):
    """Audit distinct IDs that BM25 cannot distinguish by term counts.

    Documents with identical token-count multisets have identical lengths,
    term frequencies, and BM25 scores for every query. If relevance labels
    mark only some IDs in such a group, ID tie-breaking can change measured
    retrieval quality without a meaningful lexical distinction. This is an
    evaluation-label warning, not evidence of train/test contamination.
    """
    by_signature = {}
    for doc, counts in zip(docs, frequencies):
        signature = tuple(sorted(counts.items()))
        by_signature.setdefault(signature, []).append(doc["doc_id"])
    groups = sorted(
        (sorted(ids) for ids in by_signature.values() if len(ids) > 1),
        key=lambda ids: ids[0],
    )
    partial_case_ids = []
    for case in questions:
        if not case["answerable"]:
            continue
        relevant = set(case["relevant_doc_ids"])
        if any(relevant.intersection(group) and not relevant.issuperset(group)
               for group in groups):
            partial_case_ids.append(case["id"])
    return {
        "tokenizer_mode": tokenizer_mode,
        "equivalent_group_count": len(groups),
        "equivalent_document_count": sum(len(group) for group in groups),
        "equivalent_document_id_groups": groups,
        "answerable_cases_with_partial_equivalence_labels": partial_case_ids,
        "scope": "identical_bm25_token_frequency_multisets_not_semantic_duplicates",
    }



def _positive_score_hit_bounds(scored, relevant, k):
    """Bound hit@k over every ordering of a positive-score cutoff tie.

    BM25 assigns the same score to multiple documents, then uses document IDs
    to choose which fit at k. The returned bounds consider all equally scored
    choices, without claiming any of them is semantically preferable.
    Zero-score documents never count as lexical evidence.
    """
    positive = [(doc_id, score) for doc_id, score in scored if score > 0]
    if not positive:
        return 0, 0
    relevant = set(relevant)
    if len(positive) <= k or positive[k - 1][1] != positive[k][1]:
        hit = int(any(doc_id in relevant for doc_id, _ in positive[:k]))
        return hit, hit

    boundary = positive[k - 1][1]
    fixed = [doc_id for doc_id, score in positive if score > boundary]
    tied = [doc_id for doc_id, score in positive if score == boundary]
    available_slots = k - len(fixed)
    relevant_in_tie = sum(doc_id in relevant for doc_id in tied)
    minimum_tie_hits = max(0, available_slots - (len(tied) - relevant_in_tie))
    maximum_tie_hits = min(available_slots, relevant_in_tie)
    fixed_hit = any(doc_id in relevant for doc_id in fixed)
    return int(fixed_hit or minimum_tie_hits > 0), int(fixed_hit or maximum_tie_hits > 0)


def bm25_rank_with_scores(question, docs, *, tokenizer_mode="ascii"):
    """Return (document ID, BM25 score) pairs, ordered by score then ID."""
    selected = _select_tokenizer(tokenizer_mode)
    return _rank_with_bm25_index(
        question, docs, _build_bm25_index(docs, selected), selected
    )


def bm25_rank(question, docs, *, tokenizer_mode="ascii"):
    """Rank corpus by BM25 (k1=1.2, b=0.75), breaking ties by doc ID."""
    return [
        doc_id for doc_id, _ in
        bm25_rank_with_scores(question, docs, tokenizer_mode=tokenizer_mode)
    ]


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


MAX_BENCHMARK_JSON_BYTES = 2_000_000


def load_benchmark_json(path):
    """Read bounded, regular-file JSON before ranking or reporting metrics."""
    source = Path(path)
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("benchmark input must be a regular file")
    if metadata.st_size > MAX_BENCHMARK_JSON_BYTES:
        raise ValueError("benchmark input exceeds 2,000,000 bytes")

    # Check the opened descriptor as well as the path: a swapped FIFO must
    # not block the CLI, and a replacement symlink must not escape the input.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(source, flags), "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("benchmark input must be a regular file")
        raw = handle.read(MAX_BENCHMARK_JSON_BYTES + 1)
    if len(raw) > MAX_BENCHMARK_JSON_BYTES:
        raise ValueError("benchmark input exceeds 2,000,000 bytes")
    try:
        payload = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("benchmark input must be UTF-8") from exc
    return json.loads(
        payload, object_pairs_hook=_unique_json_keys,
        parse_constant=_reject_nonfinite_json,
    )


def _fingerprint(value):
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate(docs, questions, cutoffs=(1, 3, 5), *, tokenizer_mode="ascii"):
    validate(docs, questions)
    selected_tokenizer = _select_tokenizer(tokenizer_mode)
    if (not isinstance(cutoffs, (list, tuple)) or not cutoffs
            or any(type(k) is not int or k < 1 for k in cutoffs)
            or len(set(cutoffs)) != len(cutoffs)):
        raise ValueError("cutoffs must be unique positive integers")
    # The corpus is identical for every question in this evaluation. Reuse
    # document frequencies and lengths rather than retokenizing all documents
    # for each question; retain the standalone ranking API for callers.
    index = _build_bm25_index(docs, selected_tokenizer)
    lexical_equivalence = _lexical_equivalence_diagnostics(
        docs, questions, index[0], tokenizer_mode
    )
    scored_rankings = {
        case["id"]: _rank_with_bm25_index(
            case["question"], docs, index, selected_tokenizer
        )
        for case in questions
    }
    # The baseline intentionally uses an ASCII-only lexical tokenizer. Record
    # where that tokenizer drops an entire query or encounters non-ASCII
    # alphanumeric text, so tie-only retrieval is not mistaken for coverage.
    query_token_counts = {
        case["id"]: len(tokenize(case["question"])) for case in questions
    }
    selected_query_token_counts = (
        query_token_counts if tokenizer_mode == "ascii" else {
            case["id"]: len(selected_tokenizer(case["question"]))
            for case in questions
        }
    )
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
        cutoff_cases = []
        for case in questions:
            scored = scored_rankings[case["id"]]
            ranked = rankings[case["id"]][:k]
            # A positive BM25 tie straddling the requested cutoff means the
            # selected IDs depend on the arbitrary secondary doc-ID sort.
            # Zero-score ties are already handled by evidence-only metrics.
            boundary_tie = (
                k < len(scored) and scored[k - 1][1] > 0
                and scored[k - 1][1] == scored[k][1]
            )
            tied_ids = (
                [doc_id for doc_id, score in scored
                 if score == scored[k - 1][1]]
                if boundary_tie else []
            )
            relevant_at_boundary = set(tied_ids).intersection(
                case["relevant_doc_ids"]
            )
            mixed_boundary_labels = (
                bool(relevant_at_boundary)
                and len(relevant_at_boundary) != len(tied_ids)
            )
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
                "positive_score_cutoff_tie": boundary_tie,
                "positive_score_cutoff_tied_doc_ids": tied_ids,
                "positive_score_cutoff_tie_mixed_relevance": mixed_boundary_labels,
                "query_ascii_token_count": query_token_counts[case["id"]],
                "query_selected_token_count": selected_query_token_counts[case["id"]],
                "query_has_no_selected_tokens": selected_query_token_counts[case["id"]] == 0,
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
                    "precision_at_k": precision_at_k(evidence_ranked, relevant, k=k),
                    "recall_at_k": recall_at_k(evidence_ranked, relevant, k=k),
                    "ndcg_at_k": ndcg_at_k(evidence_ranked, relevant, k=k),
                    "average_precision_at_k": average_precision_at_k(
                        evidence_ranked, relevant, k=k
                    ),
                    "reciprocal_rank": reciprocal_rank(evidence_ranked, relevant),
                    "hit_rate": hit_rate(evidence_ranked, relevant),
                }
                lower_hit, upper_hit = _positive_score_hit_bounds(scored, relevant, k)
                row["lexical_evidence_hit_lower_bound"] = lower_hit
                row["lexical_evidence_hit_upper_bound"] = upper_hit
                evidence_rows.append(row["lexical_evidence_metrics"])
            else:
                # Unanswerable questions have no relevant documents. Their
                # retrieval scores are undefined for this answerable-only report.
                row["metrics"] = None
                row["lexical_evidence_metrics"] = None
                row["lexical_evidence_hit_lower_bound"] = None
                row["lexical_evidence_hit_upper_bound"] = None
                row["unanswerable_positive_lexical_match"] = (
                    lexical_match_counts[case["id"]] > 0
                )
            cases.append(row)
            cutoff_cases.append(row)
        summaries[str(k)] = {
            "answerable_positive_score_cutoff_tie_count": sum(
                row["answerable"] and row["positive_score_cutoff_tie"]
                for row in cutoff_cases
            ),
            "answerable_mixed_relevance_cutoff_tie_count": sum(
                row["answerable"] and row["positive_score_cutoff_tie_mixed_relevance"]
                for row in cutoff_cases
            ),
            "unanswerable_positive_score_cutoff_tie_count": sum(
                not row["answerable"] and row["positive_score_cutoff_tie"]
                for row in cutoff_cases
            ),
            "lexical_evidence_hit_lower_bound": (
                sum(row["lexical_evidence_hit_lower_bound"] for row in cutoff_cases
                    if row["answerable"]) / len(rows) if rows else None
            ),
            "lexical_evidence_hit_upper_bound": (
                sum(row["lexical_evidence_hit_upper_bound"] for row in cutoff_cases
                    if row["answerable"]) / len(rows) if rows else None
            ),
            "answerable_hit_uncertain_due_to_positive_ties": sum(
                row["answerable"]
                and row["lexical_evidence_hit_lower_bound"] != row["lexical_evidence_hit_upper_bound"]
                for row in cutoff_cases
            ),
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
            # The full retrieval-quality vector excludes zero-score ID ties.
            # Retain the legacy metrics below for side-by-side auditing.
            **{
                "lexical_evidence_" + name: (
                    sum(row[name] for row in evidence_rows) / len(evidence_rows)
                    if evidence_rows else None
                )
                for name in ("precision_at_k", "ndcg_at_k", "average_precision_at_k")
            },
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
        "parameters": {
            "k1": 1.2, "b": 0.75,
            "tokenizer": ("ascii_alphanumeric" if tokenizer_mode == "ascii"
                          else "unicode_alphanumeric"),
        },
        "corpus_sha256": _fingerprint(docs),
        "questions_sha256": _fingerprint(questions),
        "corpus_size": len(docs),
        "corpus_lexical_equivalence": lexical_equivalence,
        "question_count": len(questions),
        "selected_tokenizer_diagnostics": {
            "questions_without_selected_tokens": sum(
                count == 0 for count in selected_query_token_counts.values()
            ),
            "tokenizer_mode": tokenizer_mode,
        },
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
    parser.add_argument(
        "--tokenizer", choices=("ascii", "unicode"), default="ascii",
        help="ASCII legacy baseline or Unicode-aware lexical tokenization",
    )
    args = parser.parse_args(argv)
    try:
        docs = load_benchmark_json(args.corpus)
        questions = load_benchmark_json(args.questions)
        report = evaluate(docs, questions, args.k, tokenizer_mode=args.tokenizer)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
