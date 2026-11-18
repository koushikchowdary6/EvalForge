"""Convert local BEIR-style corpus/queries/qrels into EvalForge labeled retrieval JSON.

BEIR inputs are user-supplied; this tool neither downloads nor redistributes data.
Only positive qrels are relevant. Queries without positive labels are excluded
by default: an absent qrel is *not* evidence that a query is unanswerable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from .offline_rag_demo import validate


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            def unique_pairs(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        raise ValueError(f"duplicate JSON key: {key!r}")
                    obj[key] = value
                return obj

            def reject_constant(value):
                raise ValueError(f"nonstandard JSON constant: {value}")

            try:
                record = json.loads(
                    line, object_pairs_hook=unique_pairs,
                    parse_constant=reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_no}: expected JSON object")
            yield record


def _sha256(path: Path) -> str:
    """Fingerprint exact source bytes for independent benchmark provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(corpus_path: Path, queries_path: Path, qrels_path: Path):
    docs = {}
    for row in _jsonl(corpus_path):
        doc_id = row.get("_id")
        if not isinstance(doc_id, str) or doc_id in docs:
            raise ValueError("invalid or duplicate corpus _id")
        docs[doc_id] = {"doc_id": doc_id, "title": row.get("title", ""),
                        "text": row.get("text")}
    queries = {}
    for row in _jsonl(queries_path):
        query_id = row.get("_id")
        if not isinstance(query_id, str) or query_id in queries:
            raise ValueError("invalid or duplicate query _id")
        queries[query_id] = row.get("text")
    labels = {}
    with qrels_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames or []):
            raise ValueError("qrels must have query-id, corpus-id, score columns")
        seen = set()
        for row in reader:
            key = (row["query-id"], row["corpus-id"])
            if key in seen:
                raise ValueError("duplicate qrel pair")
            seen.add(key)
            if key[0] not in queries or key[1] not in docs:
                raise ValueError("qrel references unknown query or document")
            try:
                score = int(row["score"])
            except (ValueError, TypeError) as exc:
                raise ValueError("qrel score must be an integer") from exc
            if score > 0:
                labels.setdefault(key[0], set()).add(key[1])
    questions = [
        {"id": query_id, "question": queries[query_id], "answerable": True,
         "relevant_doc_ids": sorted(labels[query_id])}
        for query_id in sorted(labels)
    ]
    documents = [docs[doc_id] for doc_id in sorted(docs)]
    validate(documents, questions)
    return documents, questions, {
        "source_sha256": {
            "corpus": _sha256(corpus_path),
            "queries": _sha256(queries_path),
            "qrels": _sha256(qrels_path),
        },
        "corpus_documents": len(documents),
        "source_queries": len(queries),
        "labeled_queries": len(questions),
        "excluded_queries_without_positive_qrels": len(queries) - len(questions),
        "scope": "positive_qrels_only_no_unanswerable_inference",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--output-corpus", required=True, type=Path)
    parser.add_argument("--output-questions", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        docs, questions, report = convert(args.corpus, args.queries, args.qrels)
        outputs = (args.output_corpus, args.output_questions)
        inputs = (args.corpus, args.queries, args.qrels)
        if outputs[0].resolve() == outputs[1].resolve():
            raise ValueError("output paths must be distinct")
        if any(p.resolve() in {i.resolve() for i in inputs} for p in outputs):
            raise ValueError("output cannot overwrite input")
        if any(p.exists() for p in outputs):
            raise ValueError("output already exists; refusing to overwrite")
        # If the second write fails, do not leave the first output looking like
        # a complete conversion. Never remove files we did not create.
        created = []
        try:
            for path, data in zip(outputs, (docs, questions)):
                with path.open("x", encoding="utf-8") as stream:
                    created.append(path)
                    json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
        except (OSError, ValueError, TypeError, RecursionError):
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        print(json.dumps(report, sort_keys=True))
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(f"BEIR conversion rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
