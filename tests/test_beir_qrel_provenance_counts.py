"""BEIR provenance must account for positive and nonpositive judgments."""

import json

from evalforge.beir_import import convert


def test_qrel_counts_include_excluded_nonpositive_judgments(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.tsv"
    corpus.write_text(
        json.dumps({"_id": "d1", "title": "First", "text": "alpha"}) + "\n"
        + json.dumps({"_id": "d2", "title": "Second", "text": "beta"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps({"_id": "q1", "text": "alpha"}) + "\n"
        + json.dumps({"_id": "q2", "text": "beta"}) + "\n",
        encoding="utf-8",
    )
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t2\n"
        "q1\td2\t0\n"
        "q2\td1\t-1\n",
        encoding="utf-8",
    )
    documents, questions, provenance = convert(corpus, queries, qrels)
    assert len(documents) == 2
    assert len(questions) == 1
    assert provenance["positive_qrel_pairs"] == 1
    assert provenance["excluded_nonpositive_qrel_pairs"] == 2
    assert provenance["excluded_queries_without_positive_qrels"] == 1
    assert provenance["positive_qrel_pairs"] + provenance["excluded_nonpositive_qrel_pairs"] == 3
