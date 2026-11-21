"""Paired tokenizer diagnostics must identify genuine per-query changes."""
import json

from evalforge.tokenizer_failure_comparison import compare, main


def fixture(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.tsv"
    corpus.write_text(
        json.dumps({"_id": "d1", "title": "", "text": "café café café"}) + "\n"
        + json.dumps({"_id": "d2", "title": "", "text": "plain words"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps({"_id": "q1", "text": "café"}) + "\n"
        + json.dumps({"_id": "q2", "text": "plain"}) + "\n"
        + json.dumps({"_id": "q3", "text": "unknown"}) + "\n",
        encoding="utf-8",
    )
    qrels.write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\nq3\td1\t1\n",
        encoding="utf-8",
    )
    return corpus, queries, qrels


def test_paired_results_are_consistent_and_deterministic(tmp_path):
    corpus, queries, qrels = fixture(tmp_path)
    report = compare(corpus, queries, qrels, k=1)
    assert report == compare(corpus, queries, qrels, k=1)
    assert report["labeled_queries"] == 3
    assert report["recovered_by_unicode"] + report["lost_by_unicode"] + report["failed_by_both"] + report["unchanged_successes"] == 3
    assert report["unicode_positive_hit_queries"] - report["ascii_positive_hit_queries"] == report["recovered_by_unicode"] - report["lost_by_unicode"]
    assert report["failed_by_both"] >= 1
    assert len(report["source"]["source_sha256"]["corpus"]) == 64


def test_cli_writes_once_and_preserves_inputs(tmp_path):
    corpus, queries, qrels = fixture(tmp_path)
    output = tmp_path / "comparison.json"
    args = ["--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels), "--output", str(output)]
    assert main(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["labeled_queries"] == 3
    assert main(args) == 2
    assert main(args[:-1] + [str(corpus)]) == 2
    assert corpus.read_text(encoding="utf-8").startswith('{"_id": "d1"')


def test_invalid_k_and_limit_rejected(tmp_path):
    corpus, queries, qrels = fixture(tmp_path)
    output = tmp_path / "comparison.json"
    args = ["--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels), "--output", str(output)]
    assert main(args + ["--k", "0"]) == 2
    assert main(args + ["--limit", "0"]) == 2
    assert not output.exists()
