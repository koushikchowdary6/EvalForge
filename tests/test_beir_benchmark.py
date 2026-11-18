"""End-to-end contract for direct BEIR evaluation on local synthetic fixtures."""
import json

from evalforge.beir_benchmark import benchmark, main


def sources(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.tsv"
    corpus.write_text(
        json.dumps({"_id": "d1", "title": "Alpha", "text": "alpha retrieval"}) + "\n"
        + json.dumps({"_id": "d2", "title": "Beta", "text": "other topic"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps({"_id": "q1", "text": "alpha retrieval"}) + "\n"
        + json.dumps({"_id": "q2", "text": "no labels"}) + "\n",
        encoding="utf-8",
    )
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    return corpus, queries, qrels


def test_direct_benchmark_reports_provenance_and_metrics(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    report = benchmark(corpus, queries, qrels, cutoffs=(1,))
    assert report["schema_version"] == 1
    assert report["source"]["labeled_queries"] == 1
    assert report["source"]["excluded_queries_without_positive_qrels"] == 1
    assert len(report["source"]["source_sha256"]["corpus"]) == 64
    assert report["evaluation"]["method"] == "lexical_bm25_retrieval_only"
    assert report == benchmark(corpus, queries, qrels, cutoffs=(1,))


def test_cli_writes_json_and_refuses_overwrite(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    out = tmp_path / "report.json"
    args = ["--corpus", str(corpus), "--queries", str(queries),
            "--qrels", str(qrels), "--k", "1", "--output", str(out)]
    assert main(args) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["source"]["labeled_queries"] == 1
    assert main(args) == 2
    assert main(args[:-1] + [str(corpus)]) == 2


def test_invalid_cutoff_does_not_create_report(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    out = tmp_path / "report.json"
    assert main(["--corpus", str(corpus), "--queries", str(queries),
                 "--qrels", str(qrels), "--k", "0", "--output", str(out)]) == 2
    assert not out.exists()

def test_failed_report_write_removes_only_new_partial_file(tmp_path, monkeypatch):
    """A write failure must not leave misleading, truncated benchmark evidence."""
    from pathlib import Path
    from unittest.mock import patch

    corpus, queries, qrels = sources(tmp_path)
    out = tmp_path / "partial.json"
    args = ["--corpus", str(corpus), "--queries", str(queries),
            "--qrels", str(qrels), "--output", str(out)]
    original_open = Path.open

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, content):
            self.stream.write(content[:12])
            raise OSError("simulated disk full")

    def failing_open(self, mode="r", *args, **kwargs):
        stream = original_open(self, mode, *args, **kwargs)
        if self == out and mode == "x":
            return FailingWriter(stream)
        return stream

    monkeypatch.setattr(Path, "open", failing_open)
    assert main(args) == 2
    assert not out.exists()
    # Exclusive creation must never modify or remove an existing report.
    out.write_text("existing evidence", encoding="utf-8")
    assert main(args) == 2
    assert out.read_text(encoding="utf-8") == "existing evidence"
