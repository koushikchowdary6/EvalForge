"""Contract tests for importing externally obtained BEIR-style retrieval data."""
import json

import pytest

from evalforge.beir_import import convert, main


def fixture(tmp_path, corpus=None, queries=None, qrels=None):
    corpus = corpus if corpus is not None else [
        {"_id": "d2", "title": "Second", "text": "other topic"},
        {"_id": "d1", "title": "First", "text": "alpha retrieval"}]
    queries = queries if queries is not None else [
        {"_id": "q2", "text": "unknown query"},
        {"_id": "q1", "text": "alpha retrieval"}]
    qrels = qrels if qrels is not None else "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n"
    for name, rows in (("corpus.jsonl", corpus), ("queries.jsonl", queries)):
        (tmp_path / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "qrels.tsv").write_text(qrels, encoding="utf-8")
    return (tmp_path / "corpus.jsonl", tmp_path / "queries.jsonl",
            tmp_path / "qrels.tsv")


def test_conversion_does_not_infer_unanswerable(tmp_path):
    corpus, queries, qrels = fixture(tmp_path)
    docs, cases, report = convert(corpus, queries, qrels)
    assert [d["doc_id"] for d in docs] == ["d1", "d2"]
    assert cases == [{"id": "q1", "question": "alpha retrieval",
                      "answerable": True, "relevant_doc_ids": ["d1"]}]
    assert report["excluded_queries_without_positive_qrels"] == 1


def test_cli_writes_valid_files_and_refuses_overwrite(tmp_path):
    corpus, queries, qrels = fixture(tmp_path)
    out_c = tmp_path / "converted-corpus.json"
    out_q = tmp_path / "converted-questions.json"
    args = ["--corpus", str(corpus), "--queries", str(queries),
            "--qrels", str(qrels), "--output-corpus", str(out_c),
            "--output-questions", str(out_q)]
    assert main(args) == 0
    assert json.loads(out_q.read_text())[0]["id"] == "q1"
    assert main(args) == 2


@pytest.mark.parametrize("qrels", [
    "query-id\tcorpus-id\tscore\nq1\tmissing\t1\n",
    "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t2\n",
    "query-id\tcorpus-id\tscore\nq1\td1\tnan\n",
    "query-id\tcorpus-id\tscore\nq1\td1\t0\n",
])
def test_invalid_or_unlabeled_inputs_rejected(tmp_path, qrels):
    corpus, queries, qrel_path = fixture(tmp_path, qrels=qrels)
    with pytest.raises(ValueError):
        convert(corpus, queries, qrel_path)


def test_duplicate_query_ids_rejected(tmp_path):
    corpus, queries, qrels = fixture(
        tmp_path, queries=[{"_id": "q1", "text": "one"},
                           {"_id": "q1", "text": "two"}])
    with pytest.raises(ValueError, match="duplicate"):
        convert(corpus, queries, qrels)
