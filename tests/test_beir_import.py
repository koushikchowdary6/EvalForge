"""Contract tests for importing externally obtained BEIR-style retrieval data."""
import json
import hashlib

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
    assert report["source_sha256"] == {
        "corpus": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "queries": hashlib.sha256(queries.read_bytes()).hexdigest(),
        "qrels": hashlib.sha256(qrels.read_bytes()).hexdigest(),
    }


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


def test_second_output_write_failure_rolls_back_first(tmp_path, monkeypatch):
    """A write failure after preflight must not leave a half-converted pair."""
    from pathlib import Path

    corpus, queries, qrels = fixture(tmp_path)
    out_c = tmp_path / "converted-corpus.json"
    out_q = tmp_path / "converted-questions.json"
    original_open = Path.open

    def fail_second_output(self, *args, **kwargs):
        if self == out_q and args and args[0] == "x":
            raise OSError("simulated second output failure")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_second_output)
    args = ["--corpus", str(corpus), "--queries", str(queries),
            "--qrels", str(qrels), "--output-corpus", str(out_c),
            "--output-questions", str(out_q)]
    assert main(args) == 2
    assert not out_c.exists()
    assert not out_q.exists()


def test_source_fingerprint_changes_when_qrels_change(tmp_path):
    corpus, queries, qrels = fixture(tmp_path)
    before = convert(corpus, queries, qrels)[2]["source_sha256"]
    qrels.write_text(qrels.read_text(encoding="utf-8").replace("q1\td1\t1", "q1\td1\t2"), encoding="utf-8")
    after = convert(corpus, queries, qrels)[2]["source_sha256"]
    assert before["corpus"] == after["corpus"]
    assert before["queries"] == after["queries"]
    assert before["qrels"] != after["qrels"]


@pytest.mark.parametrize("bad_record", [
    '{"_id":"d1","text":"safe","text":"overridden"}',
    '{"_id":"d1","text":"safe","metadata":{"x":1,"x":2}}',
    '{"_id":"d1","text":"safe","weight":NaN}',
    '{"_id":"d1","text":"safe","weight":Infinity}',
])
def test_ambiguous_or_nonstandard_jsonl_is_rejected(tmp_path, bad_record):
    corpus, queries, qrels = fixture(tmp_path)
    corpus.write_text(bad_record + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"corpus\.jsonl:1: invalid JSON"):
        convert(corpus, queries, qrels)


def test_valid_jsonl_unicode_and_nested_metadata_remain_supported(tmp_path):
    corpus, queries, qrels = fixture(tmp_path)
    corpus.write_text(
        json.dumps({"_id": "d1", "title": "Café",
                    "text": "alpha retrieval", "metadata": {"lang": "fr"}})
        + "\n" + json.dumps({"_id": "d2", "title": "", "text": "other topic"})
        + "\n", encoding="utf-8")
    documents, questions, _ = convert(corpus, queries, qrels)
    assert documents[0]["title"] == "Café"
    assert questions[0]["relevant_doc_ids"] == ["d1"]


@pytest.mark.parametrize("qrels", [
    "query-id\tcorpus-id\tscore\tscore\nq1\td1\t1\t0\n",
    "query-id\tcorpus-id\tscore\nq1\td1\t1\textra\n",
    "query-id\tcorpus-id\tscore\nq1\td1\n",
    "query-id\tcorpus-id\tscore\nq1\t\t1\n",
    "query-id\tcorpus-id\tscore\nq1\td1\t \n",
])
def test_malformed_qrels_tsv_rejected_without_silent_label_changes(tmp_path, qrels):
    corpus, queries, qrel_path = fixture(tmp_path, qrels=qrels)
    with pytest.raises(ValueError, match="qrels"):
        convert(corpus, queries, qrel_path)


def test_qrels_column_reordering_remains_supported(tmp_path):
    corpus, queries, qrel_path = fixture(
        tmp_path, qrels="score\tcorpus-id\tquery-id\n1\td1\tq1\n")
    _, questions, _ = convert(corpus, queries, qrel_path)
    assert questions[0]["relevant_doc_ids"] == ["d1"]


@pytest.mark.parametrize("source", ["corpus", "queries", "qrels"])
def test_utf8_bom_from_export_is_accepted_without_changing_fingerprint(tmp_path, source):
    corpus, queries, qrels = fixture(tmp_path)
    paths = {"corpus": corpus, "queries": queries, "qrels": qrels}
    target = paths[source]
    original = target.read_bytes()
    target.write_bytes(b"\xef\xbb\xbf" + original)

    documents, questions, provenance = convert(corpus, queries, qrels)
    assert len(documents) == 2
    assert [case["id"] for case in questions] == ["q1"]
    assert provenance["source_sha256"][source] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_bom_inside_jsonl_is_not_silently_ignored(tmp_path):
    corpus, queries, qrels = fixture(tmp_path)
    corpus.write_bytes(corpus.read_bytes() + b"\xef\xbb\xbf" + b'{"_id":"d3","text":"extra"}\n')
    with pytest.raises(ValueError, match=r"corpus\.jsonl:3: invalid JSON"):
        convert(corpus, queries, qrels)
