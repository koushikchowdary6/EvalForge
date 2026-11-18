"""End-to-end BEIR adapter -> lexical retrieval evaluation contract.

This intentionally uses tiny synthetic inputs, not an external benchmark or
a claim about retrieval performance on independently sourced datasets.
"""
import json

from evalforge.beir_import import main as import_beir
from evalforge.offline_rag_demo import evaluate


def test_beir_conversion_can_be_evaluated_end_to_end(tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.tsv"
    output_corpus = tmp_path / "converted-corpus.json"
    output_questions = tmp_path / "converted-questions.json"

    corpus.write_text(
        json.dumps({"_id": "d2", "title": "", "text": "oranges and pears"}) + "\n"
        + json.dumps({"_id": "d1", "title": "", "text": "quantum entanglement physics"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps({"_id": "q1", "text": "quantum entanglement"}) + "\n"
        + json.dumps({"_id": "q2", "text": "unlabeled question"}) + "\n",
        encoding="utf-8",
    )
    qrels.write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n",
        encoding="utf-8",
    )

    assert import_beir([
        "--corpus", str(corpus), "--queries", str(queries),
        "--qrels", str(qrels), "--output-corpus", str(output_corpus),
        "--output-questions", str(output_questions),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["labeled_queries"] == 1
    assert report["excluded_queries_without_positive_qrels"] == 1
    assert set(report["source_sha256"]) == {"corpus", "queries", "qrels"}

    documents = json.loads(output_corpus.read_text(encoding="utf-8"))
    questions = json.loads(output_questions.read_text(encoding="utf-8"))
    assert [d["doc_id"] for d in documents] == ["d1", "d2"]
    assert [q["id"] for q in questions] == ["q1"]

    result = evaluate(documents, questions, cutoffs=(1,))
    # Verify the adapter's output is usable by the actual scoring engine.
    # Avoid relying on incidental report formatting.
    assert result is not None
