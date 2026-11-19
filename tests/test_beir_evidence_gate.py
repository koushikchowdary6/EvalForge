"""Integration coverage for BEIR's conservative retrieval quality gate."""
import json

from evalforge.beir_benchmark import main


def _inputs(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.tsv"
    corpus.write_text(
        json.dumps({"_id": "d1", "title": "", "text": "alpha beta"}) + "\n"
        + json.dumps({"_id": "d2", "title": "", "text": "gamma delta"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps({"_id": "q1", "text": "alpha"}) + "\n",
        encoding="utf-8",
    )
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    return ["--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels)]


def test_gate_passes_and_writes_provenance(tmp_path, capsys):
    output = tmp_path / "report.json"
    args = _inputs(tmp_path)
    assert main(args + ["--k", "1", "--gate-k", "1",
                        "--min-evidence-hit-rate", "1", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluation"]["by_top_k"]["1"]["lexical_evidence_hit_lower_bound"] == 1
    assert set(report["source"]["source_sha256"]) == {"corpus", "queries", "qrels"}


def test_gate_fails_with_report_for_debugging(tmp_path, capsys):
    args = _inputs(tmp_path)
    # q1 has a perfect lexical match, so a zero threshold is a pass.
    # Make the query out-of-vocabulary to force a true evidence failure.
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(json.dumps({"_id": "q1", "text": "unseen"}) + "\n", encoding="utf-8")
    output = tmp_path / "failed-report.json"
    assert main(args + ["--k", "1", "--gate-k", "1",
                        "--min-evidence-hit-rate", "0.5", "--output", str(output)]) == 1
    assert "BEIR evidence gate failed" in capsys.readouterr().err
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["evaluation"]["by_top_k"]["1"]["lexical_evidence_hit_lower_bound"] == 0


def test_gate_rejects_invalid_options_before_reading_inputs(tmp_path, capsys):
    args = ["--corpus", str(tmp_path / "missing-corpus"),
            "--queries", str(tmp_path / "missing-queries"),
            "--qrels", str(tmp_path / "missing-qrels")]
    for options in (
        ["--min-evidence-hit-rate", "0.5"],
        ["--gate-k", "1"],
        ["--min-evidence-hit-rate", "nan", "--gate-k", "1"],
        ["--min-evidence-hit-rate", "1.1", "--gate-k", "1"],
        ["--min-evidence-hit-rate", "0.5", "--gate-k", "2"],
    ):
        assert main(args + options) == 2
        assert "BEIR benchmark rejected" in capsys.readouterr().err
