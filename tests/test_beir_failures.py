"""Behavioral coverage for query-level BEIR retrieval failure diagnostics."""
import json

from evalforge.beir_failures import analyze, main


def sources(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.tsv"
    corpus.write_text(
        json.dumps({"_id": "d1", "title": "apples", "text": "fruit orchard"}) + "\n"
        + json.dumps({"_id": "d2", "title": "servers", "text": "cloud compute"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps({"_id": "q1", "text": "apples"}) + "\n"
        + json.dumps({"_id": "q2", "text": "cloud"}) + "\n"
        + json.dumps({"_id": "q3", "text": "unseenword"}) + "\n",
        encoding="utf-8",
    )
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t1\n"
        "q2\td1\t1\n"
        "q3\td2\t1\n",
        encoding="utf-8",
    )
    return corpus, queries, qrels


def test_failure_reasons_and_no_fabricated_hits(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    report = analyze(corpus, queries, qrels, k=1)
    assert report["labeled_queries"] == 3
    assert report["queries_with_positive_relevant_hit"] == 1
    assert report["queries_without_positive_relevant_hit"] == 2
    failures = {item["query_id"]: item for item in report["shown_failures"]}
    assert failures["q2"]["reason"] == "no_positive_relevant_overlap"
    assert failures["q2"]["top_k_positive_results"][0]["doc_id"] == "d2"
    assert failures["q3"]["reason"] == "all_query_terms_oov"
    assert failures["q3"]["top_k_positive_results"] == []
    assert report["failure_reason_counts"] == {
        "no_query_tokens": 0,
        "all_query_terms_oov": 1,
        "no_positive_relevant_overlap": 1,
        "relevant_docs_outside_top_k": 0,
    }
    assert failures["q2"]["first_positive_relevant_rank"] is None
    assert failures["q3"]["first_positive_relevant_score"] is None
    assert report == analyze(corpus, queries, qrels, k=1)


def test_limit_preserves_total_and_provenance(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    report = analyze(corpus, queries, qrels, k=1, limit=1)
    assert len(report["shown_failures"]) == 1
    assert report["truncated"] is True
    assert len(report["source"]["source_sha256"]["corpus"]) == 64


def test_installed_cli_output_and_refuses_overwrite(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    out = tmp_path / "failures.json"
    args = ["--corpus", str(corpus), "--queries", str(queries),
            "--qrels", str(qrels), "--k", "1", "--output", str(out)]
    assert main(args) == 0
    assert json.loads(out.read_text())["queries_without_positive_relevant_hit"] == 2
    assert main(args) == 2
    assert main(args[:-1] + [str(corpus)]) == 2
    assert corpus.read_text().startswith('{"_id": "d1"')


def test_invalid_options_do_not_write(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    out = tmp_path / "failures.json"
    args = ["--corpus", str(corpus), "--queries", str(queries),
            "--qrels", str(qrels), "--output", str(out)]
    assert main(args + ["--k", "0"]) == 2
    assert main(args + ["--limit", "0"]) == 2
    assert not out.exists()


def test_ranked_relevant_document_outside_cutoff(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.tsv"
    corpus.write_text(
        json.dumps({"_id": "d1", "title": "", "text": "alpha alpha alpha alpha"}) + "\n"
        + json.dumps({"_id": "d2", "title": "", "text": "alpha"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(json.dumps({"_id": "q1", "text": "alpha"}) + "\n", encoding="utf-8")
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td2\t1\n", encoding="utf-8")
    report = analyze(corpus, queries, qrels, k=1)
    assert report["queries_without_positive_relevant_hit"] == 1
    assert report["failure_reason_counts"]["relevant_docs_outside_top_k"] == 1
    failure = report["shown_failures"][0]
    assert failure["first_positive_relevant_rank"] == 2
    assert failure["first_positive_relevant_score"] > 0


def test_hit_rate_gate_reports_results_before_failing(tmp_path, capsys):
    corpus, queries, qrels = sources(tmp_path)
    base = ["--corpus", str(corpus), "--queries", str(queries),
            "--qrels", str(qrels), "--k", "1"]
    output = tmp_path / "gate.json"
    assert main(base + ["--min-hit-rate", "0.5", "--output", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["queries_with_positive_relevant_hit"] == 1
    assert report["labeled_queries"] == 3
    assert "gate failed" in capsys.readouterr().err
    assert main(base + ["--min-hit-rate", "0.3333333333333333"]) == 0
    assert main(base + ["--min-hit-rate", "0"]) == 0


def test_invalid_hit_rate_threshold_never_writes(tmp_path):
    corpus, queries, qrels = sources(tmp_path)
    base = ["--corpus", str(corpus), "--queries", str(queries),
            "--qrels", str(qrels)]
    for invalid in ("nan", "inf", "-inf", "-0.01", "1.01"):
        output = tmp_path / "invalid.json"
        assert main(base + ["--min-hit-rate", invalid, "--output", str(output)]) == 2
        assert not output.exists()
