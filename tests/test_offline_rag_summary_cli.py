"""CLI regression coverage for compact offline benchmark reports."""
import json

from evalforge.offline_rag_demo import main


def test_summary_only_omits_cases_but_keeps_aggregate_provenance(capsys):
    main(["--k", "1", "3", "--summary-only"])
    summary = json.loads(capsys.readouterr().out)
    main(["--k", "1", "3"])
    full = json.loads(capsys.readouterr().out)

    assert "cases" not in summary
    assert full["cases"]
    assert summary == {key: value for key, value in full.items() if key != "cases"}
    assert summary["corpus_sha256"] == full["corpus_sha256"]
    assert summary["questions_sha256"] == full["questions_sha256"]
    assert summary["by_top_k"]


def test_summary_only_is_repeatable(capsys):
    main(["--k", "1", "--summary-only"])
    first = capsys.readouterr().out
    main(["--k", "1", "--summary-only"])
    assert capsys.readouterr().out == first
