"""Schema-aware SciFact preflight validates references without writing artifacts."""
import json

from evalforge.scifact_runner import main


def _fixture(tmp_path, *, invalid_qrel=False):
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    (dataset / "corpus.jsonl").write_text('{"_id":"d1","title":"","text":"example"}\n', encoding="utf-8")
    (dataset / "queries.jsonl").write_text('{"_id":"q1","text":"example"}\n', encoding="utf-8")
    doc_id = "missing" if invalid_qrel else "d1"
    (dataset / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\t" + doc_id + "\t1\n", encoding="utf-8"
    )
    return dataset


def _args(dataset):
    return [
        "--dataset", str(dataset),
        "--evalforge-revision", "test",
        "--source-url", "https://example.org/fixture",
        "--download-date", "2026-11-22",
        "--preflight-only", "--preflight-validate-schema",
    ]


def test_schema_preflight_reports_counts_without_writing(tmp_path, capsys):
    dataset = _fixture(tmp_path)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert main(_args(dataset)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "inputs_schema_validated"
    assert report["schema_validation"]["corpus_documents"] == 1
    assert report["schema_validation"]["labeled_queries"] == 1
    assert report["schema_validation"]["positive_qrel_pairs"] == 1
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_schema_preflight_rejects_dangling_qrel(tmp_path, capsys):
    assert main(_args(_fixture(tmp_path, invalid_qrel=True))) == 2
    assert "unknown query or document" in capsys.readouterr().err


def test_schema_validation_requires_preflight(tmp_path):
    import pytest
    dataset = _fixture(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main([arg for arg in _args(dataset) if arg != "--preflight-only"])
    assert exc.value.code == 2
