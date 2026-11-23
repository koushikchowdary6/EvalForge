"""Preflight must work without creating or requiring an output directory."""
import hashlib
import json

import pytest

from evalforge.scifact_runner import main


def test_preflight_without_output_dir(tmp_path, capsys):
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    payloads = {
        "corpus.jsonl": b'{"_id":"d1","title":"","text":"example"}\n',
        "queries.jsonl": b'{"_id":"q1","text":"example"}\n',
        "qrels/test.tsv": b"query-id\tcorpus-id\tscore\nq1\td1\t1\n",
    }
    for name, data in payloads.items():
        (dataset / name).write_bytes(data)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    result = main([
        "--dataset", str(dataset),
        "--evalforge-revision", "test-revision",
        "--source-url", "https://example.org/test-fixture",
        "--download-date", "2026-11-22",
        "--preflight-only",
    ])
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "inputs_fingerprinted_only"
    assert report["source_sha256"] == {
        "corpus": hashlib.sha256(payloads["corpus.jsonl"]).hexdigest(),
        "queries": hashlib.sha256(payloads["queries.jsonl"]).hexdigest(),
        "qrels": hashlib.sha256(payloads["qrels/test.tsv"]).hexdigest(),
    }
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_full_run_still_requires_output_dir(tmp_path):
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / name).write_text("fixture", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main([
            "--dataset", str(dataset),
            "--evalforge-revision", "test-revision",
            "--source-url", "https://example.org/test-fixture",
            "--download-date", "2026-11-22",
        ])
    assert error.value.code == 2
