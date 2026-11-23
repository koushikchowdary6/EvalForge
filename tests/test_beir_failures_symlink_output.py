"""Regression coverage for BEIR failure report output-path safety."""
from pathlib import Path

from evalforge.beir_failures import main


def test_dangling_symlink_output_is_rejected_without_creating_target(tmp_path: Path):
    target = tmp_path / "should-not-be-created.json"
    link = tmp_path / "report.json"
    link.symlink_to(target)
    result = main([
        "--corpus", str(tmp_path / "corpus.jsonl"),
        "--queries", str(tmp_path / "queries.jsonl"),
        "--qrels", str(tmp_path / "qrels.tsv"),
        "--output", str(link),
    ])
    assert result == 2
    assert link.is_symlink()
    assert not target.exists()
