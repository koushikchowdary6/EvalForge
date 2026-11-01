"""Reproducible synthetic multilingual lexical retrieval sanity checks.

These are hand-authored lexical pairs, not an independently sampled model
benchmark, cross-lingual retrieval test, or evidence of deployment accuracy.
"""

import json
from pathlib import Path
import subprocess
import sys

from evalforge.offline_rag_demo import evaluate, load_benchmark_json


DATA = Path(__file__).resolve().parents[1] / "datasets"
CORPUS = DATA / "multilingual_lexical_corpus.json"
QUESTIONS = DATA / "multilingual_lexical_questions.json"


def test_fixture_exposes_ascii_blind_spots_without_id_tie_credit():
    docs = load_benchmark_json(CORPUS)
    questions = load_benchmark_json(QUESTIONS)
    ascii_report = evaluate(docs, questions, (1,), tokenizer_mode="ascii")
    unicode_report = evaluate(docs, questions, (1,), tokenizer_mode="unicode")

    assert ascii_report["corpus_sha256"] == unicode_report["corpus_sha256"]
    assert ascii_report["questions_sha256"] == unicode_report["questions_sha256"]
    assert ascii_report["question_count"] == 6
    assert ascii_report["selected_tokenizer_diagnostics"]["questions_without_selected_tokens"] == 4
    assert unicode_report["selected_tokenizer_diagnostics"]["questions_without_selected_tokens"] == 0

    ascii_rows = {case["case_id"]: case for case in ascii_report["cases"]}
    unicode_rows = {case["case_id"]: case for case in unicode_report["cases"]}
    assert ascii_rows["q-en"]["positive_score_retrieved_ids"] == ["d-en"]
    for case_id, doc_id in (
        ("q-hi", "d-hi"), ("q-jp", "d-jp"),
        ("q-ar", "d-ar"), ("q-fr", "d-fr"),
    ):
        # A favorable zero-score ID tie must never count as lexical evidence.
        assert ascii_rows[case_id]["positive_score_retrieved_ids"] == []
        assert ascii_rows[case_id]["lexical_evidence_metrics"]["hit_rate"] == 0
        assert unicode_rows[case_id]["positive_score_retrieved_ids"] == [doc_id]
        assert unicode_rows[case_id]["lexical_evidence_metrics"]["hit_rate"] == 1
    assert unicode_rows["q-en"]["positive_score_retrieved_ids"] == ["d-en"]
    for report in (ascii_report, unicode_report):
        negative = {row["case_id"]: row for row in report["cases"]}["q-unanswerable"]
        assert negative["answerable"] is False
        assert negative["lexical_positive_document_count"] == 0
        assert negative["lexical_evidence_metrics"] is None
        assert report["by_top_k"]["1"]["answerable_count"] == 5
        assert report["by_top_k"]["1"]["unanswerable_excluded"] == 1


def test_multilingual_fixture_cli_is_reproducible_and_identifies_tokenizer():
    command = [
        sys.executable, "-m", "evalforge.offline_rag_demo",
        "--corpus", str(CORPUS), "--questions", str(QUESTIONS),
        "--k", "1", "--tokenizer", "unicode",
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
    second = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["method"] == "lexical_bm25_retrieval_only"
    assert report["parameters"]["tokenizer"] == "unicode_alphanumeric"
    assert report["by_top_k"]["1"]["lexical_evidence_hit_rate"] == 1.0
