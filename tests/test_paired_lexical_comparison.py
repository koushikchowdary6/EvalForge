"""Paired lexical evaluation must show regressions as well as gains."""
import json
import subprocess
import sys

import pytest

from evalforge.compare_lexical_baselines import compare_tokenizers


DOCS = [
    {"doc_id": "a", "title": "", "text": "alpha"},
    {"doc_id": "b", "title": "", "text": "alpha 日本語"},
    {"doc_id": "z", "title": "", "text": "日本語"},
]
QUESTIONS = [
    {"id": "unicode-win", "question": "日本語", "answerable": True,
     "relevant_doc_ids": ["z"]},
    {"id": "ascii-win", "question": "alpha 日本語", "answerable": True,
     "relevant_doc_ids": ["a"]},
    {"id": "unanswerable", "question": "日本語", "answerable": False,
     "relevant_doc_ids": []},
]


def test_paired_audit_reports_gains_and_regressions_on_same_labels():
    report = compare_tokenizers(DOCS, QUESTIONS, cutoffs=(1, 2))
    one = report["by_top_k"]["1"]
    assert one["answerable_count"] == 2
    assert (one["ascii_only_hit"], one["unicode_only_hit"]) == (1, 1)
    assert (one["both_hit"], one["neither_hit"]) == (0, 0)
    assert one["paired_hit_rate_delta_unicode_minus_ascii"] == 0
    assert one["unanswerable_count"] == 1
    assert one["ascii_unanswerable_positive_lexical_match_count"] == 0
    assert one["unicode_unanswerable_positive_lexical_match_count"] == 1

    rows = {row["case_id"]: row for row in report["cases"] if row["top_k"] == 1}
    assert rows["unicode-win"]["paired_recall_delta_unicode_minus_ascii"] == 1
    assert rows["ascii-win"]["paired_recall_delta_unicode_minus_ascii"] == -1
    assert rows["unanswerable"]["ascii_evidence_hit"] is None
    assert rows["unanswerable"]["paired_recall_delta_unicode_minus_ascii"] is None

    two = report["by_top_k"]["2"]
    assert two["both_hit"] == 1
    assert two["unicode_only_hit"] == 1
    assert two["ascii_only_hit"] == 0
    assert report == compare_tokenizers(DOCS, QUESTIONS, cutoffs=(1, 2))


def test_unanswerable_only_does_not_fabricate_retrieval_quality():
    report = compare_tokenizers(DOCS, [QUESTIONS[-1]], cutoffs=(1,))
    summary = report["by_top_k"]["1"]
    assert summary["answerable_count"] == 0
    for key in (
        "ascii_evidence_hit_rate", "unicode_evidence_hit_rate",
        "paired_hit_rate_delta_unicode_minus_ascii",
        "ascii_mean_evidence_recall", "unicode_mean_evidence_recall",
        "paired_mean_recall_delta_unicode_minus_ascii",
    ):
        assert summary[key] is None
    assert summary["unicode_unanswerable_positive_lexical_match_count"] == 1


@pytest.mark.parametrize("bad_cutoffs", [(), (0,), (1, 1), (True,)])
def test_invalid_cutoffs_follow_shared_benchmark_contract(bad_cutoffs):
    with pytest.raises(ValueError, match="cutoffs"):
        compare_tokenizers(DOCS, QUESTIONS, cutoffs=bad_cutoffs)


def test_cli_uses_same_bounded_loader_and_emits_reproducible_provenance():
    command = [
        sys.executable, "-m", "evalforge.compare_lexical_baselines", "--k", "1"
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=20
    )
    report = json.loads(result.stdout)
    assert report["method"] == "paired_ascii_unicode_bm25_lexical_retrieval_only"
    assert len(report["corpus_sha256"]) == 64
    assert len(report["questions_sha256"]) == 64
    assert report["cutoffs"] == [1]
    assert report["question_count"] == 6
    assert report["by_top_k"]["1"]["answerable_count"] == 5
