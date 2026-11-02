"""Paired BM25 tokenizer comparisons must disclose score-tie sensitivity."""
import pytest

from evalforge.compare_lexical_baselines import compare_tokenizers


def _doc(identifier, text):
    return {"doc_id": identifier, "title": "", "text": text}


def _question(identifier, relevant, answerable=True):
    return {
        "id": identifier,
        "question": "alpha",
        "answerable": answerable,
        "relevant_doc_ids": relevant,
    }


def test_equal_point_estimates_can_hide_opposite_tie_outcomes():
    # All four documents have identical positive scores. At k=2, either
    # tokenizer can return zero, one, or two of the relevant documents.
    docs = [_doc(name, "alpha") for name in ("a", "b", "c", "d")]
    report = compare_tokenizers(docs, [_question("tied", ["c", "d"])], (2, 4))
    at_two = report["by_top_k"]["2"]
    row = next(c for c in report["cases"] if c["top_k"] == 2)
    assert row["paired_recall_delta_unicode_minus_ascii"] == 0.0
    assert row["paired_recall_delta_lower_bound_unicode_minus_ascii"] == -1.0
    assert row["paired_recall_delta_upper_bound_unicode_minus_ascii"] == 1.0
    assert at_two["paired_mean_recall_delta_lower_bound_unicode_minus_ascii"] == -1.0
    assert at_two["paired_mean_recall_delta_upper_bound_unicode_minus_ascii"] == 1.0
    assert at_two["paired_recall_delta_tie_ambiguous_cases"] == 1
    at_four = report["by_top_k"]["4"]
    assert at_four["paired_mean_recall_delta_lower_bound_unicode_minus_ascii"] == 0.0
    assert at_four["paired_mean_recall_delta_upper_bound_unicode_minus_ascii"] == 0.0
    assert at_four["paired_recall_delta_tie_ambiguous_cases"] == 0


def test_mixed_answerable_and_unanswerable_denominator_and_null_bounds():
    docs = [_doc(name, "alpha") for name in ("a", "b", "c", "d")]
    questions = [_question("tied", ["c", "d"]),
                 _question("unanswerable", [], answerable=False)]
    report = compare_tokenizers(docs, questions, (2,))
    summary = report["by_top_k"]["2"]
    assert summary["answerable_count"] == 1
    assert summary["unanswerable_count"] == 1
    assert summary["paired_mean_recall_delta_lower_bound_unicode_minus_ascii"] == -1
    assert summary["paired_mean_recall_delta_upper_bound_unicode_minus_ascii"] == 1
    row = next(c for c in report["cases"] if c["case_id"] == "unanswerable")
    assert row["paired_recall_delta_lower_bound_unicode_minus_ascii"] is None
    assert row["paired_recall_delta_upper_bound_unicode_minus_ascii"] is None


def test_unanswerable_only_does_not_invent_delta_bounds():
    docs = [_doc("a", "alpha")]
    report = compare_tokenizers(docs, [_question("q", [], answerable=False)], (1,))
    summary = report["by_top_k"]["1"]
    assert summary["paired_mean_recall_delta_lower_bound_unicode_minus_ascii"] is None
    assert summary["paired_mean_recall_delta_upper_bound_unicode_minus_ascii"] is None
    assert summary["paired_recall_delta_tie_ambiguous_cases"] == 0


def test_reported_point_delta_lies_inside_tie_bounds():
    docs = [_doc("a", "alpha"), _doc("b", "alpha"),
            _doc("c", "alpha 日本語"), _doc("d", "日本語")]
    questions = [_question("q", ["a", "c"])]
    report = compare_tokenizers(docs, questions, (1, 2, 3))
    for row in report["cases"]:
        assert (row["paired_recall_delta_lower_bound_unicode_minus_ascii"]
                <= row["paired_recall_delta_unicode_minus_ascii"]
                <= row["paired_recall_delta_upper_bound_unicode_minus_ascii"])
