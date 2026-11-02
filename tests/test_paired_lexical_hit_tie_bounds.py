"""Paired hit-rate claims must account for arbitrary positive BM25 tie orders.

These are deterministic worst/best-case bounds, not confidence intervals.
"""
from evalforge.compare_lexical_baselines import compare_tokenizers


def _doc(identifier, text):
    return {"doc_id": identifier, "title": "", "text": text}


def _question(identifier, relevant, *, answerable=True, text="alpha"):
    return {
        "id": identifier,
        "question": text,
        "answerable": answerable,
        "relevant_doc_ids": relevant,
    }


def test_equal_hit_point_estimates_can_hide_opposite_tie_outcomes():
    # At k=2, both tokenizers report a miss under ID tie-breaking, but
    # each could independently hit either of the two relevant documents.
    docs = [_doc(name, "alpha") for name in ("a", "b", "c", "d")]
    report = compare_tokenizers(docs, [_question("tied", ["c", "d"])], (2, 4))
    two = report["by_top_k"]["2"]
    row = next(item for item in report["cases"] if item["top_k"] == 2)
    assert two["paired_hit_rate_delta_unicode_minus_ascii"] == 0
    assert two["paired_hit_rate_delta_lower_bound_unicode_minus_ascii"] == -1
    assert two["paired_hit_rate_delta_upper_bound_unicode_minus_ascii"] == 1
    assert two["paired_hit_delta_tie_ambiguous_cases"] == 1
    assert row["paired_hit_delta_lower_bound_unicode_minus_ascii"] == -1
    assert row["paired_hit_delta_upper_bound_unicode_minus_ascii"] == 1

    four = report["by_top_k"]["4"]
    assert four["paired_hit_rate_delta_lower_bound_unicode_minus_ascii"] == 0
    assert four["paired_hit_rate_delta_upper_bound_unicode_minus_ascii"] == 0
    assert four["paired_hit_delta_tie_ambiguous_cases"] == 0


def test_definite_unicode_lexical_gain_has_no_hit_tie_ambiguity():
    docs = [_doc("japanese", "日本語"), _doc("latin", "alpha")]
    report = compare_tokenizers(
        docs, [_question("unicode", ["japanese"], text="日本語")], (1,)
    )
    summary = report["by_top_k"]["1"]
    assert summary["paired_hit_rate_delta_unicode_minus_ascii"] == 1
    assert summary["paired_hit_rate_delta_lower_bound_unicode_minus_ascii"] == 1
    assert summary["paired_hit_rate_delta_upper_bound_unicode_minus_ascii"] == 1
    assert summary["paired_hit_delta_tie_ambiguous_cases"] == 0


def test_unanswerable_cases_do_not_affect_hit_bounds_denominator():
    docs = [_doc(name, "alpha") for name in ("a", "b", "c", "d")]
    report = compare_tokenizers(
        docs,
        [
            _question("tied", ["c", "d"]),
            _question("negative", [], answerable=False),
        ],
        (2,),
    )
    summary = report["by_top_k"]["2"]
    assert summary["answerable_count"] == 1
    assert summary["unanswerable_count"] == 1
    assert summary["paired_hit_rate_delta_lower_bound_unicode_minus_ascii"] == -1
    assert summary["paired_hit_rate_delta_upper_bound_unicode_minus_ascii"] == 1
    negative = next(row for row in report["cases"] if row["case_id"] == "negative")
    assert negative["paired_hit_delta_lower_bound_unicode_minus_ascii"] is None
    assert negative["paired_hit_delta_upper_bound_unicode_minus_ascii"] is None


def test_unanswerable_only_does_not_invent_hit_rate_bounds():
    report = compare_tokenizers(
        [_doc("a", "alpha")],
        [_question("negative", [], answerable=False)],
        (1,),
    )
    summary = report["by_top_k"]["1"]
    assert summary["paired_hit_rate_delta_lower_bound_unicode_minus_ascii"] is None
    assert summary["paired_hit_rate_delta_upper_bound_unicode_minus_ascii"] is None
    assert summary["paired_hit_delta_tie_ambiguous_cases"] == 0
