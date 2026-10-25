"""Invalid retrieval cutoffs must never silently inflate precision or recall."""

import pytest

from evalforge.rag_graders import precision_at_k, recall_at_k


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
@pytest.mark.parametrize("cutoff", [-3, -1, 0, True, False, 1.5, "2", []])
def test_invalid_cutoff_is_rejected(metric, cutoff):
    with pytest.raises(ValueError, match="positive integer"):
        metric(["relevant", "other"], ["relevant"], k=cutoff)


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
def test_default_and_valid_cutoffs_remain_supported(metric):
    assert metric(["relevant", "other"], ["relevant"], k=1) == 1.0
    expected = 0.5 if metric is precision_at_k else 1.0
    assert metric(["relevant", "other"], ["relevant"], k=2) == expected
    assert metric(["relevant", "other"], ["relevant"], k=None) == metric(
        ["relevant", "other"], ["relevant"]
    )


def test_cutoff_limits_retrieval_to_first_k_documents():
    assert precision_at_k(["irrelevant", "relevant"], ["relevant"], k=1) == 0.0
    assert recall_at_k(["irrelevant", "relevant"], ["relevant"], k=1) == 0.0
