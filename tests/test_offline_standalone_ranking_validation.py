"""Standalone BM25 API must uphold the same corpus integrity as evaluation."""

import pytest

from evalforge.offline_rag_demo import bm25_rank, bm25_rank_with_scores


DOCS = [
    {"doc_id": "battery", "title": "Battery", "text": "voltage sensor"},
    {"doc_id": "security", "title": "Security", "text": "access policy"},
]


@pytest.mark.parametrize("invalid_docs", [
    [],
    {},
    [{"doc_id": "battery", "title": "", "text": ""}],
    [{"doc_id": "battery", "title": "Battery", "text": 123}],
    DOCS + [dict(DOCS[0])],
])
def test_standalone_ranking_rejects_invalid_or_ambiguous_corpora(invalid_docs):
    with pytest.raises(ValueError):
        bm25_rank_with_scores("battery", invalid_docs)


@pytest.mark.parametrize("invalid_question", ["", "  ", None, 42])
def test_standalone_ranking_rejects_invalid_questions(invalid_question):
    with pytest.raises(ValueError, match="invalid benchmark question"):
        bm25_rank(invalid_question, DOCS)


def test_valid_standalone_scores_remain_deterministic():
    first = bm25_rank_with_scores("battery voltage", DOCS)
    assert first == bm25_rank_with_scores("battery voltage", DOCS)
    assert [doc_id for doc_id, _ in first] == bm25_rank("battery voltage", DOCS)
    assert first[0][0] == "battery"
