"""Metamorphic regression tests for deterministic lexical retrieval.

Document ingestion order must not change ranking or relevance decisions.
This is a correctness invariant, not an independent accuracy benchmark.
"""
import json
from pathlib import Path

import pytest

from evalforge.offline_rag_demo import bm25_rank_with_scores, evaluate


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("tokenizer", ["ascii", "unicode"])
def test_bm25_ranking_is_independent_of_corpus_order(tokenizer):
    docs = json.loads((EXAMPLES / "customer_support_corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((EXAMPLES / "customer_support_questions.json").read_text(encoding="utf-8"))
    for question in questions:
        original = bm25_rank_with_scores(question["question"], docs, tokenizer_mode=tokenizer)
        reversed_order = bm25_rank_with_scores(
            question["question"], list(reversed(docs)), tokenizer_mode=tokenizer
        )
        assert original == reversed_order


@pytest.mark.parametrize("tokenizer", ["ascii", "unicode"])
def test_retrieval_metrics_are_independent_of_corpus_order(tokenizer):
    docs = json.loads((EXAMPLES / "customer_support_corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((EXAMPLES / "customer_support_questions.json").read_text(encoding="utf-8"))
    original = evaluate(docs, questions, [1, 3], tokenizer_mode=tokenizer)
    reversed_order = evaluate(list(reversed(docs)), questions, [1, 3], tokenizer_mode=tokenizer)
    assert original["by_top_k"] == reversed_order["by_top_k"]
