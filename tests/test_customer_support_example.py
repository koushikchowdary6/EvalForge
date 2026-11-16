"""Regression coverage for the public, user-supplied retrieval fixture.

This is a hand-authored lexical retrieval example, not a production or LLM
accuracy benchmark. It deliberately includes an unanswerable question.
"""
import json
from pathlib import Path

from evalforge.offline_rag_demo import evaluate, validate


FIXTURES = Path(__file__).resolve().parents[1] / "examples"


def test_customer_support_fixture_is_valid_and_reproducible():
    corpus = json.loads((FIXTURES / "customer_support_corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((FIXTURES / "customer_support_questions.json").read_text(encoding="utf-8"))
    validate(corpus, questions)
    assert len(corpus) == 4
    assert len(questions) == 5
    assert sum(not question["answerable"] for question in questions) == 1

    first = evaluate(corpus, questions, [1, 3], tokenizer_mode="ascii")
    second = evaluate(corpus, questions, [1, 3], tokenizer_mode="ascii")
    assert first == second
    assert first["corpus_size"] == 4
    assert first["question_count"] == 5
    assert len(first["corpus_sha256"]) == 64
    assert len(first["questions_sha256"]) == 64
    assert "1" in first["by_top_k"]
    assert "3" in first["by_top_k"]
