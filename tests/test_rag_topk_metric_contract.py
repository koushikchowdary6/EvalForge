"""All reported retrieval metrics must honor the requested top-k boundary."""

from types import SimpleNamespace

import pytest

import evalforge.rag_runner as rag_runner


def test_overreturning_retriever_cannot_inflate_mrr_or_hit_rate(monkeypatch):
    """A faulty adapter may return more than k documents; do not credit extras."""

    class OverReturningPipeline:
        def __init__(self, store, model):
            assert model == "claude"

        def query(self, question, top_k):
            # Deliberately violate the top_k contract to test the scoring layer.
            return {
                "retrieved": [
                    {"doc_id": "irrelevant-1"},
                    {"doc_id": "irrelevant-2"},
                    {"doc_id": "relevant"},
                ],
                "answer": "INSUFFICIENT_CONTEXT",
                "retrieval_ms": 1.0,
                "generation_ms": 1.0,
                "total_ms": 2.0,
            }

    monkeypatch.setattr(rag_runner, "RAGPipeline", OverReturningPipeline)
    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.store = object()
    runner.model = "claude"
    runner.questions = [{
        "id": "case-1",
        "question": "Find the supporting document",
        "expected_answer": "The relevant evidence",
        "answerable": True,
        "relevant_doc_ids": ["relevant"],
    }]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.judge = object()  # The abstaining answer must not invoke a judge.
    runner.results = []
    runner.report = None

    rows = runner.run(k_values=(1, 2, 3))
    for row in rows[:2]:
        assert row["retrieval"]["reciprocal_rank"] == 0.0
        assert row["retrieval"]["hit_rate"] == 0.0
        assert row["retrieval"]["recall_at_k"] == 0.0

    assert rows[2]["retrieval"]["reciprocal_rank"] == pytest.approx(1 / 3)
    assert rows[2]["retrieval"]["hit_rate"] == 1.0

    report = runner.generate_report()["by_top_k"]
    assert report["1"]["retrieval"]["mrr"] == 0.0
    assert report["2"]["retrieval"]["hit_rate"] == 0.0
    assert report["3"]["retrieval"]["mrr"] == pytest.approx(1 / 3)
