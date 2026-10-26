"""Top-k precision must not reward underfilled retrieval lists as perfect."""

from types import SimpleNamespace

import pytest

import evalforge.rag_runner as rag_runner
from evalforge.rag_graders import precision_at_k


def test_short_retrieval_uses_requested_denominator():
    assert precision_at_k(["relevant"], ["relevant"], k=5) == pytest.approx(0.2)
    assert precision_at_k(["relevant"], ["relevant"]) == pytest.approx(1.0)
    assert precision_at_k([], ["relevant"], k=5) == 0.0
    assert precision_at_k(["relevant", "relevant"], ["relevant"], k=5) == pytest.approx(0.2)


def test_rag_sweep_propagates_requested_k_into_precision(monkeypatch):
    class StubPipeline:
        def __init__(self, store, model):
            pass

        def query(self, question, top_k):
            return {
                "retrieved": [{"doc_id": "relevant"}],
                "answer": "INSUFFICIENT_CONTEXT",
                "retrieval_ms": 1.0,
                "generation_ms": 1.0,
                "total_ms": 2.0,
            }

    monkeypatch.setattr(rag_runner, "RAGPipeline", StubPipeline)
    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.store = object()
    runner.model = "offline-test"
    runner.questions = [{
        "id": "q1", "question": "Q?", "expected_answer": "Expected",
        "answerable": True, "relevant_doc_ids": ["relevant"],
    }]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.judge = object()  # An abstention must not invoke the judge.
    runner.results = []
    runner.report = None

    rows = runner.run(k_values=(1, 5))
    assert [row["retrieval"]["precision_at_k"] for row in rows] == pytest.approx([1.0, 0.2])
    report = runner.generate_report()["by_top_k"]
    assert report["1"]["retrieval"]["precision_at_k"] == pytest.approx(1.0)
    assert report["5"]["retrieval"]["precision_at_k"] == pytest.approx(0.2)
    assert report["5"]["retrieval"]["recall_at_k"] == pytest.approx(1.0)
