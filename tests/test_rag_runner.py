"""Regression coverage for repeatable, API-free RAG evaluation sweeps."""

from types import SimpleNamespace

import evalforge.rag_runner as rag_runner


def test_repeated_sweeps_replace_results_and_invalidate_report(monkeypatch):
    class StubPipeline:
        def __init__(self, store, model):
            assert model == "claude"

        def query(self, question, top_k):
            return {
                "retrieved": [{"doc_id": "doc-1"}],
                "answer": "INSUFFICIENT_CONTEXT",
                "retrieval_ms": 1.0,
                "generation_ms": 2.0,
                "total_ms": 3.0,
            }

    monkeypatch.setattr(rag_runner, "RAGPipeline", StubPipeline)
    # Bypass __init__ to avoid external clients: the evaluation is deterministic.
    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.store = object()
    runner.model = "claude"
    runner.questions = [{
        "id": "case-1",
        "question": "What is not in the corpus?",
        "expected_answer": "INSUFFICIENT_CONTEXT",
        "answerable": False,
        "relevant_doc_ids": [],
    }]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.judge = object()
    runner.results = []
    runner.report = None

    first = runner.run(k_values=(1, 3))
    assert len(first) == 2
    first_report = runner.generate_report()
    assert first_report["total_evaluations"] == 2
    assert set(first_report["by_top_k"]) == {"1", "3"}

    second = runner.run(k_values=(5,))
    assert len(second) == 1
    assert second[0]["top_k"] == 5
    assert runner.report is None
    assert runner.generate_report()["total_evaluations"] == 1
    assert set(runner.report["by_top_k"]) == {"5"}
