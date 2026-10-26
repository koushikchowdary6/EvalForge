"""API-free tests for duplicate-aware average precision and RAG MAP@k reporting."""

from types import SimpleNamespace

import pytest

import evalforge.rag_runner as rag_runner
from evalforge.rag_graders import average_precision_at_k


def test_average_precision_rewards_early_relevant_documents():
    assert average_precision_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)
    assert average_precision_at_k(["x", "a", "b"], ["a", "b"], k=3) == pytest.approx(7 / 12)
    assert average_precision_at_k(["a", "x", "b"], ["a", "b"], k=3) == pytest.approx(5 / 6)


def test_duplicate_documents_consume_rank_slots_without_double_credit():
    assert average_precision_at_k(["a", "a", "b"], ["a", "b"], k=3) == pytest.approx(5 / 6)
    assert average_precision_at_k(["a", "a"], ["a", "b"], k=2) == pytest.approx(0.5)


def test_missing_retrievals_and_cutoff_reduce_average_precision():
    assert average_precision_at_k(["a"], ["a", "b"], k=5) == pytest.approx(0.5)
    assert average_precision_at_k(["x", "a"], ["a"], k=1) == 0.0
    assert average_precision_at_k(["x", "a"], ["a"], k=2) == pytest.approx(0.5)
    assert average_precision_at_k([], ["a"], k=5) == 0.0
    assert average_precision_at_k(["a"], ["a"]) == pytest.approx(1.0)
    assert average_precision_at_k(["x"], [], k=5) == 1.0


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "3"])
def test_invalid_average_precision_cutoffs_rejected(invalid):
    with pytest.raises(ValueError, match="k must be"):
        average_precision_at_k(["a"], ["a"], k=invalid)


def test_map_flows_into_sweep_and_report_without_api_calls(monkeypatch):
    class StubPipeline:
        def __init__(self, store, model):
            pass

        def query(self, question, top_k):
            return {
                "retrieved": [{"doc_id": doc_id} for doc_id in ["x", "a", "b"][:top_k]],
                "answer": "INSUFFICIENT_CONTEXT",
                "retrieval_ms": 1.0, "generation_ms": 1.0, "total_ms": 2.0,
            }

    monkeypatch.setattr(rag_runner, "RAGPipeline", StubPipeline)
    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.store = object()
    runner.model = "offline-test"
    runner.questions = [{
        "id": "ranked", "question": "Q?", "expected_answer": "Known",
        "answerable": True, "relevant_doc_ids": ["a", "b"],
    }]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.judge = object()
    runner.results = []
    runner.report = None

    rows = runner.run(k_values=(1, 3))
    assert rows[0]["retrieval"]["average_precision_at_k"] == 0.0
    assert rows[1]["retrieval"]["average_precision_at_k"] == pytest.approx(7 / 12)
    report = runner.generate_report()["by_top_k"]
    assert report["1"]["retrieval"]["map_at_k"] == 0.0
    assert report["3"]["retrieval"]["map_at_k"] == pytest.approx(7 / 12)
    runner.print_report()
