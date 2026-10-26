"""Offline regression tests for rank-sensitive RAG retrieval evaluation."""

import math
from types import SimpleNamespace

import pytest

import evalforge.rag_runner as rag_runner
from evalforge.rag_graders import ndcg_at_k


def test_ndcg_rewards_early_relevant_evidence():
    assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], ["a"], k=2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["a", "x"], ["a"], k=2) > ndcg_at_k(
        ["x", "a"], ["a"], k=2
    )


def test_duplicates_consume_rank_but_never_earn_double_credit():
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "a", "b"], ["a", "b"], k=3) == pytest.approx(expected)
    assert ndcg_at_k(["a", "a"], ["a", "b"], k=2) < 1.0


def test_cutoff_and_missing_results_do_not_inflate_gain():
    assert ndcg_at_k(["x", "a"], ["a"], k=1) == 0.0
    assert ndcg_at_k(["x", "a"], ["a"], k=2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["a"], ["a", "b"], k=5) == pytest.approx(
        1 / (1 + 1 / math.log2(3))
    )
    assert ndcg_at_k([], ["a"], k=5) == 0.0
    assert ndcg_at_k(["a"], ["a"]) == 1.0


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "3"])
def test_invalid_cutoffs_are_rejected(invalid):
    with pytest.raises(ValueError, match="k must be"):
        ndcg_at_k(["a"], ["a"], k=invalid)


def test_unanswerable_empty_ground_truth_convention():
    assert ndcg_at_k(["irrelevant"], [], k=5) == 1.0


def test_ndcg_flows_through_rag_sweep_and_report(monkeypatch):
    class StubPipeline:
        def __init__(self, store, model):
            pass

        def query(self, question, top_k):
            return {
                "retrieved": [{"doc_id": doc_id} for doc_id in ["x", "a", "b"][:top_k]],
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
        "id": "ranked", "question": "Q?", "expected_answer": "Known",
        "answerable": True, "relevant_doc_ids": ["a", "b"],
    }]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.judge = object()  # The explicit abstention does not need an API judge.
    runner.results = []
    runner.report = None

    rows = runner.run(k_values=(1, 3))
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert rows[0]["retrieval"]["ndcg_at_k"] == 0.0
    assert rows[1]["retrieval"]["ndcg_at_k"] == pytest.approx(expected)
    report = runner.generate_report()["by_top_k"]
    assert report["1"]["retrieval"]["ndcg_at_k"] == 0.0
    assert report["3"]["retrieval"]["ndcg_at_k"] == pytest.approx(expected)
