"""All reported retrieval metrics must honor the requested top-k boundary."""

from types import SimpleNamespace

import pytest

import evalforge.rag_runner as rag_runner


def test_overreturning_retriever_is_excluded_from_retrieval_metrics(monkeypatch):
    """A faulty adapter must not turn extra evidence into a valid measurement."""

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
        assert row["retrieval_error"] == "retriever returned more than top_k"
        assert row["generation_error"] is None
        assert row["abstention"]["outcome"] == "generation_error"
        assert all(value is None for value in row["retrieval"].values())

    # The same ranking is legitimate at k=3 and retains its actual rank.
    assert rows[2]["retrieval_error"] is None
    assert rows[2]["retrieval"]["reciprocal_rank"] == pytest.approx(1 / 3)
    assert rows[2]["retrieval"]["hit_rate"] == 1.0

    report = runner.generate_report()["by_top_k"]
    for k in ("1", "2"):
        assert report[k]["retrieval"]["errors"] == 1
        assert report[k]["retrieval"]["evaluated_queries"] == 0
        assert report[k]["retrieval"]["mrr"] is None
        assert report[k]["generation"]["attempted"] == 0
    assert report["3"]["retrieval"]["mrr"] == pytest.approx(1 / 3)


def test_overreturned_evidence_never_reaches_paid_judge(monkeypatch):
    class OverReturningPipeline:
        def __init__(self, store, model):
            pass

        def query(self, question, top_k):
            return {
                "retrieved": [{"doc_id": "a"}, {"doc_id": "b"}],
                "answer": "An attempted factual answer",
                "retrieval_ms": 1.0,
                "generation_ms": 1.0,
                "total_ms": 2.0,
            }

    class NoJudge:
        def grade_correctness(self, *args):
            raise AssertionError("invalid top-k evidence must not be judged")

        def grade_groundedness(self, *args):
            raise AssertionError("invalid top-k evidence must not be judged")

    monkeypatch.setattr(rag_runner, "RAGPipeline", OverReturningPipeline)
    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.store = object()
    runner.model = "claude"
    runner.questions = [{
        "id": "q", "question": "Find evidence", "expected_answer": "Expected",
        "answerable": True, "relevant_doc_ids": ["b"],
    }]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.judge = NoJudge()
    runner.results = []
    runner.report = None

    row = runner.run(k_values=(1,))[0]
    assert row["retrieval_error"] == "retriever returned more than top_k"
    assert row["correctness"]["score"] is None
    assert row["groundedness"]["score"] is None
    assert runner.generate_report()["by_top_k"]["1"]["retrieval"]["errors"] == 1
