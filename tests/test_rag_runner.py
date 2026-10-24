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


def test_generation_errors_do_not_count_as_hallucinations(monkeypatch):
    """Provider outages must not distort answer-quality rates or invoke judges."""

    class StubPipeline:
        def __init__(self, store, model):
            assert model == "claude"

        def query(self, question, top_k):
            failure = question == "API failure"
            return {
                "retrieved": [{"doc_id": "doc-1"}],
                "answer": "[Generation error: service unavailable]" if failure
                          else "INSUFFICIENT_CONTEXT",
                "error": "service unavailable" if failure else None,
                "retrieval_ms": 1.0,
                "generation_ms": 2.0,
                "total_ms": 3.0,
            }

    class NoJudge:
        def grade_correctness(self, *args):
            raise AssertionError("judge must not grade generation errors")

        def grade_groundedness(self, *args):
            raise AssertionError("judge must not grade generation errors")

    monkeypatch.setattr(rag_runner, "RAGPipeline", StubPipeline)
    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.store = object()
    runner.model = "claude"
    runner.questions = [
        {
            "id": "failure",
            "question": "API failure",
            "expected_answer": "INSUFFICIENT_CONTEXT",
            "answerable": False,
            "relevant_doc_ids": ["doc-1"],
        },
        {
            "id": "abstention",
            "question": "Unknown fact",
            "expected_answer": "INSUFFICIENT_CONTEXT",
            "answerable": False,
            "relevant_doc_ids": ["doc-1"],
        },
    ]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.judge = NoJudge()
    runner.results = []
    runner.report = None

    results = runner.run(k_values=(3,))
    report = runner.generate_report()["by_top_k"]["3"]

    assert len(results) == 2
    assert results[0]["generation_error"] == "service unavailable"
    assert results[0]["abstention"]["outcome"] == "generation_error"
    assert results[0]["correctness"]["score"] is None
    assert results[0]["groundedness"]["score"] is None
    assert report["generation"]["errors"] == 1
    assert report["generation"]["successful"] == 1
    assert report["generation"]["error_rate"] == 0.5
    assert report["hallucination"]["unanswerable_questions"] == 1
    assert report["hallucination"]["hallucinations"] == 0
    assert report["hallucination"]["hallucination_rate"] == 0.0
    # Both cases are unanswerable, so no retrieval ground truth is scored.
    assert report["retrieval"]["evaluated_queries"] == 0
    assert report["retrieval"]["excluded_unanswerable"] == 2
    assert report["retrieval"]["hit_rate"] is None


def test_save_removes_stale_report_when_new_sweep_has_no_aggregate(tmp_path):
    """Detailed results may be fresh while a prior summary is obsolete."""
    import json

    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.output_dir = tmp_path
    runner.results = [{"case_id": "first", "top_k": 1}]
    runner.report = {"total_evaluations": 1}
    runner.save()

    report_path = tmp_path / "rag_report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["total_evaluations"] == 1

    # Simulate a second sweep that has not generated an aggregate report yet.
    runner.results = [{"case_id": "second", "top_k": 5}]
    runner.report = None
    runner.save()

    assert not report_path.exists()
    saved = json.loads((tmp_path / "rag_results.json").read_text(encoding="utf-8"))
    assert saved == [{"case_id": "second", "top_k": 5}]


def test_unanswerable_cases_do_not_inflate_retrieval_scores(monkeypatch):
    """Unanswerable queries have no retrieval target; report coverage separately."""

    class StubPipeline:
        def __init__(self, store, model):
            pass

        def query(self, question, top_k):
            return {
                "retrieved": [{"doc_id": "unrelated"}],
                "answer": "INSUFFICIENT_CONTEXT",
                "retrieval_ms": 1.0,
                "generation_ms": 2.0,
                "total_ms": 3.0,
            }

    monkeypatch.setattr(rag_runner, "RAGPipeline", StubPipeline)
    runner = rag_runner.RAGEvalRunner.__new__(rag_runner.RAGEvalRunner)
    runner.store = object()
    runner.model = "claude"
    runner.questions = [
        {
            "id": "answerable", "question": "Where is the evidence?",
            "expected_answer": "The relevant evidence", "answerable": True,
            "relevant_doc_ids": ["missing-evidence"],
        },
        {
            "id": "unanswerable", "question": "Unknown fact",
            "expected_answer": "INSUFFICIENT_CONTEXT", "answerable": False,
            "relevant_doc_ids": [],
        },
    ]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.judge = object()
    runner.results = []
    runner.report = None
    # Both queries abstain; only the answerable query contributes retrieval
    # metrics. Empty-ground-truth conventions must not boost hit/MRR/recall.
    runner.run(k_values=(1,))
    retrieval = runner.generate_report()["by_top_k"]["1"]["retrieval"]
    assert retrieval["evaluated_queries"] == 1
    assert retrieval["excluded_unanswerable"] == 1
    assert retrieval["precision_at_k"] == 0.0
    assert retrieval["recall_at_k"] == 0.0
    assert retrieval["mrr"] == 0.0
    assert retrieval["hit_rate"] == 0.0

    runner.questions = runner.questions[1:]
    runner.run(k_values=(1,))
    retrieval = runner.generate_report()["by_top_k"]["1"]["retrieval"]
    assert retrieval["evaluated_queries"] == 0
    assert retrieval["excluded_unanswerable"] == 1
    assert all(retrieval[key] is None for key in (
        "precision_at_k", "recall_at_k", "mrr", "hit_rate"
    ))
