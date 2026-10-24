"""Retrieval outages must not be counted as poor retrieval or generation quality."""

from evalforge.rag import RAGPipeline
from evalforge.rag_runner import RAGEvalRunner


def test_pipeline_does_not_generate_when_retrieval_fails():
    class BrokenStore:
        def search(self, question, top_k):
            raise RuntimeError("embedding provider unavailable: private credential")

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = BrokenStore()

    def forbidden_generate(*args):
        raise AssertionError("generation must not run without retrieved context")

    pipeline.generate = forbidden_generate
    result = pipeline.query("What happened?", top_k=3)
    assert result["error_stage"] == "retrieval"
    assert result["retrieved"] == []
    assert result["answer"] == ""
    assert result["generation_ms"] == 0
    assert "private credential" not in result["error"]


def test_retrieval_failure_is_excluded_from_quality_denominators(monkeypatch):
    class StubPipeline:
        def __init__(self, store, model):
            pass

        def query(self, question, top_k):
            return {
                "retrieved": [], "answer": "", "error": "retrieval failed (RuntimeError)",
                "error_stage": "retrieval", "retrieval_ms": 12.0,
                "generation_ms": 0.0, "total_ms": 12.0,
            }

    monkeypatch.setattr("evalforge.rag_runner.RAGPipeline", StubPipeline)
    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.store = object()
    runner.model = "claude"
    runner.questions = [{
        "id": "case-1", "question": "Example?", "expected_answer": "Yes",
        "answerable": True, "relevant_doc_ids": ["doc-1"],
    }]
    runner.results = []
    runner.report = None
    runner.embedder = type("Embedder", (), {"api_calls": 0, "cache_hits": 0})()
    runner.judge = type("Judge", (), {
        "grade_correctness": lambda *args: (_ for _ in ()).throw(AssertionError()),
        "grade_groundedness": lambda *args: (_ for _ in ()).throw(AssertionError()),
    })()

    results = runner.run(k_values=(1,))
    assert len(results) == 1
    assert results[0]["retrieval_error"] is not None
    assert results[0]["generation_error"] is None
    assert all(value is None for value in results[0]["retrieval"].values())
    report = runner.generate_report()["by_top_k"]["1"]
    assert report["retrieval"]["errors"] == 1
    assert report["retrieval"]["evaluated_queries"] == 0
    assert report["retrieval"]["precision_at_k"] is None
    assert report["generation"]["attempted"] == 0
    assert report["generation"]["errors"] == 0
    assert report["generation"]["error_rate"] is None
