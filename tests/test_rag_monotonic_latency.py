"""RAG latency metrics must use elapsed time, not a wall clock that can jump."""

from types import SimpleNamespace

import pytest

from evalforge import rag
from evalforge.rag import RAGPipeline


def _gpt_pipeline(search):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model = "gpt"
    pipeline.store = SimpleNamespace(search=search)
    response = SimpleNamespace(choices=[
        SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="Grounded answer"),
        )
    ])
    pipeline.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    pipeline.anthropic_client = None
    return pipeline


def test_rag_query_reports_monotonic_stage_durations(monkeypatch):
    ticks = iter([100.0, 100.02, 100.02, 100.07])
    monkeypatch.setattr(rag.time, "perf_counter", lambda: next(ticks))
    pipeline = _gpt_pipeline(
        lambda question, top_k: [
            {"doc_id": "d1", "title": "Evidence", "text": "Source text"}
        ]
    )

    result = pipeline.query("Question?", top_k=1)
    assert result["answer"] == "Grounded answer"
    assert result["error"] is None
    assert result["retrieval_ms"] == pytest.approx(20.0)
    assert result["generation_ms"] == pytest.approx(50.0)
    assert result["total_ms"] == pytest.approx(70.0)


def test_rag_retrieval_failure_uses_monotonic_elapsed_time(monkeypatch):
    ticks = iter([20.0, 20.04])
    monkeypatch.setattr(rag.time, "perf_counter", lambda: next(ticks))

    def broken_search(question, top_k):
        raise RuntimeError("upstream unavailable")

    pipeline = _gpt_pipeline(broken_search)
    result = pipeline.query("Question?", top_k=1)
    assert result["error_stage"] == "retrieval"
    assert result["generation_ms"] == 0.0
    assert result["retrieval_ms"] == pytest.approx(40.0)
    assert result["total_ms"] == pytest.approx(40.0)
