"""Untrusted RAG generation evidence must be bounded before provider calls."""

import math
from types import SimpleNamespace

import pytest

from evalforge.rag import RAGPipeline, MAX_RAG_GENERATION_EVIDENCE_CHARS


def _pipeline(provider):
    calls = []
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model = provider

    def record(**kwargs):
        calls.append(kwargs)
        if provider == "claude":
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="grounded answer")],
            )
        return SimpleNamespace(choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="grounded answer"),
            )
        ])

    pipeline.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(create=record)
    )
    pipeline.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=record))
    )
    return pipeline, calls


@pytest.mark.parametrize("provider", ["claude", "gpt"])
@pytest.mark.parametrize("invalid_docs", [
    None,
    [None],
    [{"doc_id": "d1", "title": "title"}],
    [{"doc_id": "d1", "title": "title", "text": math.nan}],
    [{"doc_id": "d1", "title": "title", "text": math.inf}],
    [{"doc_id": "d1", "title": "title", "text": object()}],
    [{"doc_id": "d1", "title": "title", "text": "x" * 50_000}],
])
def test_bad_or_oversized_evidence_never_calls_provider(provider, invalid_docs):
    pipeline, calls = _pipeline(provider)
    result = pipeline.generate("Question", invalid_docs)
    assert result["answer"] == ""
    assert result["error"] == "invalid generation evidence"
    assert result["latency_ms"] >= 0
    assert calls == []


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_circular_generation_evidence_is_excluded(provider):
    pipeline, calls = _pipeline(provider)
    circular = []
    circular.append(circular)
    result = pipeline.generate("Question", [
        {"doc_id": "d1", "title": "title", "text": circular}
    ])
    assert result["error"] == "invalid generation evidence"
    assert calls == []


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_valid_generation_context_still_reaches_provider(provider):
    pipeline, calls = _pipeline(provider)
    result = pipeline.generate("Question", [
        {"doc_id": "d1", "title": "title", "text": "Evidence"}
    ])
    assert result["answer"] == "grounded answer"
    assert "error" not in result
    assert len(calls) == 1


def test_generation_budget_applies_to_serialized_payload_not_raw_text():
    # JSON escaping expands literal backslashes. A raw-text-only size
    # check would underestimate the actual provider request.
    pipeline, calls = _pipeline("gpt")
    result = pipeline.generate("Question", [
        {"doc_id": "d1", "title": "title", "text": "\\n" * 15_000}
    ])
    assert result["error"] == "invalid generation evidence"
    assert calls == []
    assert MAX_RAG_GENERATION_EVIDENCE_CHARS < 60_000
