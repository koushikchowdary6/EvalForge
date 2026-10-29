"""Fail closed on malformed custom RAG retrieval adapters before paid generation.

A retriever returning bad document shapes or repeated IDs must not crash the
reporting loop, inflate ranking metrics, or send ambiguous evidence to a model.
"""

import pytest

from evalforge.rag import RAGPipeline


def _pipeline(retrieved):
    calls = []

    class Store:
        def search(self, question, top_k):
            calls.append(("search", question, top_k))
            return retrieved

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = Store()

    def generate(question, documents):
        calls.append(("generate", question, documents))
        return {"answer": "Grounded answer", "latency_ms": 2.0}

    pipeline.generate = generate
    return pipeline, calls


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "2", None])
def test_invalid_cutoff_never_reaches_custom_retriever_or_generator(invalid):
    pipeline, calls = _pipeline([])
    result = pipeline.query("Question", top_k=invalid)

    assert calls == []
    assert result["error_stage"] == "retrieval"
    assert result["error"] == "retrieval failed (ValueError)"
    assert result["retrieved"] == []
    assert result["generation_ms"] == 0.0


@pytest.mark.parametrize("invalid_docs", [
    [None],
    [{"title": "A", "text": "Evidence"}],
    [{"doc_id": "a", "title": "A"}],
    [{"doc_id": "a", "title": "A", "text": None}],
    [{"doc_id": " a", "title": "A", "text": "Evidence"}],
    [{"doc_id": "a", "title": "\ud800", "text": "Evidence"}],
    [{"doc_id": "a", "title": "A", "text": "\udfff"}],
    [
        {"doc_id": "a", "title": "A", "text": "First"},
        {"doc_id": "a", "title": "B", "text": "Repeated ID"},
    ],
])
def test_invalid_or_duplicate_retrieval_is_excluded_before_generation(invalid_docs):
    pipeline, calls = _pipeline(invalid_docs)
    result = pipeline.query("Question", top_k=3)

    assert len(calls) == 1
    assert calls[0][0] == "search"
    assert result["error_stage"] == "retrieval"
    assert result["error"] == "retriever returned invalid or duplicate document evidence"
    assert result["retrieved"] == []
    assert result["answer"] == ""
    assert result["generation_ms"] == 0.0
    assert result["total_ms"] == result["retrieval_ms"]


def test_valid_distinct_evidence_still_generates():
    documents = [
        {"doc_id": "a", "title": "A", "text": "First"},
        {"doc_id": "b", "title": "B", "text": "Second"},
    ]
    pipeline, calls = _pipeline(documents)
    result = pipeline.query("Question", top_k=2)

    assert [entry[0] for entry in calls] == ["search", "generate"]
    assert calls[1] == ("generate", "Question", documents)
    assert result["retrieved"] == documents
    assert result["answer"] == "Grounded answer"
    assert result["error"] is None
