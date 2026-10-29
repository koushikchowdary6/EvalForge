"""A RAG ranking must not substitute evidence for indexed benchmark documents."""

import pytest

from evalforge.rag import RAGPipeline


CORPUS = [
    {"doc_id": "a", "title": "Alpha", "text": "Original evidence"},
    {"doc_id": "b", "title": "Beta", "text": "Another fact"},
]


def _query(retrieved, corpus=CORPUS):
    calls = []

    class Store:
        documents = corpus

        def search(self, question, top_k):
            calls.append("search")
            return retrieved

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = Store()

    def generate(question, docs):
        calls.append("generate")
        return {"answer": "Answer", "latency_ms": 2.0}

    pipeline.generate = generate
    return pipeline.query("Question", top_k=2), calls


@pytest.mark.parametrize("bad", [
    [{"doc_id": "unknown", "title": "Unknown", "text": "Fabricated evidence"}],
    [{"doc_id": "a", "title": "Altered title", "text": "Original evidence"}],
    [{"doc_id": "a", "title": "Alpha", "text": "Altered evidence"}],
])
def test_unknown_or_mutated_retrieval_fails_before_generation(bad):
    result, calls = _query(bad)
    assert calls == ["search"]
    assert result["error_stage"] == "retrieval"
    assert result["error"] == "retriever returned evidence outside indexed corpus"
    assert result["retrieved"] == []
    assert result["generation_ms"] == 0.0
    assert result["total_ms"] == result["retrieval_ms"]


@pytest.mark.parametrize("corpus", [
    [],
    [{"doc_id": "a", "title": "Alpha", "text": "Original evidence"}] * 2,
    [{"doc_id": [], "title": "Alpha", "text": "Original evidence"}],
    "invalid corpus",
])
def test_invalid_index_provenance_fails_closed(corpus):
    result, calls = _query([CORPUS[0]], corpus=corpus)
    assert calls == ["search"]
    assert result["error_stage"] == "retrieval"
    assert result["generation_ms"] == 0.0


def test_matching_indexed_documents_generate_normally():
    result, calls = _query([CORPUS[1], CORPUS[0]])
    assert calls == ["search", "generate"]
    assert result["retrieved"] == [CORPUS[1], CORPUS[0]]
    assert result["answer"] == "Answer"
    assert result["error"] is None


def test_empty_retrieval_is_valid_with_empty_index():
    result, calls = _query([], corpus=[])
    assert calls == ["search", "generate"]
    assert result["retrieved"] == []
    assert result["error"] is None
