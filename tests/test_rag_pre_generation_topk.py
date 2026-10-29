"""Reject faulty retrieval cardinality before any model generation call."""

from evalforge.rag import RAGPipeline


def test_overreturned_evidence_never_reaches_generation():
    class OverReturningStore:
        def search(self, question, top_k):
            assert top_k == 1
            return [
                {"doc_id": "a", "title": "A", "text": "A"},
                {"doc_id": "b", "title": "B", "text": "B"},
            ]

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = OverReturningStore()

    def forbidden_generation(*args):
        raise AssertionError("extra evidence must not reach the paid model")

    pipeline.generate = forbidden_generation
    result = pipeline.query("What happened?", top_k=1)
    assert result["error_stage"] == "retrieval"
    assert result["error"] == "retriever violated top_k result contract"
    assert len(result["retrieved"]) == 2
    assert result["answer"] == ""
    assert result["generation_ms"] == 0.0
    assert result["total_ms"] == result["retrieval_ms"]


def test_invalid_retriever_shape_fails_without_generation():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = type("Store", (), {"search": lambda self, question, top_k: None})()
    pipeline.generate = lambda *args: (_ for _ in ()).throw(AssertionError())
    result = pipeline.query("Question", top_k=2)
    assert result["error_stage"] == "retrieval"
    assert result["retrieved"] == []
    assert result["generation_ms"] == 0.0


def test_within_budget_evidence_still_generates():
    evidence = [{"doc_id": "a", "title": "A", "text": "Grounded context"}]
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = type("Store", (), {"search": lambda self, question, top_k: evidence})()
    calls = []

    def generate(question, retrieved):
        calls.append((question, retrieved))
        return {"answer": "Grounded answer", "latency_ms": 4.0}

    pipeline.generate = generate
    result = pipeline.query("Question", top_k=1)
    assert result["answer"] == "Grounded answer"
    assert result["error"] is None
    assert calls == [("Question", evidence)]
    assert result["generation_ms"] == 4.0
