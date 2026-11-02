"""Custom retrievers must not introduce invalid numeric ranking evidence.

Scores are optional because some adapters return only ordered documents. When
provided, finite BM25-like scores above one remain valid; NaN, Infinity,
booleans, and strings cannot be treated as trustworthy ranking diagnostics.
"""

import math

import pytest

from evalforge.rag import RAGPipeline


def _run_with_score(score, include_score=True):
    calls = []

    class Retriever:
        def search(self, question, top_k):
            calls.append("retrieve")
            document = {"doc_id": "doc-1", "title": "Title", "text": "Evidence"}
            if include_score:
                document["score"] = score
            return [document]

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.store = Retriever()

    def generate(question, documents):
        calls.append("generate")
        return {"answer": "Answer", "latency_ms": 1.0}

    pipeline.generate = generate
    return pipeline.query("Question", top_k=1), calls


@pytest.mark.parametrize("invalid", [
    float("nan"), float("inf"), -float("inf"), True, False,
    "0.8", None, [], {},
])
def test_invalid_adapter_scores_fail_before_paid_generation(invalid):
    result, calls = _run_with_score(invalid)
    assert calls == ["retrieve"]
    assert result["error_stage"] == "retrieval"
    assert result["error"] == "retriever returned invalid or duplicate document evidence"
    assert result["retrieved"] == []
    assert result["answer"] == ""
    assert result["generation_ms"] == 0.0


@pytest.mark.parametrize("valid", [0, 0.0, -0.7, 1.0, 17.5, -20])
def test_finite_scores_allow_noncosine_relevance_scales(valid):
    assert math.isfinite(valid)
    result, calls = _run_with_score(valid)
    assert calls == ["retrieve", "generate"]
    assert result["error"] is None
    assert result["retrieved"][0]["score"] == valid


def test_score_is_optional_for_rank_only_retrievers():
    result, calls = _run_with_score(None, include_score=False)
    assert calls == ["retrieve", "generate"]
    assert result["error"] is None
    assert "score" not in result["retrieved"][0]
