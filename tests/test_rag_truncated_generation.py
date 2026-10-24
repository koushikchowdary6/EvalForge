"""API-free regression tests for incomplete RAG model generations."""

from types import SimpleNamespace

import pytest

from evalforge.rag import RAGPipeline


@pytest.mark.parametrize(
    ("provider", "finish_reason"),
    [("claude", "max_tokens"), ("gpt", "length")],
)
def test_truncated_generation_is_an_error_not_a_scored_answer(provider, finish_reason):
    def create(**_kwargs):
        if provider == "claude":
            return SimpleNamespace(
                stop_reason=finish_reason,
                content=[SimpleNamespace(text="A plausible but incomplete answer")],
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content="A plausible but incomplete answer"),
            )]
        )

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model = provider
    pipeline.store = SimpleNamespace(search=lambda _question, top_k: [
        {"doc_id": "d1", "title": "Evidence", "text": "Known facts"}
    ])
    pipeline.anthropic_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    pipeline.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    outcome = pipeline.query("What happened?", top_k=1)
    assert outcome["retrieved"][0]["doc_id"] == "d1"
    assert outcome["answer"] == ""
    assert outcome["error"] == "generation truncated by token limit"
    assert outcome["generation_ms"] >= 0


@pytest.mark.parametrize(
    ("provider", "finish_reason"),
    [("claude", "end_turn"), ("gpt", "stop")],
)
def test_completed_generation_remains_scoreable(provider, finish_reason):
    def create(**_kwargs):
        if provider == "claude":
            return SimpleNamespace(
                stop_reason=finish_reason,
                content=[SimpleNamespace(text="A complete answer")],
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content="A complete answer"),
            )]
        )

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model = provider
    pipeline.anthropic_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    pipeline.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = pipeline.generate("Question", [])
    assert result["answer"] == "A complete answer"
    assert "error" not in result
