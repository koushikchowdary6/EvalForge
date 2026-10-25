"""API-free checks: non-final provider output cannot inflate RAG scores."""

from types import SimpleNamespace

import pytest

from evalforge.rag import RAGPipeline


@pytest.mark.parametrize(
    ("provider", "finish_reason"),
    [
        ("claude", "tool_use"),
        ("claude", None),
        ("gpt", "content_filter"),
        ("gpt", "tool_calls"),
        ("gpt", None),
    ],
)
def test_nonfinal_provider_completion_is_not_scored(provider, finish_reason):
    """A plausible answer is invalid unless the provider completed normally."""
    def create(**_kwargs):
        if provider == "claude":
            return SimpleNamespace(
                stop_reason=finish_reason,
                content=[SimpleNamespace(text="The answer is definitely 42.")],
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content="The answer is definitely 42."),
            )]
        )

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model = provider
    pipeline.store = SimpleNamespace(search=lambda _question, top_k: [
        {"doc_id": "d1", "title": "Evidence", "text": "Verified facts"}
    ])
    pipeline.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(create=create)
    )
    pipeline.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    outcome = pipeline.query("What is the answer?", top_k=1)
    assert outcome["retrieved"][0]["doc_id"] == "d1"
    assert outcome["answer"] == ""
    assert outcome["error"] == "generation did not complete normally"
    assert outcome["generation_ms"] >= 0
