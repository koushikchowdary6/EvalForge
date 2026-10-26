"""API-free tests for base evaluator provider completion integrity."""

from types import SimpleNamespace

import pytest

from evalforge.runner import RealModelClient


def _client(*, anthropic_response=None, openai_response=None):
    client = RealModelClient.__new__(RealModelClient)
    client.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kwargs: anthropic_response)
    )
    client.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: openai_response
        ))
    )
    return client


def _claude(*texts, stop_reason="end_turn", types=None):
    types = types or ["text"] * len(texts)
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type=kind, text=text)
                 for kind, text in zip(types, texts)],
        usage=SimpleNamespace(output_tokens=3),
    )


def _gpt(*texts, finish_reasons=None):
    finish_reasons = finish_reasons or ["stop"] * len(texts)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=reason,
                message=SimpleNamespace(content=text),
            )
            for reason, text in zip(finish_reasons, texts)
        ],
        usage=SimpleNamespace(completion_tokens=3),
    )


@pytest.mark.parametrize("response", [
    _claude("partial", stop_reason="max_tokens"),
    _claude("valid", "contradictory"),
    _claude("hidden", types=["thinking"]),
    _claude(""),
    _claude(),
])
def test_claude_incomplete_or_ambiguous_output_is_not_graded(response):
    result = _client(anthropic_response=response).query("claude", "Question")
    assert result.get("error")
    assert result["response"].startswith("[API Error:")


@pytest.mark.parametrize("response", [
    _gpt("partial", finish_reasons=["length"]),
    _gpt("filtered", finish_reasons=["content_filter"]),
    _gpt("answer", "contradictory"),
    _gpt(None),
    _gpt("  "),
    _gpt(),
])
def test_openai_incomplete_or_ambiguous_output_is_not_graded(response):
    result = _client(openai_response=response).query("gpt", "Question")
    assert result.get("error")
    assert result["response"].startswith("[API Error:")


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_single_completed_text_response_still_works(provider):
    client = _client(
        anthropic_response=_claude("42"),
        openai_response=_gpt("42"),
    )
    result = client.query(provider, "Question")
    assert result["response"] == "42"
    assert result["tokens_used"] == 3
    assert result.get("error") is None
