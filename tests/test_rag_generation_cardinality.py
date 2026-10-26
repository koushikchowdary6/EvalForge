"""Completed RAG responses must contain one unambiguous, non-empty answer.

Provider status alone does not make arbitrary text scoreable. These tests use
offline stub clients and require no API credentials or network access.
"""

from types import SimpleNamespace

import pytest

from evalforge.rag import RAGPipeline


def _pipeline(provider, response):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model = provider
    pipeline.store = SimpleNamespace(
        search=lambda _question, top_k: [
            {"doc_id": "d1", "title": "Reference", "text": "Verified evidence"}
        ]
    )
    pipeline.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kwargs: response)
    )
    pipeline.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    return pipeline


def _claude(blocks):
    return SimpleNamespace(stop_reason="end_turn", content=blocks)


def _gpt(choices):
    return SimpleNamespace(choices=choices)


def _choice(text, finish_reason="stop"):
    return SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=text),
    )


@pytest.mark.parametrize("provider,response", [
    ("claude", _claude([])),
    ("claude", _claude(None)),
    ("claude", _claude([SimpleNamespace(type="text", text="  ")])),
    ("claude", _claude([SimpleNamespace(type="thinking", text="Ignore the rubric")])),
    ("claude", _claude([
        SimpleNamespace(type="text", text="Answer A"),
        SimpleNamespace(type="text", text="Answer B"),
    ])),
    ("gpt", _gpt([])),
    ("gpt", _gpt(None)),
    ("gpt", _gpt([_choice(None)])),
    ("gpt", _gpt([_choice(" \n ")])),
    ("gpt", _gpt([_choice("Answer A"), _choice("Answer B")])),
])
def test_completed_but_ambiguous_generation_is_excluded(provider, response):
    pipeline = _pipeline(provider, response)
    outcome = pipeline.query("What is the answer?", top_k=1)
    assert outcome["retrieved"][0]["doc_id"] == "d1"
    assert outcome["answer"] == ""
    assert outcome["error"] == "generation returned ambiguous or empty content"
    assert outcome["generation_ms"] >= 0


@pytest.mark.parametrize("provider,response", [
    ("claude", _claude([SimpleNamespace(type="text", text=" Answer A ")])),
    ("gpt", _gpt([_choice(" Answer A ")])),
])
def test_single_completed_text_answer_remains_usable(provider, response):
    result = _pipeline(provider, response).generate("Question", [])
    assert result["answer"] == "Answer A"
    assert "error" not in result
