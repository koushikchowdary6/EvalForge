"""Provider failures must not persist request secrets in evaluation artifacts."""

from types import SimpleNamespace

import pytest

from evalforge.graders import LLMJudgeGrader
from evalforge.runner import RealModelClient


SECRET = "sk-test-sensitive-key-DO-NOT-PERSIST"


def _raise_private_provider_error(**_kwargs):
    raise RuntimeError(f"Authorization: Bearer {SECRET}; prompt=private customer context")


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_model_provider_error_is_redacted_from_result_and_stdout(provider, capsys):
    client = RealModelClient.__new__(RealModelClient)
    client.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(create=_raise_private_provider_error)
    )
    client.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_raise_private_provider_error)
        )
    )

    result = client.query(provider, "Sensitive evaluation question")
    assert result["response"] == "[API Error: RuntimeError]"
    assert result["error"] == "provider request failed (RuntimeError)"
    assert result["model"] == provider
    assert SECRET not in repr(result)
    assert SECRET not in capsys.readouterr().out


@pytest.mark.parametrize("provider", ["claude", "gpt"])
def test_judge_provider_error_is_redacted_from_result(provider):
    grader = LLMJudgeGrader.__new__(LLMJudgeGrader)
    grader.anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(create=_raise_private_provider_error)
    )
    grader.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_raise_private_provider_error)
        )
    )

    result = (
        grader._judge_with_claude("expected", "actual", "factual")
        if provider == "claude"
        else grader._judge_with_gpt("expected", "actual", "factual")
    )
    assert result["passed"] is False
    assert result["score"] == 0.0
    expected = "Claude" if provider == "claude" else "GPT"
    assert result["feedback"] == f"{expected} judging error (RuntimeError)"
    assert result["details"] == "judge provider request failed"
    assert SECRET not in repr(result)
