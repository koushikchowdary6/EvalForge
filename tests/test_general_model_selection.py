"""Model selections must be unique and ordered before any paid benchmark calls."""

import pytest

from evalforge import runner as runner_module


@pytest.mark.parametrize("models", [
    [], (), ["gpt", "gpt"], ("claude", "gpt", "claude"),
    "gpt", {"gpt"}, iter(["gpt"]), {"model": "gpt"},
    ["unknown"], [None], [["gpt"]],
])
def test_invalid_model_selection_clears_stale_report_without_provider_calls(
    tmp_path, monkeypatch, models
):
    class NoProvider:
        def query(self, *_args, **_kwargs):
            pytest.fail("invalid model selection reached paid provider")

    monkeypatch.setattr(runner_module, "RealModelClient", NoProvider)
    runner = runner_module.EvalForgeRunner(tmp_path / "unused.json", tmp_path / "out")
    runner.dataset = [{
        "id": "case-1",
        "question": "What is two plus two?",
        "expected_answer": "Four",
        "category": "factual",
    }]
    runner.results = [{"stale": True}]
    runner.report = {"stale": True}

    with pytest.raises(ValueError, match="unique, non-empty"):
        runner.run_evaluation(models=models)
    assert runner.results == []
    assert runner.report is None


def test_valid_ordered_model_selections_remain_supported(tmp_path, monkeypatch):
    class StubProvider:
        def query(self, model, _question):
            return {"model": model, "response": "Four", "latency_ms": 1}

    class StubGrader:
        def grade(self, *_args, **_kwargs):
            return {"passed": True, "score": 1.0, "parse_ok": True}

    monkeypatch.setattr(runner_module, "RealModelClient", StubProvider)
    for grader_name in ("ExactMatchGrader", "RuleBasedGrader", "LLMJudgeGrader"):
        monkeypatch.setattr(runner_module, grader_name, StubGrader)

    runner = runner_module.EvalForgeRunner(tmp_path / "unused.json", tmp_path / "out")
    runner.dataset = [{
        "id": "case-1", "question": "What is two plus two?",
        "expected_answer": "Four", "category": "factual",
    }]
    assert [r["model"] for r in runner.run_evaluation(models=("gpt", "claude"))] == [
        "gpt", "claude",
    ]
    assert runner.generate_report()["total_evaluations"] == 2
