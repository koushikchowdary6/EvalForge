"""Offline regressions for EvalForge's model-provider error handling."""

import json

import pytest

from evalforge import runner as runner_module


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    def query(self, model, question):
        return next(self.responses)


class StubGrader:
    def __init__(self):
        self.calls = 0

    def grade(self, *args, **kwargs):
        self.calls += 1
        return {"passed": True, "score": 1.0, "method": "stub"}


def _runner(tmp_path, monkeypatch, responses, count=1):
    dataset = [
        {
            "id": f"case-{i}",
            "question": "What is 2+2?",
            "expected_answer": "4",
            "category": "factual",
        }
        for i in range(count)
    ]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    monkeypatch.setattr(runner_module, "RealModelClient", lambda: StubClient(responses))
    grader = StubGrader()
    monkeypatch.setattr(runner_module, "ExactMatchGrader", lambda: grader)
    monkeypatch.setattr(runner_module, "RuleBasedGrader", lambda: grader)
    monkeypatch.setattr(runner_module, "LLMJudgeGrader", lambda: grader)
    return runner_module.EvalForgeRunner(str(path), str(tmp_path / "out")), grader


@pytest.mark.parametrize(
    "error_response",
    [
        {"model": "gpt", "response": "[OpenAI API not configured]", "error": "API key missing", "latency_ms": 0},
        {"model": "gpt", "response": "[API Error: timeout]", "error": "timeout", "latency_ms": 12},
    ],
)
def test_provider_errors_abort_without_grading(tmp_path, monkeypatch, error_response):
    runner, grader = _runner(tmp_path, monkeypatch, [error_response])
    with pytest.raises(RuntimeError, match="aborted before grading"):
        runner.run_evaluation(models=["gpt"])
    assert runner.results == []
    assert runner.report is None
    assert grader.calls == 0
    assert runner.generate_report() is None


def test_error_after_success_does_not_publish_partial_benchmark(tmp_path, monkeypatch):
    responses = [
        {"model": "gpt", "response": "4", "latency_ms": 10},
        {"model": "gpt", "response": "[API Error]", "error": "rate limit", "latency_ms": 10},
    ]
    runner, grader = _runner(tmp_path, monkeypatch, responses, count=2)
    with pytest.raises(RuntimeError, match="case-1"):
        runner.run_evaluation(models=["gpt"])
    assert grader.calls == 3  # First response was graded by all three graders.
    assert runner.results == []
    assert runner.generate_report() is None


def test_successful_model_response_is_still_graded(tmp_path, monkeypatch):
    runner, grader = _runner(
        tmp_path, monkeypatch,
        [{"model": "gpt", "response": "4", "latency_ms": 5}],
    )
    results = runner.run_evaluation(models=["gpt"])
    assert len(results) == 1
    assert grader.calls == 3
    assert runner.generate_report()["by_model"]["gpt"]["total_cases"] == 1


def test_save_results_removes_stale_report_after_provider_failure(tmp_path, monkeypatch):
    runner, _ = _runner(
        tmp_path, monkeypatch,
        [{"model": "gpt", "response": "[API Error]", "error": "timeout", "latency_ms": 1}],
    )
    report_path = tmp_path / "out" / "report.json"
    report_path.write_text('{"obsolete": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="aborted before grading"):
        runner.run_evaluation(models=["gpt"])
    runner.save_results()
    assert not report_path.exists()
    assert json.loads((tmp_path / "out" / "results.json").read_text()) == []


@pytest.mark.parametrize("invalid_response", [
    {"model": "gpt", "response": None, "latency_ms": 10},
    {"model": "gpt", "response": "  ", "latency_ms": 10},
    {"model": "gpt", "latency_ms": 10},
])
def test_empty_or_missing_stub_response_cannot_receive_grade(
    tmp_path, monkeypatch, invalid_response
):
    runner, grader = _runner(tmp_path, monkeypatch, [invalid_response])
    with pytest.raises(RuntimeError, match="aborted before grading"):
        runner.run_evaluation(models=["gpt"])
    assert runner.results == []
    assert grader.calls == 0
