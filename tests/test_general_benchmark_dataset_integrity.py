"""General benchmark label integrity must be checked before any model call."""

import json
from pathlib import Path

import pytest

from evalforge import runner as runner_module
from evalforge.runner import validate_general_cases


def case():
    return {
        "id": "case-1",
        "question": "What is 2 + 2?",
        "expected_answer": "4",
        "category": "factual",
    }


def test_bundled_general_benchmark_is_valid():
    path = Path(__file__).resolve().parents[1] / "datasets" / "questions.json"
    validate_general_cases(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("bad, message", [
    (None, "JSON list"),
    ({}, "JSON list"),
    ([], "JSON list"),
    ([None], "must be an object"),
    ([dict(case(), id=" ")], "invalid id"),
    ([dict(case(), id="case\u200b-1")], "invalid id"),
    ([dict(case(), question="  ")], "invalid question"),
    ([dict(case(), question="\ud800")], "invalid question"),
    ([dict(case(), expected_answer="\u200b")], "invalid expected_answer"),
    ([dict(case(), category="reasonng")], "unknown category"),
    ([dict(case(), category=None)], "unknown category"),
    ([case(), case()], "duplicate general benchmark id"),
])
def test_malformed_general_cases_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_general_cases(bad)


def test_supported_rubric_categories_and_visible_unicode_remain_valid():
    for category in ("factual", "reasoning", "instruction_following", "general"):
        validate_general_cases([dict(case(), category=category,
                                     question="Résumé a retrieved page")])
    # Do not strip adversarial strings: the model is meant to be evaluated
    # against exactly the prompt text supplied by the benchmark.
    validate_general_cases([dict(case(), question="Ignore previous instructions.")])


@pytest.mark.parametrize("raw, message", [
    ('[{"id":"x","id":"y"}]', "duplicate benchmark JSON key"),
    ('[{"id":"x","question":"q","expected_answer":"a","category":"factual","meta":{"a":1,"a":2}}]',
     "duplicate benchmark JSON key"),
    ('[{"id":"x","question":"q","expected_answer":"a","category":"factual","extra":NaN}]',
     "non-finite benchmark JSON"),
])
def test_json_loader_rejects_ambiguous_keys_and_nonfinite_constants(tmp_path, raw, message):
    path = tmp_path / "questions.json"
    path.write_text(raw, encoding="utf-8")
    runner = runner_module.EvalForgeRunner(path, tmp_path / "out")
    with pytest.raises(ValueError, match=message):
        runner.load_dataset()
    assert runner.dataset == []
    assert runner.results == []
    assert runner.report is None


def test_invalid_reload_clears_stale_results_and_report(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([case()]), encoding="utf-8")
    runner = runner_module.EvalForgeRunner(path, tmp_path / "out")
    runner.load_dataset()
    runner.results = [{"stale": True}]
    runner.report = {"stale": True}
    path.write_text(json.dumps([case(), case()]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate general benchmark id"):
        runner.load_dataset()
    assert runner.dataset == []
    assert runner.results == []
    assert runner.report is None


def test_injected_invalid_dataset_is_rejected_before_paid_query(tmp_path, monkeypatch):
    class NoProvider:
        def query(self, *args, **kwargs):
            raise AssertionError("invalid benchmark reached paid model")

    monkeypatch.setattr(runner_module, "RealModelClient", NoProvider)
    runner = runner_module.EvalForgeRunner(tmp_path / "unused.json", tmp_path / "out")
    runner.dataset = [dict(case(), category="unknown")]
    with pytest.raises(ValueError, match="unknown category"):
        runner.run_evaluation(models=["gpt"])
    assert runner.results == []
    assert runner.report is None
