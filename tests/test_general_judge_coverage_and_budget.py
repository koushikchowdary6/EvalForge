"""Offline regression tests for honest judge coverage and evidence budgets."""

import json
from types import SimpleNamespace

import pytest

from evalforge.graders import LLMJudgeGrader, MAX_GENERAL_JUDGE_EVIDENCE_CHARS
from evalforge.runner import EvalForgeRunner


def _result(valid, passed=False):
    return {
        "case_id": "case",
        "model": "gpt",
        "category": "factual",
        "grading": {
            "exact_match": {"passed": True},
            "rule_based": {"passed": True},
            "llm_judge": {"passed": passed, "parse_ok": valid},
        },
    }


def test_judge_report_excludes_invalid_measurements_from_denominator():
    runner = EvalForgeRunner.__new__(EvalForgeRunner)
    runner.results = [_result(True, True), _result(False), _result(True, False)]
    runner.report = None

    report = runner.generate_report()["by_model"]["gpt"]
    assert report["exact_match"]["pass_rate"] == "100.0%"
    assert report["llm_judge"] == {
        "pass": 1,
        "fail": 1,
        "evaluated": 2,
        "unavailable": 1,
        "pass_rate": "50.0%",
    }


def test_no_valid_judge_responses_is_not_zero_percent_model_quality():
    runner = EvalForgeRunner.__new__(EvalForgeRunner)
    runner.results = [_result(False), _result(False)]
    runner.report = None
    judge = runner.generate_report()["by_model"]["gpt"]["llm_judge"]
    assert judge["evaluated"] == 0
    assert judge["unavailable"] == 2
    assert judge["pass_rate"] == "N/A"


def test_valid_and_invalid_judge_responses_have_explicit_coverage():
    grader = LLMJudgeGrader(judge_model="offline")
    valid = grader._parse_judgment(
        json.dumps({"score": 8, "reason": "Reference supported"}), "gpt"
    )
    invalid = grader._parse_judgment('{"score": 8}', "gpt")
    assert valid["parse_ok"] is True
    assert invalid["parse_ok"] is False


@pytest.mark.parametrize("expected", ["x" * (MAX_GENERAL_JUDGE_EVIDENCE_CHARS + 1), float("nan")])
def test_invalid_or_oversized_evidence_never_reaches_paid_judge(expected):
    calls = []
    grader = LLMJudgeGrader.__new__(LLMJudgeGrader)
    grader.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs))
        )
    )
    result = grader._judge_with_gpt(expected, "answer", "factual")
    assert result["passed"] is False
    assert result["parse_ok"] is False
    assert result["score"] == 0.0
    assert calls == []


def test_small_valid_evidence_still_reaches_judge():
    calls = []
    grader = LLMJudgeGrader.__new__(LLMJudgeGrader)

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"score": 9, "reason": "Correct answer"}'
                    ),
                )
            ]
        )

    grader.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = grader._judge_with_gpt("four", "4", "factual")
    assert result["parse_ok"] is True
    assert result["passed"] is True
    assert len(calls) == 1
