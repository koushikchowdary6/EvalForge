"""Offline regression tests for general-purpose LLM judge response integrity."""

import json

import pytest

from evalforge.graders import LLMJudgeGrader


@pytest.fixture
def grader():
    # Unsupported provider avoids initializing external clients or API calls.
    return LLMJudgeGrader(judge_model="offline")


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not JSON",
        "{}",
        "[]",
        "null",
        '{"score": 7}',
        '{"reason": "good"}',
        '{"score": null, "reason": "good"}',
        '{"score": true, "reason": "good"}',
        '{"score": "9", "reason": "good"}',
        '{"score": -1, "reason": "good"}',
        '{"score": 11, "reason": "good"}',
        '{"score": NaN, "reason": "good"}',
        '{"score": Infinity, "reason": "good"}',
        '{"score": 8, "reason": null}',
        '{"score": 8, "reason": ""}',
        '{"score": 8, "reason": "   "}',
        '{"score": 8, "reason": ["good"]}',
        '{"score": 8, "reason": "good"} extra',
        '{"score": 0, "score": 10, "reason": "contradictory"}',
        '{"score": 10, "reason": "good", "reason": "bad"}',
        '{"score": 10, "reason": "good", "metadata": {"a": 1, "a": 2}}',
        '{"score": 10, "reason": "good", "metadata": [NaN]}',
        '{"score": 10, "reason": "good", "metadata": Infinity}',
        '{"score": 10, "reason": "good", "confidence": 1}',
        '{"score": 10, "reason": "good", "confidence": null}',
        '{"score": 10, "reason": "good", "metadata": {}}',
        '{"score": 10, "reason": "good", "metadata": []}',
    ],
)
def test_malformed_or_ambiguous_judge_response_fails_closed(grader, payload):
    result = grader._parse_judgment(payload, "claude")
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["feedback"] == "Invalid judge response"
    assert result["method"] == "llm_judge"


@pytest.mark.parametrize("score,expected_pass", [(0, False), (5.9, False), (6, True), (10, True)])
def test_valid_numeric_rubric_scores_are_preserved(grader, score, expected_pass):
    result = grader._parse_judgment(
        json.dumps({"score": score, "reason": "Grounded rubric assessment"}),
        "gpt",
    )
    assert result["passed"] is expected_pass
    assert result["score"] == score / 10
    assert result["feedback"] == "Grounded rubric assessment"


def test_non_text_provider_payload_does_not_crash_or_award_credit(grader):
    assert grader._parse_judgment(None, "claude")["score"] == 0.0
    assert grader._parse_judgment({"score": 10}, "gpt")["score"] == 0.0


@pytest.mark.parametrize(
    "reason",
    [
        "\u200b",
        "\u0301",
        "\u200b\u0301",
        "valid\u202e",
        "valid\ud800",
        "valid\u2066",
    ],
)
def test_invisible_or_spoofed_judge_reasons_fail_closed(grader, reason):
    payload = json.dumps({"score": 10, "reason": reason}, ensure_ascii=True)
    result = grader._parse_judgment(payload, "gpt")
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["feedback"] == "Invalid judge response"


@pytest.mark.parametrize(
    "reason",
    ["Résumé matches the reference", "Cafe\u0301 is supported", "Grounded 👍"],
)
def test_legitimate_unicode_judge_reasons_remain_valid(grader, reason):
    result = grader._parse_judgment(
        json.dumps({"score": 8, "reason": reason}, ensure_ascii=True), "claude"
    )
    assert result["passed"] is True
    assert result["score"] == 0.8
    assert result["feedback"] == reason


def test_extremely_large_json_integer_is_invalid_not_an_exception(grader):
    result = grader._parse_judgment(
        json.dumps({"score": 10 ** 500, "reason": "valid"}), "gpt"
    )
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["feedback"] == "Invalid judge response"
