"""Untrusted judge explanations cannot inject terminal controls or oversized report evidence.

Both general and RAG rubric parsers must exclude these grades from metrics.
The tests use local JSON only; no paid model calls are made.
"""

import json

import pytest

from evalforge.graders import LLMJudgeGrader
from evalforge.rag_graders import RAGJudge


@pytest.mark.parametrize("reason", [
    "Supported\u001b[2J",  # ANSI clear-screen escape after JSON decoding
    "Supported\u0000",  # NUL byte in a stored report
    "Supported\nForged metric: 10",
    "Supported\tspoofed-column",
    "Supported\u2028forged-line",
    "Supported\u2029forged-paragraph",
    "S" * 2049,  # Bounded explanations, independent of provider limits
])
def test_untrusted_judge_reasons_never_count_as_scores(reason):
    payload = json.dumps({"score": 10, "reason": reason}, ensure_ascii=True)

    rag = RAGJudge._parse_score(payload)
    assert rag["parse_ok"] is False

    general = LLMJudgeGrader(judge_model="offline")._parse_judgment(payload, "offline")
    assert general["parse_ok"] is False
    assert general["passed"] is False
    assert general["score"] == 0.0


@pytest.mark.parametrize("reason", [
    "Supported",
    "Café and résumé match",
    "Evidence supports 👍",
    "S" * 2048,
])
def test_visible_bounded_explanations_remain_valid(reason):
    payload = json.dumps({"score": 8, "reason": reason}, ensure_ascii=True)

    rag = RAGJudge._parse_score(payload)
    assert rag["parse_ok"] is True
    assert rag["score"] == pytest.approx(0.8)

    general = LLMJudgeGrader(judge_model="offline")._parse_judgment(payload, "offline")
    assert general["parse_ok"] is True
    assert general["score"] == pytest.approx(0.8)
