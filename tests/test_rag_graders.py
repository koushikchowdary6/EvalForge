"""
Unit tests for RAG retrieval metrics and abstention logic.

Every test here is deterministic and API-free, so the whole file runs in
CI. The LLM-judge paths are covered separately and marked 'api'.
"""

import pytest

from evalforge.rag import cosine_similarity, RAGPipeline
from evalforge.rag_graders import (
    precision_at_k, recall_at_k, reciprocal_rank, hit_rate,
    did_abstain, grade_abstention, RAGJudge,
)


class TestCosineSimilarity:
    """Cosine similarity is the core of retrieval - it must be exactly right."""

    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_ignores_magnitude(self):
        """A doc twice as long must not score differently for that reason."""
        assert cosine_similarity([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        """Degenerate embeddings must not raise ZeroDivisionError."""
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_half_relevant(self):
        assert precision_at_k(["a", "b"], ["a"]) == pytest.approx(0.5)

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], ["a"]) == 0.0

    def test_empty_retrieval(self):
        assert precision_at_k([], ["a"]) == 0.0

    def test_truncates_to_k(self):
        """Only the top k should count, not the whole ranking."""
        assert precision_at_k(["a", "x", "y"], ["a"], k=1) == pytest.approx(1.0)


class TestRecallAtK:
    def test_found_all(self):
        assert recall_at_k(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_found_half(self):
        assert recall_at_k(["a", "x"], ["a", "b"]) == pytest.approx(0.5)

    def test_found_none(self):
        assert recall_at_k(["x"], ["a"]) == 0.0

    def test_no_relevant_docs_is_perfect(self):
        """Unanswerable questions must not penalise the retriever."""
        assert recall_at_k(["x"], []) == 1.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "x"], ["a"]) == pytest.approx(1.0)

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], ["a"]) == pytest.approx(0.5)

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1.0 / 3.0)

    def test_not_found(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_rewards_earlier_rank(self):
        """MRR must strictly prefer a higher-ranked relevant doc."""
        assert reciprocal_rank(["a", "x"], ["a"]) > reciprocal_rank(["x", "a"], ["a"])


class TestHitRate:
    def test_hit(self):
        assert hit_rate(["x", "a"], ["a"]) == 1.0

    def test_miss(self):
        assert hit_rate(["x", "y"], ["a"]) == 0.0

    def test_no_relevant_docs(self):
        assert hit_rate(["x"], []) == 1.0


class TestAbstention:
    def test_detects_abstain_token(self):
        assert did_abstain("INSUFFICIENT_CONTEXT") is True

    def test_detects_abstain_case_insensitive(self):
        assert did_abstain("insufficient_context") is True

    def test_normal_answer_is_not_abstention(self):
        assert did_abstain("The capital is Canberra.") is False

    def test_none_answer_is_safe(self):
        assert did_abstain(None) is False

    def test_correct_answer_outcome(self):
        result = grade_abstention("Canberra.", answerable=True)
        assert result["outcome"] == "correct_answer"
        assert result["passed"] is True

    def test_correct_abstain_outcome(self):
        result = grade_abstention("INSUFFICIENT_CONTEXT", answerable=False)
        assert result["outcome"] == "correct_abstain"
        assert result["passed"] is True

    def test_hallucination_outcome(self):
        """Answering an unanswerable question is the failure we most care about."""
        result = grade_abstention("The rate was 4.2 percent.", answerable=False)
        assert result["outcome"] == "hallucination"
        assert result["passed"] is False

    def test_over_abstention_outcome(self):
        result = grade_abstention("INSUFFICIENT_CONTEXT", answerable=True)
        assert result["outcome"] == "over_abstention"
        assert result["passed"] is False


class TestJudgeParsing:
    """Judge response parsing must survive the formats models actually emit."""

    def test_parses_plain_json(self):
        result = RAGJudge._parse_score('{"score": 8, "reason": "good"}')
        assert result["score"] == pytest.approx(0.8)
        assert result["parse_ok"] is True

    def test_parses_markdown_fenced_json(self):
        raw = '```json\n{"score": 10, "reason": "perfect"}\n```'
        result = RAGJudge._parse_score(raw)
        assert result["score"] == pytest.approx(1.0)
        assert result["parse_ok"] is True

    def test_parses_json_with_surrounding_prose(self):
        raw = 'Here is my grade: {"score": 6, "reason": "partial"} - hope that helps'
        result = RAGJudge._parse_score(raw)
        assert result["score"] == pytest.approx(0.6)

    def test_unparseable_returns_neutral_and_flags(self):
        """A parse failure must be visible, not silently scored as a pass."""
        result = RAGJudge._parse_score("I think it was pretty good actually")
        assert result["score"] == pytest.approx(0.5)
        assert result["parse_ok"] is False

    def test_clamps_out_of_range_score(self):
        result = RAGJudge._parse_score('{"score": 50}')
        assert result["score"] == pytest.approx(1.0)


class TestPromptConstruction:
    def test_prompt_includes_abstention_instruction(self):
        """Without this instruction, hallucination cannot be measured."""
        docs = [{"doc_id": "d1", "title": "T", "text": "Body text."}]
        prompt = RAGPipeline.build_prompt("Q?", docs)
        assert "INSUFFICIENT_CONTEXT" in prompt

    def test_prompt_includes_retrieved_context(self):
        docs = [{"doc_id": "d1", "title": "Title", "text": "Unique body text."}]
        prompt = RAGPipeline.build_prompt("Q?", docs)
        assert "Unique body text." in prompt
        assert "d1" in prompt

    def test_prompt_includes_question(self):
        docs = [{"doc_id": "d1", "title": "T", "text": "B"}]
        prompt = RAGPipeline.build_prompt("What is groundedness?", docs)
        assert "What is groundedness?" in prompt
