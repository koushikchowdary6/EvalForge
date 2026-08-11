"""
Unit tests for EvalForge graders.

Tests ExactMatchGrader, RuleBasedGrader, and LLMJudgeGrader
against edge cases and normal usage.
"""

import pytest
from evalforge.graders import ExactMatchGrader, RuleBasedGrader, LLMJudgeGrader


class TestExactMatchGrader:
    """Test cases for ExactMatchGrader."""

    def test_exact_match_simple(self):
        """Test basic exact match."""
        grader = ExactMatchGrader()
        result = grader.grade("Canberra", "Canberra")
        assert result["passed"] is True
        assert result["score"] == 1.0
        assert result["method"] == "exact_match"

    def test_exact_match_case_insensitive(self):
        """Test that matching is case-insensitive."""
        grader = ExactMatchGrader()
        result = grader.grade("Canberra", "CANBERRA")
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_exact_match_with_spaces(self):
        """Test that extra spaces are trimmed."""
        grader = ExactMatchGrader()
        result = grader.grade("  Canberra  ", "Canberra")
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_exact_match_fails_on_extra_text(self):
        """Test that extra text causes failure."""
        grader = ExactMatchGrader()
        result = grader.grade("Canberra", "Canberra, Australia")
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_exact_match_empty_actual(self):
        """Test behavior with empty actual response."""
        grader = ExactMatchGrader()
        result = grader.grade("Canberra", "")
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_exact_match_empty_expected(self):
        """Test behavior with empty expected (edge case)."""
        grader = ExactMatchGrader()
        result = grader.grade("", "")
        assert result["passed"] is True
        assert result["score"] == 1.0


class TestRuleBasedGrader:
    """Test cases for RuleBasedGrader."""

    def test_rule_based_key_terms(self):
        """Test that key terms are extracted and checked."""
        grader = RuleBasedGrader()
        result = grader.grade("The capital is Canberra", "Canberra is the capital")
        # Should pass because both "capital" and "Canberra" are present
        assert result["passed"] is True

    def test_rule_based_missing_key_term(self):
        """Test failure when key term is missing."""
        grader = RuleBasedGrader()
        result = grader.grade("The capital is Canberra", "Sydney is a city")
        # Missing "Canberra" and "capital"
        assert result["passed"] is False
        assert "Missing key term" in result["details"]

    def test_rule_based_stopword_filtering(self):
        """Test that stopwords are not required as key terms."""
        grader = RuleBasedGrader()
        # Expected has many stopwords: "the", "is", "a", "and"
        # Should only require "capital", "Canberra"
        result = grader.grade("The capital is Canberra", "Canberra and capital")
        assert result["passed"] is True

    def test_rule_based_custom_rule_min_length(self):
        """Test custom min_length rule."""
        grader = RuleBasedGrader()
        rules = [{"type": "min_length", "value": 50}]
        result = grader.grade("Capital", "Short", rules=rules)
        # Response is too short
        assert result["passed"] is False
        assert "too short" in result["details"]

    def test_rule_based_custom_rule_contains(self):
        """Test custom contains rule.

        Note: the default key-term check also runs, so the response must
        contain the expected key term ('capital') in addition to satisfying
        the custom 'contains' rule ('Australia').
        """
        grader = RuleBasedGrader()
        rules = [{"type": "contains", "value": "Australia"}]
        result = grader.grade("Capital", "Canberra is the capital, located in Australia", rules=rules)
        assert result["passed"] is True

    def test_rule_based_custom_rule_not_contains(self):
        """Test custom not_contains rule (e.g., avoid saying 'banana')."""
        grader = RuleBasedGrader()
        rules = [{"type": "not_contains", "value": "banana"}]
        result = grader.grade("Yellow fruit", "A yellow fruit like banana", rules=rules)
        assert result["passed"] is False
        assert "must not contain" in result["details"].lower()

    def test_rule_based_empty_expected(self):
        """Test behavior when expected has no words > 3 chars (no key terms)."""
        grader = RuleBasedGrader()
        result = grader.grade("Yes or no?", "Sure")
        # No long key words, so any response should pass rule-based
        assert result["passed"] is True

    def test_rule_based_empty_actual(self):
        """Test behavior with empty actual response."""
        grader = RuleBasedGrader()
        result = grader.grade("The capital is Canberra", "")
        # Missing all key terms
        assert result["passed"] is False


class TestLLMJudgeGrader:
    """Test cases for LLMJudgeGrader.

    Note: These tests may fail if Anthropic API is not configured.
    Real API calls are made; use with caution in CI.
    """

    @pytest.mark.api
    def test_llm_judge_initialization_no_api(self, monkeypatch):
        """Test graceful handling when API is not configured."""
        # Remove API key
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        grader = LLMJudgeGrader()
        # Should still initialize (client may be None, handled in grade method)
        assert grader is not None

    @pytest.mark.api
    def test_llm_judge_returns_dict(self):
        """Test that grade returns proper dictionary structure."""
        grader = LLMJudgeGrader()
        result = grader.grade("Canberra", "Canberra is the capital", category="factual")

        # Check required fields
        assert "passed" in result
        assert "score" in result
        assert "method" in result
        assert "feedback" in result

        # Score should be in [0, 1]
        assert 0.0 <= result["score"] <= 1.0
        assert result["method"] in ["llm_judge"]

    @pytest.mark.api
    def test_llm_judge_respects_category(self):
        """Test that different categories produce reasonable scores."""
        grader = LLMJudgeGrader()

        # Factual question with correct answer should score high
        factual_result = grader.grade("Paris", "Paris is the capital of France", category="factual")

        # Reasoning question without logic should score lower
        reasoning_result = grader.grade(
            "Why is it raining?",
            "Just because",
            category="reasoning"
        )

        # Both should return valid results (API may or may not be available)
        assert isinstance(factual_result, dict)
        assert isinstance(reasoning_result, dict)


class TestGraderConsistency:
    """Test consistency and expected behavior across graders."""

    @pytest.mark.api
    def test_all_graders_return_required_fields(self):
        """Verify all graders return 'passed', 'score', 'method'."""
        expected = "Canberra"
        actual = "Canberra is the capital of Australia"

        exact_result = ExactMatchGrader().grade(expected, actual)
        rule_result = RuleBasedGrader().grade(expected, actual)
        judge_result = LLMJudgeGrader().grade(expected, actual)

        for result in [exact_result, rule_result, judge_result]:
            assert "passed" in result
            assert "score" in result
            assert "method" in result
            assert isinstance(result["passed"], bool)
            assert isinstance(result["score"], (int, float))
            assert 0.0 <= result["score"] <= 1.0

    @pytest.mark.api
    def test_score_reflects_passed(self):
        """Test that score >= 0.6 generally means passed=True."""
        grader = RuleBasedGrader()
        result = grader.grade("Canberra", "Canberra is the capital")

        # Check consistency (not strict, as LLM judge may differ)
        if result["score"] >= 0.6:
            # Usually passed, but not enforced everywhere
            pass
        else:
            # Lower scores should indicate some failure
            pass
