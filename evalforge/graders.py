"""
Evaluation graders for EvalForge.

Different grading strategies to evaluate model responses.
"""

import re


class ExactMatchGrader:
    """Grades responses that must match exactly (case-insensitive)."""

    def grade(self, expected, actual):
        """
        Check if actual response matches expected exactly.

        Args:
            expected: The ground-truth answer
            actual: The model's response

        Returns:
            dict with 'passed' (bool) and 'details' (str)
        """
        expected_clean = expected.strip().lower()
        actual_clean = actual.strip().lower()

        passed = expected_clean == actual_clean

        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "method": "exact_match",
            "details": f"Expected: '{expected}' | Got: '{actual}'"
        }


class RuleBasedGrader:
    """Grades responses based on rules (contains required text, length, format, etc)."""

    def grade(self, expected, actual, rules=None):
        """
        Grade response based on configurable rules.

        Args:
            expected: The ground-truth answer (used as reference)
            actual: The model's response
            rules: List of rule checks to perform

        Returns:
            dict with 'passed', 'score', 'details'
        """
        if rules is None:
            rules = []

        results = {
            "passed": True,
            "score": 1.0,
            "method": "rule_based",
            "failed_rules": [],
            "details": ""
        }

        # Default rule: must contain key words from expected answer
        # Extract key words (longer than 3 chars)
        key_words = [word for word in expected.split() if len(word) > 3]

        for keyword in key_words:
            if keyword.lower() not in actual.lower():
                results["failed_rules"].append(f"Missing key term: '{keyword}'")
                results["passed"] = False
                results["score"] -= 0.1

        # Custom rules
        for rule in rules:
            rule_type = rule.get("type")

            if rule_type == "min_length":
                min_len = rule.get("value", 10)
                if len(actual) < min_len:
                    results["failed_rules"].append(f"Response too short (min {min_len} chars)")
                    results["passed"] = False
                    results["score"] -= 0.2

            elif rule_type == "max_length":
                max_len = rule.get("value", 500)
                if len(actual) > max_len:
                    results["failed_rules"].append(f"Response too long (max {max_len} chars)")
                    results["passed"] = False
                    results["score"] -= 0.2

            elif rule_type == "contains":
                required_text = rule.get("value", "")
                if required_text.lower() not in actual.lower():
                    results["failed_rules"].append(f"Must contain: '{required_text}'")
                    results["passed"] = False
                    results["score"] -= 0.2

            elif rule_type == "not_contains":
                forbidden_text = rule.get("value", "")
                if forbidden_text.lower() in actual.lower():
                    results["failed_rules"].append(f"Must NOT contain: '{forbidden_text}'")
                    results["passed"] = False
                    results["score"] -= 0.2

        results["score"] = max(0.0, results["score"])
        results["details"] = " | ".join(results["failed_rules"]) if results["failed_rules"] else "All checks passed"

        return results


class LLMJudgeGrader:
    """
    Simulates an LLM-as-judge grader.
    In production, this would call Claude/GPT to evaluate quality.
    For now, we mock simple heuristics that mimic LLM judgment.
    """

    def grade(self, expected, actual, category=None):
        """
        Grade using heuristics that simulate LLM judgment.

        Args:
            expected: Ground-truth answer
            actual: Model's response
            category: Question category (factual, reasoning, instruction_following)

        Returns:
            dict with score (0-1) and reasoning
        """
        score = 0.5  # Start at neutral
        feedback = []

        # Length check: response should be somewhat proportional to expected
        if len(expected) > 50:  # Complex question
            if len(actual) < 20:
                score -= 0.3
                feedback.append("Response too brief for complex question")
            elif len(actual) > 2000:
                score -= 0.2
                feedback.append("Response unnecessarily verbose")

        # Semantic similarity (naive implementation)
        expected_words = set(w.lower() for w in expected.split() if len(w) > 3)
        actual_words = set(w.lower() for w in actual.split() if len(w) > 3)

        if expected_words:
            overlap = len(expected_words & actual_words) / len(expected_words)
            score += overlap * 0.4

        # Factual questions: exact match is heavily weighted
        if category == "factual":
            if expected.lower() in actual.lower():
                score = 0.95
            else:
                score -= 0.2

        # Reasoning: check for logical markers
        elif category == "reasoning":
            logical_phrases = ["because", "therefore", "thus", "since", "however", "but", "although"]
            if any(phrase in actual.lower() for phrase in logical_phrases):
                score += 0.2
            else:
                feedback.append("Reasoning lacks logical connectors")

        # Instruction following: check for compliance signals
        elif category == "instruction_following":
            # This would be more sophisticated in real LLM judge
            if len(actual.strip()) > 0:
                score += 0.2

        score = min(1.0, max(0.0, score))

        return {
            "passed": score >= 0.6,
            "score": score,
            "method": "llm_judge_mock",
            "feedback": " | ".join(feedback) if feedback else "Response quality acceptable",
            "details": f"Semantic overlap: {overlap:.1%}" if expected_words else "No comparison possible"
        }
