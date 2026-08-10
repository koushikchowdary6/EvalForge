"""
Evaluation graders for EvalForge.

Different grading strategies to evaluate model responses.
"""

import re
import os
from dotenv import load_dotenv

load_dotenv()

# Common English stopwords to filter out when extracting key terms
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "can", "this", "that", "these", "those", "i", "you",
    "he", "she", "it", "we", "they", "what", "which", "who", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "same", "so", "than", "too",
    "very", "just", "as", "because", "if", "then", "also", "here", "there",
    "about", "after", "before", "between", "during", "from", "by", "through",
    "without", "over", "under", "up", "down", "out", "off", "above", "below",
    "into", "out", "across", "along", "around", "as", "before", "behind",
    "between", "during", "inside", "outside", "over", "through", "toward", "under",
    "until", "while", "within", "without", "your", "our", "their", "my", "his", "her"
}


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
        # Extract key words: longer than 3 chars AND not a stopword
        key_words = [
            word.strip('.,!?;:') for word in expected.split()
            if len(word) > 3 and word.lower().strip('.,!?;:') not in STOPWORDS
        ]

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
    Uses Claude as a real LLM judge to evaluate response quality.
    Makes an actual API call to grade responses against criteria.
    """

    def __init__(self):
        """Initialize Anthropic client for judging."""
        self.client = None
        try:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = Anthropic(api_key=api_key)
        except Exception as e:
            print(f"⚠ LLMJudgeGrader: Could not initialize Anthropic client: {e}")

    def grade(self, expected, actual, category=None):
        """
        Grade response using Claude as a real LLM judge.

        Args:
            expected: Ground-truth answer (reference)
            actual: Model's response to evaluate
            category: Question category (factual, reasoning, instruction_following)

        Returns:
            dict with score (0-1), passed (bool), and feedback
        """
        if not self.client:
            return {
                "passed": False,
                "score": 0.0,
                "method": "llm_judge",
                "feedback": "LLM judge unavailable (Anthropic API not configured)",
                "details": "API key missing or client initialization failed"
            }

        # Build the grading rubric
        rubric = self._build_rubric(category)

        # Call Claude to judge
        try:
            message = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are an expert evaluation judge. Grade the following response.

Question Category: {category or 'general'}
Expected/Reference Answer: {expected}

Actual Response: {actual}

{rubric}

Respond ONLY with a single JSON line: {{"score": <0-10>, "reason": "<brief reason>"}}
No other text."""
                    }
                ]
            )

            # Parse Claude's response
            response_text = message.content[0].text.strip()
            import json as json_lib
            try:
                judgment = json_lib.loads(response_text)
                score = float(judgment.get("score", 5)) / 10.0  # Convert 0-10 scale to 0-1
                score = min(1.0, max(0.0, score))  # Clamp to [0, 1]
                reason = judgment.get("reason", "")
            except (json_lib.JSONDecodeError, ValueError):
                # If Claude's response isn't valid JSON, extract score if present
                score = 0.5
                reason = response_text[:100]

            return {
                "passed": score >= 0.6,
                "score": score,
                "method": "llm_judge",
                "feedback": reason,
                "details": f"Claude-evaluated; raw score: {score:.2f}"
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "method": "llm_judge",
                "feedback": f"Judging error: {str(e)[:80]}",
                "details": str(e)
            }

    def _build_rubric(self, category):
        """Build grading rubric based on question category."""
        if category == "factual":
            return """Grading Criteria (score 0-10):
- 9-10: Contains correct core fact(s) clearly stated
- 7-8: Correct but with minor elaboration or imprecision
- 5-6: Partially correct, some key information missing or confused
- 3-4: Mostly incorrect, incorrect core fact
- 0-2: Completely wrong or non-responsive"""

        elif category == "reasoning":
            return """Grading Criteria (score 0-10):
- 9-10: Logical chain clear; justified conclusion
- 7-8: Sound reasoning with minor gaps
- 5-6: Some logic but incomplete or unclear
- 3-4: Flawed reasoning, unsupported conclusion
- 0-2: No coherent reasoning"""

        elif category == "instruction_following":
            return """Grading Criteria (score 0-10):
- 9-10: Follows all instructions precisely
- 7-8: Follows most instructions, minor deviations
- 5-6: Follows ~50% of instructions
- 3-4: Follows some instructions but misses key ones
- 0-2: Ignores or violates most instructions"""

        else:
            return """Grading Criteria (score 0-10):
- 9-10: Excellent; answers question well, coherent
- 7-8: Good; mostly correct, minor issues
- 5-6: Acceptable; correct but incomplete
- 3-4: Poor; incorrect or confusing
- 0-2: Very poor; non-responsive or wrong"""
