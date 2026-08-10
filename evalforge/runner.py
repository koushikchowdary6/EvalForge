"""
EvalForge Evaluation Runner

Loads test dataset, queries models, grades responses, generates reports.
"""

import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from evalforge.graders import ExactMatchGrader, RuleBasedGrader, LLMJudgeGrader

# Load environment variables from .env file
load_dotenv()


class RealModelClient:
    """
    Real LLM client that calls OpenAI and Anthropic APIs.
    Uses API keys from .env file.
    """

    def __init__(self):
        """Initialize API clients with keys from .env"""
        # Import here to avoid errors if libraries aren't installed
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            print(f"⚠ OpenAI client failed: {e}")
            self.openai_client = None

        try:
            from anthropic import Anthropic
            self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except Exception as e:
            print(f"⚠ Anthropic client failed: {e}")
            self.anthropic_client = None

    def query(self, model_name, question):
        """
        Query a real model via API.

        Args:
            model_name: "claude" or "gpt"
            question: Question text

        Returns:
            dict with response and metadata
        """
        start_time = time.time()

        try:
            if model_name == "claude":
                if not self.anthropic_client:
                    return {
                        "model": "claude",
                        "response": "[Anthropic API not configured]",
                        "latency_ms": 0,
                        "error": "API key missing"
                    }

                message = self.anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    messages=[
                        {"role": "user", "content": question}
                    ]
                )
                response = message.content[0].text
                latency_ms = (time.time() - start_time) * 1000

                return {
                    "model": "claude",
                    "response": response,
                    "latency_ms": latency_ms,
                    "tokens_used": message.usage.output_tokens
                }

            elif model_name == "gpt":
                if not self.openai_client:
                    return {
                        "model": "gpt",
                        "response": "[OpenAI API not configured]",
                        "latency_ms": 0,
                        "error": "API key missing"
                    }

                completion = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": question}
                    ],
                    max_tokens=256
                )
                response = completion.choices[0].message.content
                latency_ms = (time.time() - start_time) * 1000

                return {
                    "model": "gpt",
                    "response": response,
                    "latency_ms": latency_ms,
                    "tokens_used": completion.usage.completion_tokens
                }

        except Exception as e:
            print(f"✗ Error querying {model_name}: {str(e)}")
            return {
                "model": model_name,
                "response": f"[API Error: {str(e)[:100]}]",
                "latency_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }


class EvalForgeRunner:
    """Main evaluation runner."""

    def __init__(self, dataset_path, output_dir="results"):
        """
        Initialize the runner.

        Args:
            dataset_path: Path to questions.json
            output_dir: Where to save results
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.dataset = []
        self.results = []
        self.model_client = RealModelClient()

        os.makedirs(output_dir, exist_ok=True)

    def load_dataset(self):
        """Load test dataset from JSON."""
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} test cases")
        return self.dataset

    def run_evaluation(self, models=None):
        """
        Run evaluation across all test cases and models.

        Args:
            models: List of model names to evaluate (default: ["claude", "gpt"])
        """
        if models is None:
            models = ["claude", "gpt"]

        if not self.dataset:
            self.load_dataset()

        print(f"\nStarting evaluation across {len(self.dataset)} cases for {len(models)} models...\n")

        # Initialize graders
        exact_grader = ExactMatchGrader()
        rule_grader = RuleBasedGrader()
        judge_grader = LLMJudgeGrader()

        for test_case in self.dataset:
            case_id = test_case["id"]
            question = test_case["question"]
            expected = test_case["expected_answer"]
            category = test_case["category"]

            for model in models:
                # Query the real model
                print(f"  Querying {model}...", end="", flush=True)
                result = self.model_client.query(model, question)
                actual = result["response"]
                latency = result.get("latency_ms", 0)
                print(f" ✓ ({latency:.0f}ms)")

                # Grade with all methods
                exact_result = exact_grader.grade(expected, actual)
                rule_result = rule_grader.grade(expected, actual)
                judge_result = judge_grader.grade(expected, actual, category)

                # Combine results
                eval_result = {
                    "case_id": case_id,
                    "model": model,
                    "question": question,
                    "category": category,
                    "expected": expected,
                    "actual": actual,
                    "latency_ms": latency,
                    "grading": {
                        "exact_match": exact_result,
                        "rule_based": rule_result,
                        "llm_judge": judge_result
                    },
                    "timestamp": datetime.now().isoformat()
                }

                self.results.append(eval_result)

                # Summary for debugging
                em_pass = "✓" if exact_result["passed"] else "✗"
                print(f"{em_pass} {case_id:15} | {model:6} | {category:20}")

        print(f"\nEvaluation complete. {len(self.results)} total results.")
        return self.results

    def generate_report(self):
        """Generate evaluation report with statistics."""
        if not self.results:
            print("No results to report. Run evaluation first.")
            return

        # Group by model
        by_model = {}
        for result in self.results:
            model = result["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)

        # Calculate statistics
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(set(r["case_id"] for r in self.results)),
            "total_evaluations": len(self.results),
            "models_evaluated": list(by_model.keys()),
            "by_model": {}
        }

        for model, results in by_model.items():
            exact_passes = sum(1 for r in results if r["grading"]["exact_match"]["passed"])
            rule_passes = sum(1 for r in results if r["grading"]["rule_based"]["passed"])
            judge_passes = sum(1 for r in results if r["grading"]["llm_judge"]["passed"])

            # By category
            by_category = {}
            for result in results:
                cat = result["category"]
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(result)

            category_stats = {}
            for cat, cat_results in by_category.items():
                cat_exact = sum(1 for r in cat_results if r["grading"]["exact_match"]["passed"])
                category_stats[cat] = {
                    "total": len(cat_results),
                    "exact_match_pass": cat_exact,
                    "exact_match_rate": f"{(cat_exact / len(cat_results) * 100):.1f}%"
                }

            report["by_model"][model] = {
                "total_cases": len(results),
                "exact_match": {
                    "pass": exact_passes,
                    "fail": len(results) - exact_passes,
                    "pass_rate": f"{(exact_passes / len(results) * 100):.1f}%"
                },
                "rule_based": {
                    "pass": rule_passes,
                    "fail": len(results) - rule_passes,
                    "pass_rate": f"{(rule_passes / len(results) * 100):.1f}%"
                },
                "llm_judge": {
                    "pass": judge_passes,
                    "fail": len(results) - judge_passes,
                    "pass_rate": f"{(judge_passes / len(results) * 100):.1f}%"
                },
                "by_category": category_stats
            }

        self.report = report
        return report

    def save_results(self):
        """Save detailed results and report to JSON."""
        # Save raw results
        results_file = os.path.join(self.output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved detailed results to {results_file}")

        # Save report
        if hasattr(self, 'report'):
            report_file = os.path.join(self.output_dir, "report.json")
            with open(report_file, 'w') as f:
                json.dump(self.report, f, indent=2)
            print(f"Saved report to {report_file}")

    def print_report(self):
        """Print a human-readable report."""
        if not hasattr(self, 'report'):
            self.generate_report()

        print("\n" + "="*70)
        print("EVALFORGE EVALUATION REPORT")
        print("="*70)
        print(f"Timestamp: {self.report['timestamp']}")
        print(f"Total Cases: {self.report['total_cases']}")
        print(f"Models: {', '.join(self.report['models_evaluated'])}")

        for model, stats in self.report["by_model"].items():
            print(f"\n{'-'*70}")
            print(f"Model: {model.upper()}")
            print(f"{'-'*70}")
            print(f"  Exact Match: {stats['exact_match']['pass']}/{stats['exact_match']['pass_rate']}")
            print(f"  Rule-Based:  {stats['rule_based']['pass']}/{stats['rule_based']['pass_rate']}")
            print(f"  LLM Judge:   {stats['llm_judge']['pass']}/{stats['llm_judge']['pass_rate']}")

            # Show latency if available
            model_results = [r for r in self.results if r["model"] == model]
            if model_results:
                latencies = [r.get("latency_ms", 0) for r in model_results]
                avg_latency = sum(latencies) / len(latencies)
                print(f"  Avg Latency: {avg_latency:.0f}ms")

            if stats.get("by_category"):
                print(f"\n  By Category:")
                for cat, cat_stats in stats["by_category"].items():
                    print(f"    {cat:20} {cat_stats['exact_match_pass']}/{cat_stats['total']} ({cat_stats['exact_match_rate']})")

        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    runner = EvalForgeRunner("datasets/questions.json")
    runner.load_dataset()
    runner.run_evaluation(models=["claude", "gpt"])
    runner.generate_report()
    runner.print_report()
    runner.save_results()
