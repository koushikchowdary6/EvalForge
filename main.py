#!/usr/bin/env python3
"""
EvalForge Entry Point

Run this file to execute a full evaluation.
"""

from evalforge.runner import EvalForgeRunner


def main():
    print("="*70)
    print("EVALFORGE v0.1 - LLM Evaluation Platform")
    print("="*70)

    # Initialize runner with dataset
    runner = EvalForgeRunner(
        dataset_path="datasets/questions.json",
        output_dir="results"
    )

    # Load and evaluate
    runner.load_dataset()
    runner.run_evaluation(models=["claude"])  # add "gpt" back once that account has credits

    # Generate and print report
    runner.generate_report()
    runner.print_report()

    # Save results
    runner.save_results()

    print("✓ Evaluation complete! Check results/ folder for detailed output.")


if __name__ == "__main__":
    main()
