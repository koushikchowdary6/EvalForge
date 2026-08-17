#!/usr/bin/env python3
"""
EvalForge v0.3 - RAG Evaluation entry point.

Usage:
    python main_rag.py                      # full run, top_k sweep 1/3/5, Claude
    python main_rag.py --limit 3            # smoke test on first 3 questions
    python main_rag.py --model gpt          # generate answers with GPT-4o-mini
    python main_rag.py --k 3                # single top_k setting

Run with --limit first. It costs a few cents and confirms the wiring
before the full sweep.
"""

import argparse

from evalforge.rag_runner import RAGEvalRunner


def main():
    parser = argparse.ArgumentParser(description="EvalForge RAG evaluation")
    parser.add_argument("--model", default="claude", choices=["claude", "gpt"],
                        help="Model that generates answers (default: claude)")
    parser.add_argument("--judge", default="claude", choices=["claude", "gpt"],
                        help="Model that grades answers (default: claude)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5],
                        help="top_k values to sweep (default: 1 3 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (cost control)")
    args = parser.parse_args()

    print("=" * 72)
    print("EVALFORGE v0.3 - RAG EVALUATION")
    print("=" * 72)

    runner = RAGEvalRunner(
        corpus_path="datasets/rag_corpus.json",
        questions_path="datasets/rag_questions.json",
        output_dir="results",
        model=args.model,
        judge_model=args.judge,
    )

    runner.load()

    if args.limit:
        runner.questions = runner.questions[:args.limit]
        print(f"[limit] Evaluating first {args.limit} questions only")

    runner.run(k_values=tuple(args.k))
    runner.generate_report()
    runner.print_report()
    runner.save()

    print("Done. See results/rag_report.json")


if __name__ == "__main__":
    main()
