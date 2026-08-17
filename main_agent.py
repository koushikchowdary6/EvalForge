#!/usr/bin/env python3
"""
EvalForge v0.4 - Agent / tool-call evaluation entry point.

Usage:
    python main_agent.py                  # all 24 tasks, Claude
    python main_agent.py --limit 3        # smoke test (cheap)
    python main_agent.py --model gpt      # evaluate GPT-4o-mini
    python main_agent.py --category injection_resistance

Run with --limit first to confirm wiring before a full run.
"""

import argparse

from evalforge.agent_runner import AgentEvalRunner


def main():
    parser = argparse.ArgumentParser(description="EvalForge agent evaluation")
    parser.add_argument("--model", default="claude", choices=["claude", "gpt"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N tasks (cost control)")
    parser.add_argument("--category", default=None,
                        help="Run only tasks in this category")
    parser.add_argument("--no-defense", action="store_true",
                        help="Ablation: remove the injection-defense instruction "
                             "from the system prompt to measure its effect")
    args = parser.parse_args()

    print("=" * 72)
    print("EVALFORGE v0.4 - AGENT EVALUATION")
    print("=" * 72)

    runner = AgentEvalRunner(
        tasks_path="datasets/agent_tasks.json",
        output_dir="results",
        model=args.model,
        defense_enabled=not args.no_defense,
    )
    runner.load()

    if args.category:
        runner.tasks = [t for t in runner.tasks if t.get("category") == args.category]
        print(f"[filter] category={args.category} -> {len(runner.tasks)} tasks")

    if args.limit:
        runner.tasks = runner.tasks[:args.limit]
        print(f"[limit] first {args.limit} tasks only")

    runner.run()
    runner.generate_report()
    runner.print_report()
    runner.save()

    print("Done. See results/agent_report.json")


if __name__ == "__main__":
    main()
