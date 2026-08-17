"""
Agent evaluation runner.

Runs every task through the agent, grades the resulting trajectory, and
aggregates task success, tool-call accuracy, safety behaviour, latency,
and estimated cost - broken down by task category so failures are
attributable rather than averaged away.
"""

import json
import os
from collections import Counter
from datetime import datetime

from evalforge.agent import ToolCallingAgent
from evalforge.agent_graders import (
    grade_trajectory, injection_resisted, refused_destructive, estimate_cost,
)


class AgentEvalRunner:
    """Runs agent evaluation over a task dataset."""

    def __init__(self, tasks_path, output_dir="results", model="claude",
                 defense_enabled=True):
        self.tasks_path = tasks_path
        self.output_dir = output_dir
        self.model = model
        self.defense_enabled = defense_enabled
        self.tasks = []
        self.results = []
        self.report = None
        os.makedirs(output_dir, exist_ok=True)

    def load(self):
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            self.tasks = json.load(f)
        print(f"Loaded {len(self.tasks)} agent tasks from {self.tasks_path}")
        return self.tasks

    def run(self):
        agent = ToolCallingAgent(model=self.model,
                                defense_enabled=self.defense_enabled)
        defense_label = "on" if self.defense_enabled else "OFF (ablation)"
        print(f"\nRunning {len(self.tasks)} tasks against '{self.model}' "
              f"| injection defense: {defense_label}\n")

        for task in self.tasks:
            trajectory = agent.run(task["prompt"], task.get("mock_responses"))
            grade = grade_trajectory(task, trajectory)

            self.results.append({
                "case_id": task["id"],
                "category": task.get("category"),
                "model": self.model,
                "prompt": task["prompt"],
                "tool_calls": trajectory.get("tool_calls", []),
                "final_answer": trajectory.get("final_answer", ""),
                "grade": grade,
                "injection_resisted": injection_resisted(task, trajectory),
                "refused_destructive": refused_destructive(task, trajectory),
                "turns": trajectory.get("turns", 0),
                "latency_ms": trajectory.get("latency_ms", 0),
                "tokens": trajectory.get("tokens", {}),
                "cost_usd": estimate_cost(trajectory.get("tokens", {}), self.model),
                "error": trajectory.get("error"),
                "timestamp": datetime.now().isoformat(),
            })

            flag = "PASS" if grade["success"] else "FAIL"
            called = ",".join(c["tool"] for c in trajectory.get("tool_calls", [])) or "-"
            note = f" [{grade['failure_category']}]" if grade["failure_category"] else ""
            print(f"  {flag} {task['id']:11} {task.get('category',''):22} "
                  f"tools={called:32.32}{note}")

        print(f"\nComplete. {len(self.results)} trajectories.")
        return self.results

    @staticmethod
    def _rate(values):
        vals = [v for v in values if v is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    @staticmethod
    def _mean(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    def generate_report(self):
        by_category = {}
        for r in self.results:
            by_category.setdefault(r["category"], []).append(r)

        failure_counts = Counter(
            r["grade"]["failure_category"] for r in self.results
            if r["grade"]["failure_category"]
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "injection_defense_enabled": self.defense_enabled,
            "total_tasks": len(self.results),
            "overall": {
                "task_success_rate": self._rate([r["grade"]["success"] for r in self.results]),
                "tool_selection_accuracy": self._mean(
                    [r["grade"]["tool_selection_score"] for r in self.results]),
                "argument_accuracy": self._mean(
                    [r["grade"]["argument_accuracy"] for r in self.results]),
                "injection_resistance_rate": self._rate(
                    [r["injection_resisted"] for r in self.results]),
                "destructive_refusal_rate": self._rate(
                    [r["refused_destructive"] for r in self.results]),
                "unnecessary_call_rate": self._rate(
                    [bool(r["grade"]["unnecessary_calls"]) for r in self.results]),
                "avg_latency_ms": self._mean([r["latency_ms"] for r in self.results]),
                "avg_turns": self._mean([r["turns"] for r in self.results]),
                "total_cost_usd": sum(r["cost_usd"] or 0 for r in self.results),
            },
            "failure_categories": dict(failure_counts),
            "by_category": {},
        }

        for category, rows in sorted(by_category.items()):
            report["by_category"][category] = {
                "tasks": len(rows),
                "success_rate": self._rate([r["grade"]["success"] for r in rows]),
                "tool_selection_accuracy": self._mean(
                    [r["grade"]["tool_selection_score"] for r in rows]),
                "avg_latency_ms": self._mean([r["latency_ms"] for r in rows]),
            }

        self.report = report
        return report

    def print_report(self):
        if not self.report:
            self.generate_report()

        def pct(value):
            return "n/a" if value is None else f"{value * 100:.1f}%"

        o = self.report["overall"]
        print("\n" + "=" * 72)
        print("EVALFORGE v0.4 - AGENT EVALUATION REPORT")
        print("=" * 72)
        print(f"Model: {self.report['model']}    Tasks: {self.report['total_tasks']}"
              f"    Injection defense: "
              f"{'on' if self.report['injection_defense_enabled'] else 'OFF (ablation)'}")
        print("-" * 72)
        print(f"  Task success rate .......... {pct(o['task_success_rate'])}")
        print(f"  Tool selection accuracy .... {pct(o['tool_selection_accuracy'])}")
        print(f"  Argument accuracy .......... {pct(o['argument_accuracy'])}")
        print(f"  Injection resistance ....... {pct(o['injection_resistance_rate'])}")
        print(f"  Destructive refusal ........ {pct(o['destructive_refusal_rate'])}")
        print(f"  Unnecessary call rate ...... {pct(o['unnecessary_call_rate'])}")
        print(f"  Avg latency ................ {o['avg_latency_ms']:.0f}ms")
        print(f"  Avg turns .................. {o['avg_turns']:.2f}")
        print(f"  Estimated cost ............. ${o['total_cost_usd']:.4f}")

        print("\n  By category:")
        print(f"    {'category':24}{'tasks':>7}{'success':>10}{'tool_acc':>10}")
        for category, stats in self.report["by_category"].items():
            print(f"    {category:24}{stats['tasks']:>7}"
                  f"{pct(stats['success_rate']):>10}"
                  f"{pct(stats['tool_selection_accuracy']):>10}")

        if self.report["failure_categories"]:
            print("\n  Failure breakdown:")
            for name, count in sorted(self.report["failure_categories"].items(),
                                      key=lambda kv: -kv[1]):
                print(f"    {name:28}{count:>4}")

        print("=" * 72 + "\n")

    def save(self):
        detail_path = os.path.join(self.output_dir, "agent_results.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved trajectories to {detail_path}")

        if self.report:
            report_path = os.path.join(self.output_dir, "agent_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.report, f, indent=2)
            print(f"Saved report to {report_path}")
