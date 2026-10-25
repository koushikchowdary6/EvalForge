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

from evalforge.artifacts import atomic_json_dump
from evalforge.agent_tools import TOOL_REGISTRY
from evalforge.agent import ToolCallingAgent
from evalforge.agent_graders import (
    grade_trajectory, injection_resisted, refused_destructive, estimate_cost,
)


def validate_agent_tasks(tasks):
    """Reject invalid benchmark labels before making any model/API calls.

    A malformed case can silently inflate safety metrics (for example, an
    injection-resistance task with no forbidden tools always 'resists') or
    crash halfway through an expensive evaluation sweep.
    """
    if not isinstance(tasks, list):
        raise ValueError("agent dataset must be a JSON list of tasks")

    seen_ids = set()
    for index, task in enumerate(tasks):
        location = f"agent task at index {index}"
        if not isinstance(task, dict):
            raise ValueError(f"{location} must be an object")

        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"{location} must have a non-empty string id")
        if task_id in seen_ids:
            raise ValueError(f"duplicate agent task id: {task_id}")
        seen_ids.add(task_id)

        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError(f"{task_id}: prompt must be a non-empty string")
        if "category" in task and (
            not isinstance(task["category"], str) or not task["category"].strip()
        ):
            raise ValueError(f"{task_id}: category must be a non-empty string")

        for field in ("expected_tools", "forbidden_tools", "answer_must_contain"):
            values = task.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{task_id}: {field} must be a list of non-empty strings")
            # Expected tool calls are an ordered sequence: the same tool may
            # legitimately be required more than once (for example, two
            # independent lookups). Sequence and extra-call graders already
            # account for multiplicity. Forbidden tools remain a set.
            if field in ("expected_tools", "forbidden_tools"):
                unknown = sorted(set(values) - TOOL_REGISTRY.keys())
                if unknown:
                    raise ValueError(f"{task_id}: {field} contains unknown tools: {unknown}")
            if field == "forbidden_tools" and len(set(values)) != len(values):
                raise ValueError(f"{task_id}: duplicate tool in {field}")

        expected = set(task.get("expected_tools", []))
        forbidden = set(task.get("forbidden_tools", []))
        if expected & forbidden:
            raise ValueError(f"{task_id}: expected and forbidden tools overlap")
        if task.get("category") in ("injection_resistance", "destructive_refusal") and not forbidden:
            raise ValueError(f"{task_id}: safety evaluation requires forbidden_tools")

        expected_args = task.get("expected_args", {})
        if not isinstance(expected_args, dict) or any(
            not isinstance(name, str) or not isinstance(args, dict)
            for name, args in expected_args.items()
        ):
            raise ValueError(f"{task_id}: expected_args must map tools to argument objects")
        if set(expected_args) - expected:
            raise ValueError(f"{task_id}: expected_args references an unexpected tool")

        responses = task.get("mock_responses", {})
        if not isinstance(responses, dict) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in responses.items()
        ):
            raise ValueError(f"{task_id}: mock_responses must map tools to strings")
        unknown_mocks = sorted(set(responses) - TOOL_REGISTRY.keys())
        if unknown_mocks:
            raise ValueError(f"{task_id}: mock_responses contains unknown tools: {unknown_mocks}")


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
        # Never retain a previous valid dataset if reloading fails validation.
        self.tasks = []
        self.results = []
        self.report = None
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            candidate = json.load(f)
        validate_agent_tasks(candidate)
        self.tasks = candidate
        print(f"Loaded {len(self.tasks)} agent tasks from {self.tasks_path}")
        return self.tasks

    def run(self):
        validate_agent_tasks(self.tasks)
        # Each run is an independent evaluation sweep. Reusing a runner should
        # reuse loaded tasks/configuration, not accumulate trajectories from a
        # previous experiment or expose a stale aggregate report.
        self.results = []
        self.report = None

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
                "category": task.get("category") or "uncategorized",
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
            by_category.setdefault(r.get("category") or "uncategorized", []).append(r)

        evaluated = [r for r in self.results if not r.get("error")]

        failure_counts = Counter(
            r["grade"]["failure_category"] for r in self.results
            if r["grade"]["failure_category"]
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "injection_defense_enabled": self.defense_enabled,
            "total_tasks": len(self.results),
            "evaluated_tasks": len(evaluated),
            "execution_errors": len(self.results) - len(evaluated),
            "overall": {
                "task_success_rate": self._rate([r["grade"]["success"] for r in evaluated]),
                "tool_selection_accuracy": self._mean(
                    [r["grade"]["tool_selection_score"] for r in evaluated]),
                "tool_sequence_accuracy": self._mean(
                    [r["grade"].get("tool_sequence_score") for r in evaluated]),
                "argument_accuracy": self._mean(
                    [r["grade"]["argument_accuracy"] for r in evaluated]),
                "injection_resistance_rate": self._rate(
                    [r["injection_resisted"] for r in self.results]),
                "destructive_refusal_rate": self._rate(
                    [r["refused_destructive"] for r in self.results]),
                "unnecessary_call_rate": self._rate(
                    [bool(r["grade"]["unnecessary_calls"]) for r in evaluated]),
                "avg_latency_ms": self._mean([r["latency_ms"] for r in self.results]),
                "avg_turns": self._mean([r["turns"] for r in self.results]),
                "total_cost_usd": sum(r["cost_usd"] or 0 for r in self.results),
            },
            "failure_categories": dict(failure_counts),
            "by_category": {},
        }

        for category, rows in sorted(by_category.items()):
            # Match the overall success-rate denominator: transport/provider
            # errors are not observed agent behavior. A category with only
            # execution errors must report n/a rather than a misleading 0%.
            valid_rows = [r for r in rows if not r.get("error")]
            report["by_category"][category] = {
                "tasks": len(rows),
                "evaluated_tasks": len(valid_rows),
                "execution_errors": len(rows) - len(valid_rows),
                "success_rate": self._rate(
                    [r["grade"]["success"] for r in valid_rows]),
                "tool_selection_accuracy": self._mean(
                    [r["grade"]["tool_selection_score"] for r in valid_rows]),
                "tool_sequence_accuracy": self._mean(
                    [r["grade"].get("tool_sequence_score") for r in valid_rows]),
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
        print(f"  Evaluated tasks ............ {self.report['evaluated_tasks']}/{self.report['total_tasks']}")
        print(f"  Execution errors ........... {self.report['execution_errors']}")
        print(f"  Task success rate .......... {pct(o['task_success_rate'])}")
        print(f"  Tool selection accuracy .... {pct(o['tool_selection_accuracy'])}")
        print(f"  Tool sequence accuracy ..... {pct(o['tool_sequence_accuracy'])}")
        print(f"  Argument accuracy .......... {pct(o['argument_accuracy'])}")
        print(f"  Injection resistance ....... {pct(o['injection_resistance_rate'])}")
        print(f"  Destructive refusal ........ {pct(o['destructive_refusal_rate'])}")
        print(f"  Unnecessary call rate ...... {pct(o['unnecessary_call_rate'])}")
        avg_latency = "n/a" if o["avg_latency_ms"] is None else f"{o['avg_latency_ms']:.0f}ms"
        avg_turns = "n/a" if o["avg_turns"] is None else f"{o['avg_turns']:.2f}"
        print(f"  Avg latency ................ {avg_latency}")
        print(f"  Avg turns .................. {avg_turns}")
        print(f"  Estimated cost ............. ${o['total_cost_usd']:.4f}")

        print("\n  By category:")
        print(f"    {'category':24}{'tasks':>7}{'success':>10}{'tool_acc':>10}{'seq_acc':>10}")
        for category, stats in self.report["by_category"].items():
            print(f"    {category:24}{stats['tasks']:>7}"
                  f"{pct(stats['success_rate']):>10}"
                  f"{pct(stats['tool_selection_accuracy']):>10}"
                  f"{pct(stats['tool_sequence_accuracy']):>10}")

        if self.report["failure_categories"]:
            print("\n  Failure breakdown:")
            for name, count in sorted(self.report["failure_categories"].items(),
                                      key=lambda kv: -kv[1]):
                print(f"    {name:28}{count:>4}")

        print("=" * 72 + "\n")

    def save(self):
        detail_path = os.path.join(self.output_dir, "agent_results.json")
        atomic_json_dump(detail_path, self.results)
        print(f"Saved trajectories to {detail_path}")

        report_path = os.path.join(self.output_dir, "agent_report.json")
        if self.report is not None:
            atomic_json_dump(report_path, self.report)
            print(f"Saved report to {report_path}")
        elif os.path.exists(report_path):
            # An incomplete or unreported sweep must not leave a stale
            # benchmark artifact that looks like the current run.
            os.remove(report_path)
