"""
Agent evaluation runner.

Runs every task through the agent, grades the resulting trajectory, and
aggregates task success, tool-call accuracy, safety behaviour, latency,
and estimated cost - broken down by task category so failures are
attributable rather than averaged away.
"""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime

from evalforge.artifacts import atomic_json_dump
from evalforge.identifiers import valid_benchmark_id
from evalforge.agent_tools import TOOL_REGISTRY, has_visible_argument_text
from evalforge.agent import ToolCallingAgent
from evalforge.agent_graders import (
    grade_trajectory, injection_resisted, refused_destructive, estimate_cost,
)


def _reject_duplicate_json_keys(pairs):
    """Reject ambiguous benchmark labels, including nested tool arguments.

    Python's default JSON decoder silently keeps the last occurrence of a
    duplicate key, allowing case identity or expected tool arguments to change
    without an explicit dataset edit being visible to reviewers.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


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
        if not valid_benchmark_id(task_id):
            raise ValueError(f"{location} must have a non-empty string id")
        if task_id in seen_ids:
            raise ValueError(f"duplicate agent task id: {task_id}")
        seen_ids.add(task_id)

        if not has_visible_argument_text(task.get("prompt")):
            raise ValueError(f"{task_id}: prompt must be a non-empty string")
        if "category" in task and (
            not has_visible_argument_text(task["category"])
        ):
            raise ValueError(f"{task_id}: category must be a non-empty string")

        for field in ("expected_tools", "forbidden_tools", "answer_must_contain"):
            values = task.get(field, [])
            if not isinstance(values, list) or any(
                not has_visible_argument_text(value) for value in values
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
            not isinstance(name, str) or not (
                isinstance(args, dict)
                or (isinstance(args, list) and bool(args)
                    and all(isinstance(item, dict) for item in args))
            )
            for name, args in expected_args.items()
        ):
            raise ValueError(
                f"{task_id}: expected_args must map tools to argument objects "
                "or non-empty lists of argument objects"
            )
        if set(expected_args) - expected:
            raise ValueError(f"{task_id}: expected_args references an unexpected tool")
        expected_counts = Counter(task.get("expected_tools", []))
        for name, args in expected_args.items():
            if isinstance(args, list) and len(args) != expected_counts[name]:
                raise ValueError(
                    f"{task_id}: expected_args for {name} must have one "
                    "argument object per required invocation"
                )
            # An expected argument name that the tool does not accept can
            # never be supplied by a valid provider call. Previously such
            # benchmark labels were silently graded as model failures.
            # Partial expectations are allowed, but every supplied field
            # must be a meaningful string parameter from the tool schema.
            properties = TOOL_REGISTRY[name]["parameters"]["properties"]
            for required in (args if isinstance(args, list) else [args]):
                if not required or any(
                    not isinstance(key, str) or key not in properties
                    or properties[key].get("type") != "string"
                    or not has_visible_argument_text(value)
                    for key, value in required.items()
                ):
                    raise ValueError(
                        f"{task_id}: expected_args for {name} contains "
                        "unknown, empty, or non-string tool parameters"
                    )

        responses = task.get("mock_responses", {})
        if not isinstance(responses, dict) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in responses.items()
        ):
            raise ValueError(f"{task_id}: mock_responses must map tools to strings")
        unknown_mocks = sorted(set(responses) - TOOL_REGISTRY.keys())
        if unknown_mocks:
            raise ValueError(f"{task_id}: mock_responses contains unknown tools: {unknown_mocks}")


def _agent_dataset_sha256(tasks):
    """Hash parsed benchmark cases, ignoring JSON key order and formatting.

    Case order, prompts, labels, and mocked tool responses remain part of
    the fingerprint. The digest is an identity marker, not proof of quality.
    ASCII escaping also supports JSON-escaped Unicode surrogate metadata.
    """
    canonical = json.dumps(tasks, sort_keys=True, ensure_ascii=True,
                           allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        self.tasks_sha256 = None
        os.makedirs(output_dir, exist_ok=True)

    def load(self):
        # Never retain a previous valid dataset if reloading fails validation.
        self.tasks = []
        self.results = []
        self.report = None
        self.tasks_sha256 = None
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            candidate = json.load(f, object_pairs_hook=_reject_duplicate_json_keys)
        validate_agent_tasks(candidate)
        fingerprint = _agent_dataset_sha256(candidate)
        self.tasks = candidate
        self.tasks_sha256 = fingerprint
        print(f"Loaded {len(self.tasks)} agent tasks from {self.tasks_path}")
        return self.tasks

    def run(self):
        # Invalidate all prior measurements before validation. A failed
        # revalidation must not leave a previous sweep's report or trajectories
        # available to save under a missing/stale dataset fingerprint.
        self.results = []
        self.report = None
        self.tasks_sha256 = None
        validate_agent_tasks(self.tasks)
        self.tasks_sha256 = _agent_dataset_sha256(self.tasks)

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

    @staticmethod
    def _pooled_tool_metrics(rows):
        """Micro-average observed tool calls, excluding provider outages.

        Unlike the per-case macro mean, each actual/required invocation
        contributes one count. No-tool cases contribute zero calls; an entirely
        no-tool benchmark still reports perfect precision and recall.
        """
        counts = [
            row["grade"].get("tool_call_counts")
            for row in rows if not row.get("error")
            and row["grade"].get("tool_call_counts") is not None
        ]
        matched = sum(item["matched"] for item in counts)
        predicted = sum(item["predicted"] for item in counts)
        required = sum(item["required"] for item in counts)
        return {
            "precision": (
                matched / predicted if predicted else (1.0 if not required else 0.0)
            ) if counts else None,
            "recall": (matched / required if required else 1.0) if counts else None,
            "samples": len(counts),
            "matched": matched,
            "predicted": predicted,
            "required": required,
        }

    def generate_report(self):
        by_category = {}
        for r in self.results:
            by_category.setdefault(r.get("category") or "uncategorized", []).append(r)

        evaluated = [r for r in self.results if not r.get("error")]
        # Do not present a partial or unpriced sweep as a zero-dollar total.
        # Retain observed spend separately, with explicit coverage.
        known_costs = [
            r["cost_usd"] for r in self.results if r.get("cost_usd") is not None
        ]

        failure_counts = Counter(
            r["grade"]["failure_category"] for r in self.results
            if r["grade"]["failure_category"]
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "injection_defense_enabled": self.defense_enabled,
            "benchmark_provenance": {"agent_tasks_sha256": self.tasks_sha256},
            "total_tasks": len(self.results),
            "evaluated_tasks": len(evaluated),
            "execution_errors": len(self.results) - len(evaluated),
            # Safety attempts with observed forbidden actions count as failures
            # even if the provider later errors. Unobserved outages stay out
            # of the rate denominator; expose coverage to prevent overclaiming.
            "injection_resistance_evaluated": sum(
                r["injection_resisted"] is not None for r in self.results
            ),
            "destructive_refusal_evaluated": sum(
                r["refused_destructive"] is not None for r in self.results
            ),
            "overall": {
                "task_success_rate": self._rate([r["grade"]["success"] for r in evaluated]),
                "tool_selection_accuracy": self._mean(
                    [r["grade"]["tool_selection_score"] for r in evaluated]),
                "tool_call_precision": self._mean(
                    [r["grade"].get("tool_call_precision") for r in evaluated]),
                "tool_call_recall": self._mean(
                    [r["grade"].get("tool_call_recall") for r in evaluated]),
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
                "latency_samples": len(evaluated),
                "turn_samples": len(evaluated),
                # Failed requests often exit quickly and at turn zero. Mixing
                # them into completed-run means rewards provider outages.
                "avg_latency_ms": self._mean([r["latency_ms"] for r in evaluated]),
                "avg_turns": self._mean([r["turns"] for r in evaluated]),
                "cost_samples": len(known_costs),
                "observed_cost_usd": sum(known_costs),
                "total_cost_usd": (
                    sum(known_costs) if len(known_costs) == len(self.results) else None
                ),
            },
            "failure_categories": dict(failure_counts),
            "by_category": {},
        }

        pooled = self._pooled_tool_metrics(evaluated)
        report["overall"].update({
            "tool_call_micro_precision": pooled["precision"],
            "tool_call_micro_recall": pooled["recall"],
            "tool_call_micro_samples": pooled["samples"],
            "tool_call_micro_matched": pooled["matched"],
            "tool_call_micro_predicted": pooled["predicted"],
            "tool_call_micro_required": pooled["required"],
        })

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
                "tool_call_precision": self._mean(
                    [r["grade"].get("tool_call_precision") for r in valid_rows]),
                "tool_call_recall": self._mean(
                    [r["grade"].get("tool_call_recall") for r in valid_rows]),
                "tool_sequence_accuracy": self._mean(
                    [r["grade"].get("tool_sequence_score") for r in valid_rows]),
                "latency_samples": len(valid_rows),
                "avg_latency_ms": self._mean([r["latency_ms"] for r in valid_rows]),
            }
            category_pooled = self._pooled_tool_metrics(valid_rows)
            report["by_category"][category].update({
                "tool_call_micro_precision": category_pooled["precision"],
                "tool_call_micro_recall": category_pooled["recall"],
                "tool_call_micro_samples": category_pooled["samples"],
                "tool_call_micro_matched": category_pooled["matched"],
                "tool_call_micro_predicted": category_pooled["predicted"],
                "tool_call_micro_required": category_pooled["required"],
            })

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
        print("  Dataset SHA256 ............. "
              + (self.report["benchmark_provenance"]["agent_tasks_sha256"] or "n/a"))
        print(f"  Evaluated tasks ............ {self.report['evaluated_tasks']}/{self.report['total_tasks']}")
        print(f"  Execution errors ........... {self.report['execution_errors']}")
        print(f"  Latency samples ............ {o['latency_samples']}/{self.report['total_tasks']}")
        print(f"  Injection safety observed .. {self.report['injection_resistance_evaluated']}")
        print(f"  Destructive safety observed  {self.report['destructive_refusal_evaluated']}")
        print(f"  Task success rate .......... {pct(o['task_success_rate'])}")
        print(f"  Tool selection accuracy .... {pct(o['tool_selection_accuracy'])}")
        print(f"  Tool-call precision ........ {pct(o['tool_call_precision'])}")
        print(f"  Tool-call recall ........... {pct(o['tool_call_recall'])}")
        print(f"  Micro tool precision ....... {pct(o['tool_call_micro_precision'])}")
        print(f"  Micro tool recall .......... {pct(o['tool_call_micro_recall'])}")
        print(f"  Pooled matched/pred/req .... {o['tool_call_micro_matched']}/"
              f"{o['tool_call_micro_predicted']}/{o['tool_call_micro_required']}"
              f" ({o['tool_call_micro_samples']} evaluated cases)")
        print(f"  Tool sequence accuracy ..... {pct(o['tool_sequence_accuracy'])}")
        print(f"  Argument accuracy .......... {pct(o['argument_accuracy'])}")
        print(f"  Injection resistance ....... {pct(o['injection_resistance_rate'])}")
        print(f"  Destructive refusal ........ {pct(o['destructive_refusal_rate'])}")
        print(f"  Unnecessary call rate ...... {pct(o['unnecessary_call_rate'])}")
        avg_latency = "n/a" if o["avg_latency_ms"] is None else f"{o['avg_latency_ms']:.0f}ms"
        avg_turns = "n/a" if o["avg_turns"] is None else f"{o['avg_turns']:.2f}"
        print(f"  Avg latency ................ {avg_latency}")
        print(f"  Avg turns .................. {avg_turns}")
        total_cost = o["total_cost_usd"]
        cost_text = f"${total_cost:.4f}" if total_cost is not None else "n/a (incomplete usage)"
        print(f"  Estimated cost ............. {cost_text}")
        print(f"  Cost coverage .............. {o['cost_samples']}/{self.report['total_tasks']}")
        if total_cost is None:
            print(f"  Observed partial cost ...... ${o['observed_cost_usd']:.4f}")

        print("\n  By category:")
        print(f"    {'category':24}{'tasks':>7}{'success':>10}{'tool_acc':>10}{'precision':>10}{'recall':>10}{'seq_acc':>10}")
        for category, stats in self.report["by_category"].items():
            print(f"    {category:24}{stats['tasks']:>7}"
                  f"{pct(stats['success_rate']):>10}"
                  f"{pct(stats['tool_selection_accuracy']):>10}"
                  f"{pct(stats['tool_call_precision']):>10}"
                  f"{pct(stats['tool_call_recall']):>10}"
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
