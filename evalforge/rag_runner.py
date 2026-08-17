"""
RAG evaluation runner.

Sweeps retrieval settings (top-k), runs the full retrieve-then-generate
pipeline over the question set, and scores retrieval and generation
separately so a failure can be attributed to the right stage.
"""

import json
import os
from datetime import datetime

from evalforge.rag import VectorStore, RAGPipeline, EmbeddingClient
from evalforge.rag_graders import (
    precision_at_k, recall_at_k, reciprocal_rank, hit_rate,
    grade_abstention, RAGJudge,
)


class RAGEvalRunner:
    """Runs RAG evaluation across retrieval configurations."""

    def __init__(self, corpus_path, questions_path,
                 output_dir="results", model="claude", judge_model="claude"):
        self.corpus_path = corpus_path
        self.questions_path = questions_path
        self.output_dir = output_dir
        self.model = model

        self.questions = []
        self.results = []
        self.report = None

        os.makedirs(output_dir, exist_ok=True)

        # One embedding client shared across all configs so the corpus is
        # embedded once and reused - otherwise a top-k sweep pays for
        # embeddings N times over.
        self.embedder = EmbeddingClient()
        self.store = VectorStore(self.embedder)
        self.judge = RAGJudge(judge_model=judge_model)

    def load(self):
        self.store.load_corpus(self.corpus_path)
        with open(self.questions_path, "r", encoding="utf-8") as f:
            self.questions = json.load(f)
        print(f"Loaded {len(self.questions)} RAG questions")
        self.store.build_index()

    def run(self, k_values=(1, 3, 5)):
        """Evaluate every question at each top-k setting."""
        pipeline = RAGPipeline(self.store, model=self.model)

        total = len(self.questions) * len(k_values)
        print(f"\nRunning {total} evaluations "
              f"({len(self.questions)} questions x {len(k_values)} k-values)\n")

        for k in k_values:
            print(f"--- top_k = {k} ---")
            for case in self.questions:
                outcome = pipeline.query(case["question"], top_k=k)
                retrieved_ids = [d["doc_id"] for d in outcome["retrieved"]]
                relevant_ids = case.get("relevant_doc_ids", [])
                answerable = case.get("answerable", True)

                retrieval_scores = {
                    "precision_at_k": precision_at_k(retrieved_ids, relevant_ids),
                    "recall_at_k": recall_at_k(retrieved_ids, relevant_ids),
                    "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
                    "hit_rate": hit_rate(retrieved_ids, relevant_ids),
                }

                abstention = grade_abstention(outcome["answer"], answerable)

                # Only judge answers the model actually attempted. Grading a
                # correct abstention for correctness would penalise the right
                # behaviour, so those are scored by abstention alone.
                if answerable and not abstention["abstained"]:
                    correctness = self.judge.grade_correctness(
                        case["question"], case["expected_answer"], outcome["answer"])
                    groundedness = self.judge.grade_groundedness(
                        outcome["answer"], outcome["retrieved"])
                else:
                    correctness = {"metric": "correctness", "score": None,
                                   "passed": None, "reason": "skipped (abstention case)"}
                    groundedness = {"metric": "groundedness", "score": None,
                                    "passed": None, "reason": "skipped (abstention case)"}

                self.results.append({
                    "case_id": case["id"],
                    "top_k": k,
                    "model": self.model,
                    "question": case["question"],
                    "answerable": answerable,
                    "expected": case["expected_answer"],
                    "answer": outcome["answer"],
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant_ids,
                    "retrieval": retrieval_scores,
                    "abstention": abstention,
                    "correctness": correctness,
                    "groundedness": groundedness,
                    "latency": {
                        "retrieval_ms": outcome["retrieval_ms"],
                        "generation_ms": outcome["generation_ms"],
                        "total_ms": outcome["total_ms"],
                    },
                    "timestamp": datetime.now().isoformat(),
                })

                flag = "OK " if abstention["passed"] else "FAIL"
                print(f"  {flag} {case['id']:9} | hit={retrieval_scores['hit_rate']:.0f} "
                      f"| {outcome['total_ms']:.0f}ms")

        print(f"\nComplete. {len(self.results)} evaluations.")
        return self.results

    @staticmethod
    def _mean(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    def generate_report(self):
        """Aggregate results per top-k setting."""
        by_k = {}
        for r in self.results:
            by_k.setdefault(r["top_k"], []).append(r)

        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "total_questions": len(self.questions),
            "total_evaluations": len(self.results),
            "embedding_api_calls": self.embedder.api_calls,
            "embedding_cache_hits": self.embedder.cache_hits,
            "by_top_k": {},
        }

        for k, rows in sorted(by_k.items()):
            answerable = [r for r in rows if r["answerable"]]
            unanswerable = [r for r in rows if not r["answerable"]]
            hallucinations = [r for r in unanswerable
                              if r["abstention"]["outcome"] == "hallucination"]
            over_abstentions = [r for r in answerable
                                if r["abstention"]["outcome"] == "over_abstention"]

            report["by_top_k"][str(k)] = {
                "retrieval": {
                    "precision_at_k": self._mean([r["retrieval"]["precision_at_k"] for r in rows]),
                    "recall_at_k": self._mean([r["retrieval"]["recall_at_k"] for r in rows]),
                    "mrr": self._mean([r["retrieval"]["reciprocal_rank"] for r in rows]),
                    "hit_rate": self._mean([r["retrieval"]["hit_rate"] for r in rows]),
                },
                "generation": {
                    "correctness": self._mean([r["correctness"]["score"] for r in rows]),
                    "groundedness": self._mean([r["groundedness"]["score"] for r in rows]),
                },
                "hallucination": {
                    "unanswerable_questions": len(unanswerable),
                    "hallucinations": len(hallucinations),
                    "hallucination_rate": (len(hallucinations) / len(unanswerable)
                                           if unanswerable else None),
                    "over_abstentions": len(over_abstentions),
                    "over_abstention_rate": (len(over_abstentions) / len(answerable)
                                             if answerable else None),
                },
                "latency": {
                    "avg_retrieval_ms": self._mean([r["latency"]["retrieval_ms"] for r in rows]),
                    "avg_generation_ms": self._mean([r["latency"]["generation_ms"] for r in rows]),
                    "avg_total_ms": self._mean([r["latency"]["total_ms"] for r in rows]),
                },
            }

        self.report = report
        return report

    def print_report(self):
        if not self.report:
            self.generate_report()

        def fmt(value, pct=False):
            if value is None:
                return "n/a"
            return f"{value * 100:.1f}%" if pct else f"{value:.3f}"

        print("\n" + "=" * 72)
        print("EVALFORGE v0.3 - RAG EVALUATION REPORT")
        print("=" * 72)
        print(f"Model: {self.report['model']}   "
              f"Questions: {self.report['total_questions']}   "
              f"Evaluations: {self.report['total_evaluations']}")
        print(f"Embedding calls: {self.report['embedding_api_calls']} "
              f"(cache hits: {self.report['embedding_cache_hits']})")

        header = (f"\n{'top_k':<6}{'P@k':>8}{'R@k':>8}{'MRR':>8}{'Hit':>8}"
                  f"{'Correct':>10}{'Ground':>9}{'Halluc':>9}{'ms':>8}")
        print(header)
        print("-" * 72)

        for k, stats in self.report["by_top_k"].items():
            r, g = stats["retrieval"], stats["generation"]
            h, lat = stats["hallucination"], stats["latency"]
            print(f"{k:<6}"
                  f"{fmt(r['precision_at_k']):>8}"
                  f"{fmt(r['recall_at_k']):>8}"
                  f"{fmt(r['mrr']):>8}"
                  f"{fmt(r['hit_rate']):>8}"
                  f"{fmt(g['correctness']):>10}"
                  f"{fmt(g['groundedness']):>9}"
                  f"{fmt(h['hallucination_rate'], pct=True):>9}"
                  f"{lat['avg_total_ms']:>8.0f}")

        print("=" * 72 + "\n")

    def save(self):
        detail_path = os.path.join(self.output_dir, "rag_results.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved detailed results to {detail_path}")

        if self.report:
            report_path = os.path.join(self.output_dir, "rag_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.report, f, indent=2)
            print(f"Saved report to {report_path}")
