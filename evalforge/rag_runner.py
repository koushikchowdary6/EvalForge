"""
RAG evaluation runner.

Sweeps retrieval settings (top-k), runs the full retrieve-then-generate
pipeline over the question set, and scores retrieval and generation
separately so a failure can be attributed to the right stage.
"""

import json
import os
from datetime import datetime

from evalforge.artifacts import atomic_json_dump as _atomic_json_dump
from evalforge.rag import VectorStore, RAGPipeline, EmbeddingClient
from evalforge.rag_graders import (
    precision_at_k, recall_at_k, reciprocal_rank, hit_rate,
    grade_abstention, RAGJudge, ABSTAIN_TOKEN,
)


def validate_rag_questions(questions, document_ids):
    """Fail closed on invalid ground truth before any paid embedding requests.

    Bad references or duplicate case IDs silently distort retrieval metrics,
    while missing answerability labels can misclassify hallucinations.
    """
    if not isinstance(questions, list) or not questions:
        raise ValueError("RAG questions must be a non-empty list")
    seen_ids = set()
    for index, case in enumerate(questions):
        if not isinstance(case, dict):
            raise ValueError(f"RAG question #{index} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"RAG question #{index}: id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate RAG question id: {case_id}")
        seen_ids.add(case_id)
        for field in ("question", "expected_answer"):
            value = case.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RAG question {case_id}: {field} must be a non-empty string")
        if not isinstance(case.get("answerable"), bool):
            raise ValueError(f"RAG question {case_id}: answerable must be a boolean")
        relevant = case.get("relevant_doc_ids")
        if not isinstance(relevant, list) or any(
            not isinstance(doc_id, str) or not doc_id.strip() for doc_id in relevant
        ):
            raise ValueError(f"RAG question {case_id}: relevant_doc_ids must be a list of IDs")
        if len(relevant) != len(set(relevant)):
            raise ValueError(f"RAG question {case_id}: duplicate relevant_doc_ids")
        unknown = set(relevant) - document_ids
        if unknown:
            raise ValueError(f"RAG question {case_id}: unknown document IDs: {sorted(unknown)}")
        # Retrieval quality cannot be measured for an answerable question
        # without at least one labeled evidence document. Otherwise recall,
        # MRR and hit rate can appear perfect by empty-set convention.
        if case["answerable"] and not relevant:
            raise ValueError(
                f"RAG question {case_id}: answerable questions require relevant_doc_ids"
            )
        if not case["answerable"] and relevant:
            raise ValueError(
                f"RAG question {case_id}: unanswerable questions cannot label relevant_doc_ids"
            )
        expected_abstention = case["expected_answer"].strip().casefold() == ABSTAIN_TOKEN.casefold()
        if case["answerable"] and expected_abstention:
            raise ValueError(
                f"RAG question {case_id}: answerable question cannot expect abstention"
            )
        if not case["answerable"] and not expected_abstention:
            raise ValueError(
                f"RAG question {case_id}: unanswerable question must expect {ABSTAIN_TOKEN}"
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
        self.questions = []
        with open(self.questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        validate_rag_questions(
            questions, {doc["doc_id"] for doc in self.store.documents}
        )
        self.questions = questions
        print(f"Loaded {len(self.questions)} RAG questions")
        self.store.build_index()

    def run(self, k_values=(1, 3, 5)):
        """Evaluate every question at each top-k setting."""
        # A second sweep must not mix prior results with fresh measurements.
        # Keep the store/embedding cache reusable across sweeps.
        self.results = []
        self.report = None
        if (not isinstance(k_values, (list, tuple)) or not k_values
                or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                       for k in k_values)
                or len(k_values) != len(set(k_values))):
            raise ValueError("k_values must be unique positive integers")
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

                retrieval_error = outcome.get("error_stage") == "retrieval"
                # A failed retrieval has no measured ranking. Recording zeros
                # would silently misrepresent an outage as model quality.
                retrieval_scores = (
                    {name: None for name in (
                        "precision_at_k", "recall_at_k", "reciprocal_rank", "hit_rate"
                    )}
                    if retrieval_error else {
                        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids),
                        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids),
                        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
                        "hit_rate": hit_rate(retrieved_ids, relevant_ids),
                    }
                )

                # A transport/provider failure is not a model answer. Keep its
                # retrieval measurements, but exclude it from answer-quality rates.
                generation_error = outcome.get("error")
                if generation_error:
                    abstention = {"outcome": "generation_error", "passed": None,
                                  "abstained": None, "reason": "generation failed"}
                else:
                    abstention = grade_abstention(outcome["answer"], answerable)

                # Only judge answers the model actually attempted. Grading a
                # correct abstention for correctness would penalise the right
                # behaviour, so those are scored by abstention alone.
                if (not generation_error and answerable and not abstention["abstained"]
                        and abstention["outcome"] != "empty_answer"):
                    correctness = self.judge.grade_correctness(
                        case["question"], case["expected_answer"], outcome["answer"])
                    groundedness = self.judge.grade_groundedness(
                        outcome["answer"], outcome["retrieved"])
                else:
                    skip_reason = (
                        "skipped (generation failed)" if generation_error
                        else "skipped (empty answer)" if abstention["outcome"] == "empty_answer"
                        else "skipped (abstention case)"
                    )
                    correctness = {"metric": "correctness", "score": None,
                                   "passed": None, "reason": skip_reason}
                    groundedness = {"metric": "groundedness", "score": None,
                                    "passed": None, "reason": skip_reason}

                self.results.append({
                    "case_id": case["id"],
                    "top_k": k,
                    "model": self.model,
                    "question": case["question"],
                    "answerable": answerable,
                    "expected": case["expected_answer"],
                    "answer": outcome["answer"],
                    "generation_error": generation_error if not retrieval_error else None,
                    "retrieval_error": outcome.get("error") if retrieval_error else None,
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

                flag = "ERR" if generation_error else ("OK " if abstention["passed"] else "FAIL")
                hit = retrieval_scores["hit_rate"]
                hit_text = f"{hit:.0f}" if hit is not None else "n/a"
                print(f"  {flag} {case['id']:9} | hit={hit_text} "
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
            retrieval_errors = [r for r in rows if r.get("retrieval_error")]
            attempted_generation = [r for r in rows if not r.get("retrieval_error")]
            successful = [r for r in attempted_generation if not r.get("generation_error")]
            generation_errors = len(attempted_generation) - len(successful)
            answerable = [r for r in successful if r["answerable"]]
            unanswerable = [r for r in successful if not r["answerable"]]
            empty_answers = [r for r in successful
                             if r["abstention"]["outcome"] == "empty_answer"]
            # Blank outputs are failed generations, but are neither factual
            # hallucinations nor explicit refusals. Keep them out of those
            # rates' denominators and report their prevalence separately.
            evaluated_unanswerable = [r for r in unanswerable
                                      if r["abstention"]["outcome"] != "empty_answer"]
            evaluated_answerable = [r for r in answerable
                                    if r["abstention"]["outcome"] != "empty_answer"]
            hallucinations = [r for r in unanswerable
                              if r["abstention"]["outcome"] == "hallucination"]
            over_abstentions = [r for r in answerable
                                if r["abstention"]["outcome"] == "over_abstention"]

            # No ground-truth relevant documents exist for unanswerable
            # queries. The pure graders use empty-set conventions, but those
            # synthetic perfect recall/MRR/hit scores must not inflate a
            # mixed benchmark's retrieval metrics.
            retrieval_rows = [r for r in rows
                              if r["answerable"] and not r.get("retrieval_error")]
            report["by_top_k"][str(k)] = {
                "retrieval": {
                    "evaluated_queries": len(retrieval_rows),
                    "excluded_unanswerable": sum(not r["answerable"] for r in rows),
                    "errors": len(retrieval_errors),
                    "precision_at_k": self._mean([r["retrieval"]["precision_at_k"] for r in retrieval_rows]),
                    "recall_at_k": self._mean([r["retrieval"]["recall_at_k"] for r in retrieval_rows]),
                    "mrr": self._mean([r["retrieval"]["reciprocal_rank"] for r in retrieval_rows]),
                    "hit_rate": self._mean([r["retrieval"]["hit_rate"] for r in retrieval_rows]),
                },
                "generation": {
                    "attempted": len(attempted_generation),
                    "successful": len(successful),
                    "errors": generation_errors,
                    "error_rate": (generation_errors / len(attempted_generation)
                                   if attempted_generation else None),
                    "empty_answers": len(empty_answers),
                    "empty_answer_rate": (len(empty_answers) / len(successful)
                                          if successful else None),
                    # Judge outages and malformed responses are not low-quality
                    # model answers. Track coverage separately from score.
                    "correctness_judged": sum(r["correctness"]["score"] is not None for r in rows),
                    "groundedness_judged": sum(r["groundedness"]["score"] is not None for r in rows),
                    "correctness_judge_errors": sum(r["correctness"].get("parse_ok") is False for r in rows),
                    "groundedness_judge_errors": sum(r["groundedness"].get("parse_ok") is False for r in rows),
                    "correctness": self._mean([r["correctness"]["score"] for r in rows]),
                    "groundedness": self._mean([r["groundedness"]["score"] for r in rows]),
                },
                "hallucination": {
                    "unanswerable_questions": len(unanswerable),
                    "hallucination_evaluated": len(evaluated_unanswerable),
                    "hallucinations": len(hallucinations),
                    "hallucination_rate": (len(hallucinations) / len(evaluated_unanswerable)
                                           if evaluated_unanswerable else None),
                    "over_abstention_evaluated": len(evaluated_answerable),
                    "over_abstentions": len(over_abstentions),
                    "over_abstention_rate": (len(over_abstentions) / len(evaluated_answerable)
                                             if evaluated_answerable else None),
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
        _atomic_json_dump(detail_path, self.results)
        print(f"Saved detailed results to {detail_path}")

        report_path = os.path.join(self.output_dir, "rag_report.json")
        if self.report is not None:
            _atomic_json_dump(report_path, self.report)
            print(f"Saved report to {report_path}")
        elif os.path.exists(report_path):
            # A new sweep without aggregation must not expose a previous
            # sweep's report as if it belonged to the current results.
            os.remove(report_path)
