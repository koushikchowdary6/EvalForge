"""Coverage for BM25 benchmarks containing only unanswerable questions."""
import json
import subprocess
import sys

from evalforge.offline_rag_demo import evaluate


DOCS = [
    {"doc_id": "a", "title": "Battery telemetry", "text": "voltage sensor"},
    {"doc_id": "b", "title": "IAM policies", "text": "permission audit"},
]
UNANSWERABLE = [
    {"id": "u1", "question": "fictional nebula", "answerable": False, "relevant_doc_ids": []},
    {"id": "u2", "question": "battery", "answerable": False, "relevant_doc_ids": []},
]
QUALITY_KEYS = (
    "precision_at_k", "recall_at_k", "ndcg_at_k", "average_precision_at_k",
    "reciprocal_rank", "hit_rate", "lexical_evidence_hit_rate",
    "lexical_evidence_recall_at_k", "lexical_evidence_mrr",
)


def test_unanswerable_only_reports_null_quality_without_losing_coverage():
    report = evaluate(DOCS, UNANSWERABLE, (1, 3))
    assert report == evaluate(DOCS, UNANSWERABLE, (1, 3))
    assert len(report["cases"]) == 4
    for k in ("1", "3"):
        summary = report["by_top_k"][k]
        assert summary["answerable_count"] == 0
        assert summary["unanswerable_excluded"] == 2
        assert summary["answerable_zero_lexical_overlap_count"] == 0
        assert summary["unanswerable_zero_lexical_overlap_count"] == 1
        assert summary["tie_only_hit_count"] == 0
        assert all(summary[key] is None for key in QUALITY_KEYS)
    assert all(case["metrics"] is None and case["lexical_evidence_metrics"] is None
               for case in report["cases"])
    assert len(report["corpus_sha256"]) == len(report["questions_sha256"]) == 64


def test_mixed_benchmark_still_aggregates_only_answerable_cases():
    answerable = {
        "id": "a1", "question": "battery voltage", "answerable": True,
        "relevant_doc_ids": ["a"],
    }
    report = evaluate(DOCS, UNANSWERABLE + [answerable], (1,))
    summary = report["by_top_k"]["1"]
    assert summary["answerable_count"] == 1
    assert summary["unanswerable_excluded"] == 2
    assert summary["hit_rate"] == 1.0
    assert summary["lexical_evidence_hit_rate"] == 1.0


def test_cli_serializes_undefined_metrics_as_json_null(tmp_path):
    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(UNANSWERABLE), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.offline_rag_demo",
         "--corpus", str(corpus), "--questions", str(questions), "--k", "1"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    report = json.loads(result.stdout)
    assert all(report["by_top_k"]["1"][key] is None for key in QUALITY_KEYS)
    assert report["by_top_k"]["1"]["unanswerable_excluded"] == 2
