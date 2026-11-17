"""Contract tests for the custom-dataset lexical regression gate."""
import json
import subprocess
import sys

import pytest

from evalforge.custom_gate import gate

DOCS = [
    {"doc_id": "a", "title": "Refund policy", "text": "Request refund for purchases"},
    {"doc_id": "b", "title": "Shipping status", "text": "Track shipping packages"},
]
QUESTIONS = [
    {"id": "q1", "question": "refund purchases", "answerable": True, "relevant_doc_ids": ["a"]},
    {"id": "q2", "question": "unknown", "answerable": False, "relevant_doc_ids": []},
]


def test_passes_with_reproducible_evidence():
    report = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    assert report == gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    assert report["passed"] is True
    assert report["benchmark"]["corpus_sha256"] == report["dataset_audit"]["corpus_sha256"]
    assert report["benchmark"]["questions_sha256"] == report["dataset_audit"]["questions_sha256"]
    assert "cases" not in report["benchmark"]


def test_machine_readable_failure_reasons_are_stable():
    passing = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    assert passing["failure_reasons"] == []
    unreachable = [dict(QUESTIONS[0], question="unmatchedword")]
    invalid = gate(DOCS, unreachable, k=1, min_hit_rate=0)
    assert "dataset_preflight_warnings" in invalid["failure_reasons"]
    assert "undefined_lexical_evidence_hit_rate" not in invalid["failure_reasons"] or invalid["observed_lexical_evidence_hit_rate"] is None
    threshold = gate(DOCS, QUESTIONS, k=1, min_hit_rate=1)
    if threshold["observed_lexical_evidence_hit_rate"] < 1:
        assert "lexical_evidence_hit_rate_below_minimum" in threshold["failure_reasons"]
    assert gate(DOCS, QUESTIONS, k=1, min_hit_rate=0) == passing


def test_fails_threshold_and_unreachable_labels():
    unreachable = [dict(QUESTIONS[0], question="unmatchedword")]
    report = gate(DOCS, unreachable, min_hit_rate=0)
    assert not report["passed"]
    assert report["preflight_warnings"]["answerable_question_ids_without_relevant_doc_token_overlap"] == ["q1"]


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError):
        gate(DOCS, QUESTIONS, min_hit_rate=value)


def test_cli_exit_codes_and_no_overwrite(tmp_path):
    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    output = tmp_path / "report.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--min-hit-rate", "0", "--output", str(output)]
    success = subprocess.run(cmd, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert json.loads(output.read_text())["passed"]
    assert subprocess.run(cmd, capture_output=True).returncode == 2
    questions.write_text(json.dumps([dict(QUESTIONS[0], question="unmatchedword")]), encoding="utf-8")
    output.unlink()
    failed = subprocess.run(cmd, capture_output=True, text=True)
    assert failed.returncode == 1, failed.stderr
    assert not json.loads(output.read_text())["passed"]


def test_github_annotations_are_opt_in_and_escape_untrusted_ids(tmp_path):
    from evalforge.custom_gate import github_annotations
    assert github_annotations(gate(DOCS, QUESTIONS, min_hit_rate=0)) == []
    unsafe = [dict(QUESTIONS[0], id="q%::warning", question="unmatchedword")]
    report = gate(DOCS, unsafe, min_hit_rate=0)
    annotations = github_annotations(report)
    assert annotations and all(line.startswith("::error title=") for line in annotations)
    assert any("q%25%3A%3Awarning" in line for line in annotations)
    assert all("\n" not in line and "\r" not in line for line in annotations)
    injected = dict(report, preflight_warnings={"question_ids": ["q%\n::warning"]})
    injected_annotations = github_annotations(injected)
    assert any("q%25%0A%3A%3Awarning" in line for line in injected_annotations)
    assert all("\n" not in line and "\r" not in line for line in injected_annotations)

    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(unsafe), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--min-hit-rate", "0"]
    plain = subprocess.run(cmd, capture_output=True, text=True)
    annotated = subprocess.run(cmd + ["--github-annotations"], capture_output=True, text=True)
    assert plain.returncode == annotated.returncode == 1
    assert "::error" not in plain.stderr
    assert "::error title=EvalForge dataset preflight::" in annotated.stderr
    assert json.loads(plain.stdout) == json.loads(annotated.stdout)


def test_opt_in_case_evidence_preserves_aggregate_contract(tmp_path):
    compact = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0)
    detailed = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0, include_cases=True)
    assert "cases" not in compact
    assert detailed["cases"]
    assert {key: value for key, value in detailed.items() if key != "cases"} == compact
    assert detailed == gate(DOCS, QUESTIONS, k=1, min_hit_rate=0, include_cases=True)
    with pytest.raises(ValueError):
        gate(DOCS, QUESTIONS, include_cases="yes")

    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--k", "1", "--min-hit-rate", "0"]
    plain = subprocess.run(cmd, capture_output=True, text=True)
    detailed_cli = subprocess.run(cmd + ["--include-cases"], capture_output=True, text=True)
    assert plain.returncode == detailed_cli.returncode == 0
    assert "cases" not in json.loads(plain.stdout)
    assert json.loads(detailed_cli.stdout) == detailed


def test_ambiguous_cutoff_tie_is_opt_in_failure():
    # Equal positive BM25 scores at k=1; only one tied document is relevant.
    docs = [
        {"doc_id": "a", "title": "", "text": "shared term"},
        {"doc_id": "b", "title": "", "text": "shared term"},
    ]
    questions = [{"id": "q", "question": "shared", "answerable": True,
                  "relevant_doc_ids": ["a"]}]
    ordinary = gate(docs, questions, k=1, min_hit_rate=0)
    strict = gate(docs, questions, k=1, min_hit_rate=0,
                  fail_on_ambiguous_ties=True)
    assert ordinary["ambiguous_relevance_cutoff_tie_count"] == 1
    assert ordinary["passed"] is True
    assert strict["passed"] is False
    from evalforge.custom_gate import github_annotations
    assert any("ambiguous BM25 cutoff" in line for line in github_annotations(strict))
    with pytest.raises(ValueError):
        gate(docs, questions, fail_on_ambiguous_ties="true")


def test_ambiguous_tie_cli_failure_preserves_evidence(tmp_path):
    docs = [{"doc_id": "a", "title": "", "text": "shared term"},
            {"doc_id": "b", "title": "", "text": "shared term"}]
    questions = [{"id": "q", "question": "shared", "answerable": True,
                  "relevant_doc_ids": ["a"]}]
    corpus_path = tmp_path / "corpus.json"
    questions_path = tmp_path / "questions.json"
    output = tmp_path / "gate.json"
    corpus_path.write_text(json.dumps(docs), encoding="utf-8")
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus_path),
           "--questions", str(questions_path), "--k", "1", "--min-hit-rate", "0",
           "--fail-on-ambiguous-ties", "--output", str(output)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 1, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ambiguous_relevance_cutoff_tie_count"] == 1
    assert not report["passed"]


def test_minimum_answerable_sample_size_is_a_real_gate(tmp_path):
    from evalforge.custom_gate import github_annotations
    passing = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0, min_answerable_questions=1)
    assert passing["passed"] is True
    assert passing["failure_reasons"] == []
    failed = gate(DOCS, QUESTIONS, k=1, min_hit_rate=0, min_answerable_questions=2)
    assert failed["passed"] is False
    assert failed["failure_reasons"] == ["insufficient_answerable_questions"]
    assert failed["minimum_answerable_questions"] == 2
    assert any("dataset size" in line for line in github_annotations(failed))
    for value in (-1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            gate(DOCS, QUESTIONS, min_answerable_questions=value)
    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--k", "1", "--min-hit-rate", "0",
           "--min-answerable-questions", "2"]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 1, completed.stderr
    assert json.loads(completed.stdout)["failure_reasons"] == ["insufficient_answerable_questions"]
    assert subprocess.run(cmd[:-1] + ["-1"], capture_output=True).returncode == 2


def test_distinct_relevant_document_coverage_gate(tmp_path):
    """Repeated labels for one document must not inflate dataset diversity."""
    from evalforge.custom_gate import github_annotations
    repeated = [QUESTIONS[0], dict(QUESTIONS[0], id="q3", question="refund")]
    report = gate(DOCS, repeated, k=1, min_hit_rate=0,
                  min_distinct_relevant_docs=2)
    assert report["observed_distinct_relevant_documents"] == 1
    assert report["minimum_distinct_relevant_documents"] == 2
    assert report["passed"] is False
    assert "insufficient_distinct_relevant_documents" in report["failure_reasons"]
    assert any("relevance diversity" in line for line in github_annotations(report))
    assert gate(DOCS, repeated, k=1, min_hit_rate=0,
                min_distinct_relevant_docs=1)["passed"]
    for invalid in (-1, True, 1.5, "2"):
        with pytest.raises(ValueError):
            gate(DOCS, QUESTIONS, min_distinct_relevant_docs=invalid)

    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(repeated), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--k", "1", "--min-hit-rate", "0",
           "--min-distinct-relevant-docs", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["observed_distinct_relevant_documents"] == 1


def test_minimum_unanswerable_coverage_is_enforced(tmp_path):
    """A benchmark with only positive queries must not silently satisfy a negative-coverage gate."""
    from evalforge.custom_gate import github_annotations
    positives = [QUESTIONS[0]]
    report = gate(DOCS, positives, k=1, min_hit_rate=0,
                  min_unanswerable_questions=1)
    assert report["passed"] is False
    assert report["observed_unanswerable_questions"] == 0
    assert report["failure_reasons"] == ["insufficient_unanswerable_questions"]
    assert any("negative coverage" in line for line in github_annotations(report))
    assert gate(DOCS, QUESTIONS, k=1, min_hit_rate=0,
                min_unanswerable_questions=1)["passed"]
    assert gate(DOCS, positives, k=1, min_hit_rate=0)["passed"]
    for invalid in (-1, True, 1.5, "1"):
        with pytest.raises(ValueError):
            gate(DOCS, QUESTIONS, min_unanswerable_questions=invalid)

    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(positives), encoding="utf-8")
    cmd = [sys.executable, "-m", "evalforge.custom_gate", "--corpus", str(corpus),
           "--questions", str(questions), "--k", "1", "--min-hit-rate", "0",
           "--min-unanswerable-questions", "1"]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 1, completed.stderr
    assert json.loads(completed.stdout)["failure_reasons"] == ["insufficient_unanswerable_questions"]
    assert subprocess.run(cmd[:-1] + ["-1"], capture_output=True).returncode == 2
