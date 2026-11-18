"""End-to-end dataset audit coverage with valid and pathological labels."""
import json
import subprocess
import sys

from evalforge.dataset_audit import audit


DOCS = [
    {"doc_id": "a", "title": "Refund", "text": "Request a refund"},
    {"doc_id": "b", "title": "Shipping", "text": "Track parcel"},
]
QUESTIONS = [
    {"id": "q1", "question": "How do I request a refund?", "answerable": True, "relevant_doc_ids": ["a"]},
    {"id": "q2", "question": "Unknown policy?", "answerable": False, "relevant_doc_ids": []},
]


def test_audit_is_deterministic_and_reports_provenance():
    first = audit(DOCS, QUESTIONS)
    assert first == audit(DOCS, QUESTIONS)
    assert first["answerable_questions"] == 1
    assert first["unanswerable_questions"] == 1
    assert first["answerable_question_ids_without_corpus_token_overlap"] == []
    assert len(first["corpus_sha256"]) == 64


def test_audit_warns_about_unreachable_relevance():
    docs = [{"doc_id": "empty", "title": "!!!", "text": "..."}]
    questions = [{"id": "q", "question": "refund", "answerable": True, "relevant_doc_ids": ["empty"]}]
    result = audit(docs, questions)
    assert result["unindexable_document_ids"] == ["empty"]
    assert result["question_ids_with_unindexable_relevance_labels"] == ["q"]
    assert result["answerable_question_ids_without_corpus_token_overlap"] == ["q"]


def test_installed_module_cli_strict_exit_and_json(tmp_path):
    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    command = [sys.executable, "-m", "evalforge.dataset_audit",
               "--corpus", str(corpus), "--questions", str(questions), "--strict"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema"] == "evalforge.dataset_audit.v1"
    bad = [dict(q) for q in QUESTIONS]
    bad[0]["question"] = "xyznotincorpus"
    questions.write_text(json.dumps(bad), encoding="utf-8")
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["answerable_question_ids_without_corpus_token_overlap"] == ["q1"]


def test_invalid_labels_fail_without_json_artifact(tmp_path):
    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text(json.dumps(DOCS), encoding="utf-8")
    invalid = [dict(QUESTIONS[0], relevant_doc_ids=["missing"])]
    questions.write_text(json.dumps(invalid), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "evalforge.dataset_audit",
                             "--corpus", str(corpus), "--questions", str(questions)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stdout == ""


def test_overlap_with_irrelevant_documents_does_not_mask_unreachable_label():
    docs = [
        {"doc_id": "relevant", "title": "Account recovery", "text": "Reset password"},
        {"doc_id": "distractor", "title": "Refund", "text": "Request a refund"},
    ]
    questions = [
        {"id": "q1", "question": "Refund request",
         "answerable": True, "relevant_doc_ids": ["relevant"]},
    ]
    result = audit(docs, questions)
    assert result["answerable_question_ids_without_corpus_token_overlap"] == []
    assert result["answerable_question_ids_without_relevant_doc_token_overlap"] == ["q1"]


def test_strict_cli_rejects_only_irrelevant_document_overlap(tmp_path):
    docs = [
        {"doc_id": "a", "title": "Password", "text": "Reset credentials"},
        {"doc_id": "b", "title": "Refund", "text": "Request a refund"},
    ]
    questions = [
        {"id": "q1", "question": "Refund request", "answerable": True,
         "relevant_doc_ids": ["a"]},
    ]
    corpus = tmp_path / "corpus.json"
    labels = tmp_path / "questions.json"
    corpus.write_text(json.dumps(docs), encoding="utf-8")
    labels.write_text(json.dumps(questions), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.dataset_audit",
         "--corpus", str(corpus), "--questions", str(labels), "--strict"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["answerable_question_ids_without_corpus_token_overlap"] == []
    assert report["answerable_question_ids_without_relevant_doc_token_overlap"] == ["q1"]


def test_unanswerable_overlap_is_reported_without_failing_strict_preflight(tmp_path):
    docs = [{"doc_id": "a", "title": "Refund policy", "text": "Request a refund"}]
    questions = [
        {"id": "unanswerable-overlap", "question": "Refund eligibility",
         "answerable": False, "relevant_doc_ids": []},
        {"id": "unanswerable-no-overlap", "question": "Quantum entanglement",
         "answerable": False, "relevant_doc_ids": []},
    ]
    report = audit(docs, questions)
    assert report["unanswerable_question_ids_with_corpus_token_overlap"] == [
        "unanswerable-overlap"
    ]
    assert report["unanswerable_questions"] == 2
    corpus = tmp_path / "corpus.json"
    labels = tmp_path / "questions.json"
    corpus.write_text(json.dumps(docs), encoding="utf-8")
    labels.write_text(json.dumps(questions), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.dataset_audit",
         "--corpus", str(corpus), "--questions", str(labels), "--strict"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["unanswerable_question_ids_with_corpus_token_overlap"] == [
        "unanswerable-overlap"
    ]
