"""End-to-end CLI contract for a user-supplied, labeled retrieval dataset.

Exercises the public module entrypoint rather than calling evaluate() directly.
The synthetic corpus is a packaging/onboarding fixture, not a model benchmark.
"""
import json
import subprocess
import sys
from pathlib import Path

from evalforge.offline_rag_demo import evaluate


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "evalforge.offline_rag_demo", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def test_custom_dataset_cli_produces_reproducible_artifact(tmp_path):
    corpus = EXAMPLES / "customer_support_corpus.json"
    questions = EXAMPLES / "customer_support_questions.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = ("--corpus", corpus, "--questions", questions,
            "--k", "1", "3", "--summary-only")

    for target in (first, second):
        result = _run(*args, "--output", target)
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""
        assert target.exists()

    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_text(encoding="utf-8"))
    expected = evaluate(
        json.loads(corpus.read_text(encoding="utf-8")),
        json.loads(questions.read_text(encoding="utf-8")),
        [1, 3], tokenizer_mode="ascii",
    )
    expected.pop("cases")
    assert report == expected
    assert "cases" not in report
    assert report["question_count"] == 5
    assert report["corpus_size"] == 4

    # The CLI must not silently overwrite previously published evidence.
    before = first.read_bytes()
    repeated = _run(*args, "--output", first)
    assert repeated.returncode != 0
    assert first.read_bytes() == before


def test_custom_dataset_cli_rejects_broken_relevance_labels_without_artifact(tmp_path):
    corpus = EXAMPLES / "customer_support_corpus.json"
    questions = json.loads(
        (EXAMPLES / "customer_support_questions.json").read_text(encoding="utf-8")
    )
    questions[0]["relevant_doc_ids"] = ["missing-document"]
    bad_questions = tmp_path / "bad-questions.json"
    bad_questions.write_text(json.dumps(questions), encoding="utf-8")
    target = tmp_path / "should-not-exist.json"

    result = _run("--corpus", corpus, "--questions", bad_questions,
                  "--k", "1", "3", "--summary-only", "--output", target)
    assert result.returncode != 0
    assert not target.exists()


def test_custom_dataset_cli_rejects_ambiguous_json_without_artifact(tmp_path):
    """Fail closed on ambiguous labels and JSON before writing evaluation evidence."""
    corpus = EXAMPLES / "customer_support_corpus.json"
    original = (EXAMPLES / "customer_support_questions.json").read_text(encoding="utf-8")
    cases = {
        "duplicate-key": original.replace('"answerable":', '"answerable": true, "answerable":', 1),
        "nonfinite": original.replace('"answerable":', '"unexpected": NaN, "answerable":', 1),
    }
    for name, payload in cases.items():
        bad = tmp_path / (name + ".json")
        bad.write_text(payload, encoding="utf-8")
        target = tmp_path / (name + "-report.json")
        result = _run("--corpus", corpus, "--questions", bad,
                      "--k", "1", "3", "--summary-only", "--output", target)
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        assert not target.exists(), name


def test_custom_dataset_cli_rejects_duplicate_question_ids_without_artifact(tmp_path):
    """Duplicate question IDs must never overwrite one another in metric reports."""
    corpus = EXAMPLES / "customer_support_corpus.json"
    questions = json.loads(
        (EXAMPLES / "customer_support_questions.json").read_text(encoding="utf-8")
    )
    questions[1]["id"] = questions[0]["id"]
    bad = tmp_path / "duplicate-ids.json"
    bad.write_text(json.dumps(questions), encoding="utf-8")
    target = tmp_path / "duplicate-ids-report.json"
    result = _run("--corpus", corpus, "--questions", bad,
                  "--k", "1", "3", "--summary-only", "--output", target)
    assert result.returncode != 0, (result.stdout, result.stderr)
    assert not target.exists()
