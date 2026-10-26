"""Benchmark provenance makes RAG reports comparable without external APIs."""

import json
from types import SimpleNamespace

import pytest

from evalforge.rag_runner import RAGEvalRunner, _dataset_sha256


def _question(answer="Evidence"):
    return {
        "id": "q-1",
        "question": "Where is the evidence?",
        "expected_answer": answer,
        "answerable": True,
        "relevant_doc_ids": ["doc-1"],
    }


class StubStore:
    def __init__(self, *, fail_build=False):
        self.documents = []
        self.fail_build = fail_build

    def load_corpus(self, path):
        self.documents = json.loads(path.read_text(encoding="utf-8"))

    def build_index(self):
        if self.fail_build:
            raise RuntimeError("embedding unavailable")


def _runner(tmp_path, *, fail_build=False):
    corpus_path = tmp_path / "corpus.json"
    questions_path = tmp_path / "questions.json"
    corpus_path.write_text(json.dumps([{
        "doc_id": "doc-1", "title": "Evidence", "text": "Supporting text"
    }]), encoding="utf-8")
    questions_path.write_text(json.dumps([_question()]), encoding="utf-8")

    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.corpus_path = corpus_path
    runner.questions_path = questions_path
    runner.store = StubStore(fail_build=fail_build)
    runner.questions = []
    runner.corpus_sha256 = "stale-corpus"
    runner.questions_sha256 = "stale-questions"
    return runner


def test_canonical_fingerprint_ignores_key_order_but_not_labels_or_case_order():
    left = [{"id": "q-1", "answerable": True}, {"id": "q-2", "answerable": False}]
    reordered_keys = [{"answerable": True, "id": "q-1"}, {"answerable": False, "id": "q-2"}]
    assert _dataset_sha256(left) == _dataset_sha256(reordered_keys)
    assert _dataset_sha256(left) != _dataset_sha256(list(reversed(left)))
    assert _dataset_sha256(left) != _dataset_sha256([
        {"id": "q-1", "answerable": False}, left[1]
    ])


def test_validated_benchmark_hashes_are_recorded_after_index_build(tmp_path):
    runner = _runner(tmp_path)
    runner.load()

    assert runner.corpus_sha256 == _dataset_sha256(runner.store.documents)
    assert runner.questions_sha256 == _dataset_sha256(runner.questions)
    assert len(runner.corpus_sha256) == len(runner.questions_sha256) == 64

    original = runner.questions_sha256
    runner.questions_path.write_text(json.dumps([_question("Updated evidence")]), encoding="utf-8")
    runner.load()
    assert runner.questions_sha256 != original


def test_invalid_reload_and_index_outage_clear_old_provenance(tmp_path):
    runner = _runner(tmp_path)
    runner.questions_path.write_text(json.dumps([_question("")]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_answer"):
        runner.load()
    assert runner.corpus_sha256 is None
    assert runner.questions_sha256 is None

    runner = _runner(tmp_path, fail_build=True)
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        runner.load()
    assert runner.corpus_sha256 is None
    assert runner.questions_sha256 is None


def test_report_records_dataset_and_model_configuration(tmp_path):
    runner = _runner(tmp_path)
    runner.load()
    runner.model = "claude"
    runner.judge_model = "openai"
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)
    runner.results = []
    runner.report = None

    provenance = runner.generate_report()["benchmark_provenance"]
    assert provenance["corpus_sha256"] == runner.corpus_sha256
    assert provenance["questions_sha256"] == runner.questions_sha256
    assert provenance["embedding_model"] == "text-embedding-3-small"
    assert provenance["judge_model"] == "openai"


def test_reload_invalidates_old_rows_and_report_even_on_failure(tmp_path):
    runner = _runner(tmp_path)
    runner.results = [{"case_id": "previous-run", "top_k": 1}]
    runner.report = {"by_top_k": {"1": {"stale": True}}}
    runner.load()
    assert runner.results == []
    assert runner.report is None

    runner.results = [{"case_id": "previous-run", "top_k": 1}]
    runner.report = {"by_top_k": {"1": {"stale": True}}}
    runner.questions_path.write_text(json.dumps([_question("")]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_answer"):
        runner.load()
    assert runner.results == []
    assert runner.report is None
    assert runner.corpus_sha256 is None
    assert runner.questions_sha256 is None


def test_canonical_hash_handles_escaped_unicode_metadata():
    # JSON permits escaped lone surrogates even though they are not UTF-8
    # characters. Unused metadata must not crash report fingerprinting.
    data = {"note": "\\ud800", "title": "Résumé"}
    assert len(_dataset_sha256(data)) == 64
    assert _dataset_sha256(data) == _dataset_sha256(dict(reversed(list(data.items()))))
