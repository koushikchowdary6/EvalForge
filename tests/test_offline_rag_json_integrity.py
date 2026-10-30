"""Ambiguous benchmark files must never produce apparently reproducible scores."""
import json
import subprocess
import sys

import pytest

from evalforge.offline_rag_demo import load_benchmark_json


CORPUS = '[{"doc_id":"a","title":"Battery","text":"voltage"}]'
QUESTIONS = '[{"id":"q","question":"voltage","answerable":true,"relevant_doc_ids":["a"]}]'


@pytest.mark.parametrize(
    "corpus,questions,expected",
    [
        ('[{"doc_id":"a","doc_id":"b","title":"Battery","text":"voltage"}]',
         QUESTIONS, "duplicate JSON key"),
        (CORPUS,
         '[{"id":"q","question":"voltage","answerable":true,'
         '"answerable":false,"relevant_doc_ids":["a"]}]', "duplicate JSON key"),
        ('[{"doc_id":"a","title":"Battery","text":"voltage","metadata":{"x":1,"x":2}}]',
         QUESTIONS, "duplicate JSON key"),
        ('[{"doc_id":"a","title":"Battery","text":"voltage","extra":NaN}]',
         QUESTIONS, "non-finite JSON constant"),
        (CORPUS,
         '[{"id":"q","question":"voltage","answerable":true,'
         '"relevant_doc_ids":["a"],"extra":Infinity}]', "non-finite JSON constant"),
    ],
)
def test_cli_rejects_ambiguous_benchmark_json(tmp_path, corpus, questions, expected):
    corpus_path = tmp_path / "corpus.json"
    question_path = tmp_path / "questions.json"
    corpus_path.write_text(corpus, encoding="utf-8")
    question_path.write_text(questions, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "evalforge.offline_rag_demo",
         "--corpus", str(corpus_path), "--questions", str(question_path)],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode != 0
    assert expected in result.stderr
    assert result.stdout == ""


def test_strict_loader_accepts_valid_json_and_preserves_data(tmp_path):
    path = tmp_path / "valid.json"
    path.write_text(CORPUS, encoding="utf-8")
    assert load_benchmark_json(path) == json.loads(CORPUS)
