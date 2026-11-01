"""Deeply nested benchmark JSON must fail before recursive decoder exhaustion."""

import json
import subprocess
import sys

import pytest

from evalforge.offline_rag_demo import (
    MAX_BENCHMARK_JSON_DEPTH,
    load_benchmark_json,
)


def test_nested_benchmark_at_exact_depth_limit_loads(tmp_path):
    source = tmp_path / "boundary.json"
    source.write_text("[" * MAX_BENCHMARK_JSON_DEPTH + "0" +
                      "]" * MAX_BENCHMARK_JSON_DEPTH, encoding="utf-8")
    result = load_benchmark_json(source)
    for _ in range(MAX_BENCHMARK_JSON_DEPTH):
        result = result[0]
    assert result == 0


def test_excessive_nesting_fails_with_clear_error(tmp_path):
    source = tmp_path / "deep.json"
    source.write_text("[" * (MAX_BENCHMARK_JSON_DEPTH + 1) + "0" +
                      "]" * (MAX_BENCHMARK_JSON_DEPTH + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark JSON nesting exceeds"):
        load_benchmark_json(source)


def test_brackets_inside_quoted_strings_and_escaped_quotes_are_not_nesting(tmp_path):
    source = tmp_path / "strings.json"
    payload = {"text": "[" * 500 + ' escaped "quote" ' + "}" * 500}
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert load_benchmark_json(source) == payload


def test_cli_rejects_deep_corpus_without_traceback_or_partial_report(tmp_path):
    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    corpus.write_text("[" * (MAX_BENCHMARK_JSON_DEPTH + 1) + "0" +
                      "]" * (MAX_BENCHMARK_JSON_DEPTH + 1), encoding="utf-8")
    questions.write_text("[]", encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-m", "evalforge.offline_rag_demo",
         "--corpus", str(corpus), "--questions", str(questions)],
        capture_output=True, text=True, timeout=20,
    )
    assert run.returncode != 0
    assert "benchmark JSON nesting exceeds" in run.stderr
    assert "Traceback" not in run.stderr
    assert run.stdout == ""
