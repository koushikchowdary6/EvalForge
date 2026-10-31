"""Regression tests for bounded and safe offline benchmark JSON ingestion."""
import os

import pytest

from evalforge.offline_rag_demo import (
    MAX_BENCHMARK_JSON_BYTES,
    load_benchmark_json,
)


VALID_CORPUS = '[{"doc_id":"a","title":"Battery","text":"voltage"}]'


def test_regular_json_file_still_loads(tmp_path):
    source = tmp_path / "corpus.json"
    source.write_text(VALID_CORPUS, encoding="utf-8")
    assert load_benchmark_json(source)[0]["doc_id"] == "a"


def test_benchmark_json_symlink_is_rejected(tmp_path):
    target = tmp_path / "target.json"
    target.write_text(VALID_CORPUS, encoding="utf-8")
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="regular file"):
        load_benchmark_json(alias)


def test_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        load_benchmark_json(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs unavailable")
def test_fifo_is_rejected_without_blocking(tmp_path):
    source = tmp_path / "input.fifo"
    os.mkfifo(source)
    with pytest.raises(ValueError, match="regular file"):
        load_benchmark_json(source)


def test_oversized_file_is_rejected_before_parsing(tmp_path):
    source = tmp_path / "oversized.json"
    source.write_bytes(b"x" * (MAX_BENCHMARK_JSON_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_benchmark_json(source)


def test_exact_byte_limit_is_accepted_for_valid_json(tmp_path):
    source = tmp_path / "limit.json"
    source.write_bytes(b"[]" + b" " * (MAX_BENCHMARK_JSON_BYTES - 2))
    assert load_benchmark_json(source) == []


def test_invalid_utf8_fails_with_a_clear_validation_error(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_benchmark_json(source)
