"""Agent benchmark input is bounded before JSON parsing or paid model calls."""

import json
import os

import pytest

import evalforge.agent_runner as module
from evalforge.agent_runner import (
    AgentEvalRunner, MAX_AGENT_BENCHMARK_JSON_BYTES, _load_agent_benchmark,
)


def _valid_task():
    return {"id": "safe-case", "prompt": "Summarize this document."}


def test_regular_agent_benchmark_loads(tmp_path):
    source = tmp_path / "agent.json"
    source.write_text(json.dumps([_valid_task()]), encoding="utf-8")
    runner = AgentEvalRunner(source, output_dir=tmp_path)
    assert runner.load() == [_valid_task()]
    assert runner.tasks_sha256 is not None


@pytest.mark.parametrize("unsafe", ["directory", "symlink", "fifo"])
def test_unsafe_agent_benchmark_input_rejected_without_blocking(tmp_path, unsafe):
    source = tmp_path / "agent.json"
    if unsafe == "directory":
        source.mkdir()
    elif unsafe == "symlink":
        target = tmp_path / "target.json"
        target.write_text(json.dumps([_valid_task()]), encoding="utf-8")
        try:
            source.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs unavailable")
        os.mkfifo(source)
    with pytest.raises(ValueError, match="regular file"):
        _load_agent_benchmark(source)


def test_oversized_input_rejected_before_json_parsing_and_stale_state_cleared(tmp_path):
    source = tmp_path / "agent.json"
    source.write_text(json.dumps([_valid_task()]), encoding="utf-8")
    runner = AgentEvalRunner(source, output_dir=tmp_path)
    runner.load()
    runner.results = [{"stale": True}]
    runner.report = {"stale": True}
    source.write_bytes(b" " * (MAX_AGENT_BENCHMARK_JSON_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        runner.load()
    assert runner.tasks == []
    assert runner.results == []
    assert runner.report is None
    assert runner.tasks_sha256 is None


def test_exact_byte_limit_allows_valid_json(tmp_path):
    source = tmp_path / "agent.json"
    encoded = json.dumps([_valid_task()]).encode("utf-8")
    assert len(encoded) < MAX_AGENT_BENCHMARK_JSON_BYTES
    source.write_bytes(encoded + b" " * (MAX_AGENT_BENCHMARK_JSON_BYTES - len(encoded)))
    assert _load_agent_benchmark(source) == [_valid_task()]


def test_invalid_utf8_and_nonfinite_json_rejected(tmp_path):
    source = tmp_path / "agent.json"
    source.write_bytes(b"\\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        _load_agent_benchmark(source)
    source.write_text('[{"id": "x", "prompt": "ok", "extra": NaN}]', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        _load_agent_benchmark(source)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs unavailable")
def test_path_swapped_to_fifo_is_rejected_after_open(tmp_path, monkeypatch):
    source = tmp_path / "agent.json"
    source.write_text(json.dumps([_valid_task()]), encoding="utf-8")
    fifo = tmp_path / "swapped.fifo"
    os.mkfifo(fifo)
    original_open = os.open

    def swapped_open(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(source):
            return original_open(fifo, flags, *args, **kwargs)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", swapped_open)
    with pytest.raises(ValueError, match="regular file"):
        _load_agent_benchmark(source)
