"""RAG sweeps must not grade mutated labels against a stale embedding index."""

import pytest

import evalforge.rag_runner as rag_runner
from tests.test_rag_benchmark_provenance import _runner


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda runner: runner.questions[0].__setitem__(
            "expected_answer", "Changed reference answer"), "questions changed"),
        (lambda runner: runner.store.documents[0].__setitem__(
            "text", "Changed indexed evidence"), "corpus changed"),
    ],
)
def test_in_memory_snapshot_drift_fails_before_provider_calls(
    tmp_path, monkeypatch, mutation, message
):
    runner = _runner(tmp_path)
    runner.load()
    runner.model = "claude"
    runner.results = [{"case_id": "stale", "top_k": 1}]
    runner.report = {"total_evaluations": 1}

    def forbid_provider(*args, **kwargs):
        raise AssertionError("must not construct a provider after snapshot drift")

    monkeypatch.setattr(rag_runner, "RAGPipeline", forbid_provider)
    mutation(runner)

    with pytest.raises(ValueError, match=message):
        runner.run(k_values=(1,))

    assert runner.results == []
    assert runner.report is None


def test_runner_requires_successful_load_before_using_provenance(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.model = "claude"
    runner.results = [{"case_id": "stale"}]
    runner.report = {"total_evaluations": 1}
    monkeypatch.setattr(
        rag_runner, "RAGPipeline",
        lambda *args, **kwargs: pytest.fail("provider must not be constructed"),
    )
    runner.corpus_sha256 = None
    runner.questions_sha256 = None

    with pytest.raises(ValueError, match="corpus changed or was not loaded"):
        runner.run(k_values=(1,))

    assert runner.results == []
    assert runner.report is None
