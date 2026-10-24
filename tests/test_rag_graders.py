"""
Unit tests for RAG retrieval metrics and abstention logic.

Every test here is deterministic and API-free, so the whole file runs in
CI. The LLM-judge paths are covered separately and marked 'api'.
"""

import pytest

from evalforge.rag import cosine_similarity, RAGPipeline
from evalforge.rag_graders import (
    precision_at_k, recall_at_k, reciprocal_rank, hit_rate,
    did_abstain, grade_abstention, RAGJudge,
)


class TestCosineSimilarity:
    """Cosine similarity is the core of retrieval - it must be exactly right."""

    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_ignores_magnitude(self):
        """A doc twice as long must not score differently for that reason."""
        assert cosine_similarity([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        """Degenerate embeddings must not raise ZeroDivisionError."""
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_half_relevant(self):
        assert precision_at_k(["a", "b"], ["a"]) == pytest.approx(0.5)

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], ["a"]) == 0.0

    def test_empty_retrieval(self):
        assert precision_at_k([], ["a"]) == 0.0

    def test_truncates_to_k(self):
        """Only the top k should count, not the whole ranking."""
        assert precision_at_k(["a", "x", "y"], ["a"], k=1) == pytest.approx(1.0)

    def test_duplicate_relevant_doc_does_not_inflate_precision(self):
        """A retriever repeating one relevant chunk must not earn extra credit."""
        assert precision_at_k(["a", "a", "x"], ["a"]) == pytest.approx(1.0 / 3.0)


class TestRecallAtK:
    def test_found_all(self):
        assert recall_at_k(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_found_half(self):
        assert recall_at_k(["a", "x"], ["a", "b"]) == pytest.approx(0.5)

    def test_found_none(self):
        assert recall_at_k(["x"], ["a"]) == 0.0

    def test_duplicate_relevant_ids_do_not_inflate_recall(self):
        assert recall_at_k(["a"], ["a", "a", "b"]) == pytest.approx(0.5)

    def test_no_relevant_docs_is_perfect(self):
        """Unanswerable questions must not penalise the retriever."""
        assert recall_at_k(["x"], []) == 1.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "x"], ["a"]) == pytest.approx(1.0)

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], ["a"]) == pytest.approx(0.5)

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1.0 / 3.0)

    def test_not_found(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_rewards_earlier_rank(self):
        """MRR must strictly prefer a higher-ranked relevant doc."""
        assert reciprocal_rank(["a", "x"], ["a"]) > reciprocal_rank(["x", "a"], ["a"])


class TestHitRate:
    def test_hit(self):
        assert hit_rate(["x", "a"], ["a"]) == 1.0

    def test_miss(self):
        assert hit_rate(["x", "y"], ["a"]) == 0.0

    def test_no_relevant_docs(self):
        assert hit_rate(["x"], []) == 1.0


class TestAbstention:
    def test_detects_abstain_token(self):
        assert did_abstain("INSUFFICIENT_CONTEXT") is True

    def test_detects_abstain_case_insensitive(self):
        assert did_abstain("insufficient_context") is True

    def test_embedded_abstention_token_does_not_mask_unsupported_claims(self):
        answer = "INSUFFICIENT_CONTEXT, but the rate was 4.2 percent."
        assert did_abstain(answer) is False
        assert grade_abstention(answer, answerable=False)["outcome"] == "hallucination"

    def test_abstention_token_with_surrounding_whitespace(self):
        assert did_abstain("  INSUFFICIENT_CONTEXT\n") is True

    def test_normal_answer_is_not_abstention(self):
        assert did_abstain("The capital is Canberra.") is False

    def test_none_answer_is_safe(self):
        assert did_abstain(None) is False

    @pytest.mark.parametrize("answer", ["", "  \t\n", None])
    @pytest.mark.parametrize("answerable", [True, False])
    def test_empty_answer_is_a_failure_not_a_hallucination_or_refusal(self, answer, answerable):
        result = grade_abstention(answer, answerable=answerable)
        assert result == {"outcome": "empty_answer", "abstained": False, "passed": False}

    def test_correct_answer_outcome(self):
        result = grade_abstention("Canberra.", answerable=True)
        assert result["outcome"] == "correct_answer"
        assert result["passed"] is True

    def test_correct_abstain_outcome(self):
        result = grade_abstention("INSUFFICIENT_CONTEXT", answerable=False)
        assert result["outcome"] == "correct_abstain"
        assert result["passed"] is True

    def test_hallucination_outcome(self):
        """Answering an unanswerable question is the failure we most care about."""
        result = grade_abstention("The rate was 4.2 percent.", answerable=False)
        assert result["outcome"] == "hallucination"
        assert result["passed"] is False

    def test_over_abstention_outcome(self):
        result = grade_abstention("INSUFFICIENT_CONTEXT", answerable=True)
        assert result["outcome"] == "over_abstention"
        assert result["passed"] is False


class TestJudgeParsing:
    """Judge response parsing must survive the formats models actually emit."""

    def test_parses_plain_json(self):
        result = RAGJudge._parse_score('{"score": 8, "reason": "good"}')
        assert result["score"] == pytest.approx(0.8)
        assert result["parse_ok"] is True

    def test_parses_markdown_fenced_json(self):
        raw = '```json\n{"score": 10, "reason": "perfect"}\n```'
        result = RAGJudge._parse_score(raw)
        assert result["score"] == pytest.approx(1.0)
        assert result["parse_ok"] is True

    @pytest.mark.parametrize("raw", [
        'Here is my grade: {"score": 6, "reason": "partial"} - hope that helps',
        '{"score": 10, "reason": "perfect"} trailing commentary',
        '{"score": 10} {"score": 0}',
        '```json\n{"score": 10}\n``` trailing commentary',
        '```python\n{"score": 10}\n```',
        '```json\n{"score": 10}',
    ])
    def test_rejects_json_with_extra_text_or_incomplete_fences(self, raw):
        result = RAGJudge._parse_score(raw)
        assert result["parse_ok"] is False
        assert result["score"] == pytest.approx(0.5)

    def test_rejected_judge_prose_is_excluded_from_grading(self, monkeypatch):
        judge = RAGJudge.__new__(RAGJudge)
        monkeypatch.setattr(
            judge, "_call",
            lambda prompt: 'Here is a perfect score: {"score": 10, "reason": "yes"}',
        )
        result = judge.grade_correctness("Q", "expected", "actual")
        assert result["parse_ok"] is False
        assert result["score"] is None
        assert result["passed"] is None

    def test_unparseable_returns_neutral_and_flags(self):
        """A parse failure must be visible, not silently scored as a pass."""
        result = RAGJudge._parse_score("I think it was pretty good actually")
        assert result["score"] == pytest.approx(0.5)
        assert result["parse_ok"] is False

    @pytest.mark.parametrize("raw", ['{"score": 50}', '{"score": -1}', '{"score": true}'])
    def test_out_of_range_or_boolean_scores_are_rejected(self, raw):
        result = RAGJudge._parse_score(raw)
        assert result["parse_ok"] is False
        assert result["score"] == pytest.approx(0.5)




    @pytest.mark.parametrize("raw", [
        '{}',
        '{"reason": "no score"}',
        '{"score": "not-a-number"}',
        '{"score": "NaN"}',
        '{"score": "Infinity"}',
        '{"score": null}',
    ])
    def test_invalid_judge_scores_flag_parse_failure(self, raw):
        result = RAGJudge._parse_score(raw)
        assert result["parse_ok"] is False
        assert result["score"] == pytest.approx(0.5)

    def test_valid_zero_score_is_not_confused_with_missing_score(self):
        result = RAGJudge._parse_score('{"score": 0, "reason": "unsupported"}')
        assert result["parse_ok"] is True
        assert result["score"] == pytest.approx(0.0)

    @pytest.mark.parametrize("raw", [
        '{"score": 0, "score": 10, "reason": "ambiguous"}',
        '{"score": 8, "reason": "first", "reason": "second"}',
        '{"score": 8, "metadata": {"x": 1, "x": 2}}',
        '{"score": "9", "reason": "string instead of JSON number"}',
        '{"score": 8, "reason": ["structured", "instead of text"]}',
        '{"score": 8, "reason": null}',
    ])
    def test_ambiguous_or_schema_invalid_judge_output_is_excluded(self, raw):
        result = RAGJudge._parse_score(raw)
        assert result["parse_ok"] is False
        assert result["score"] == pytest.approx(0.5)

    def test_oversized_integer_judge_score_is_excluded_without_crashing(self):
        raw = '{"score": ' + '9' * 309 + ', "reason": "overflow"}'
        result = RAGJudge._parse_score(raw)
        assert result["parse_ok"] is False
        assert result["score"] == pytest.approx(0.5)

    def test_valid_integer_and_fractional_judge_scores_remain_supported(self):
        for raw, expected in [
            ('{"score": 10, "reason": "full"}', 1.0),
            ('{"score": 7.5, "reason": "partial"}', 0.75),
        ]:
            result = RAGJudge._parse_score(raw)
            assert result["parse_ok"] is True
            assert result["score"] == pytest.approx(expected)


class TestPromptConstruction:
    def test_prompt_includes_abstention_instruction(self):
        """Without this instruction, hallucination cannot be measured."""
        docs = [{"doc_id": "d1", "title": "T", "text": "Body text."}]
        prompt = RAGPipeline.build_prompt("Q?", docs)
        assert "INSUFFICIENT_CONTEXT" in prompt

    def test_prompt_includes_retrieved_context(self):
        docs = [{"doc_id": "d1", "title": "Title", "text": "Unique body text."}]
        prompt = RAGPipeline.build_prompt("Q?", docs)
        assert "Unique body text." in prompt
        assert "d1" in prompt

    def test_prompt_includes_question(self):
        docs = [{"doc_id": "d1", "title": "T", "text": "B"}]
        prompt = RAGPipeline.build_prompt("What is groundedness?", docs)
        assert "What is groundedness?" in prompt



def test_embedding_cache_is_model_scoped(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import evalforge.rag as rag

    client = rag.EmbeddingClient(cache_path=str(tmp_path / "cache.json"))
    calls = []

    def create(model, input):
        calls.append((model, input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])])

    client.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    monkeypatch.setattr(rag, "EMBED_MODEL", "model-a")
    client.embed("sample")
    client.embed("sample")
    monkeypatch.setattr(rag, "EMBED_MODEL", "model-b")
    client.embed("sample")
    assert calls == [("model-a", "sample"), ("model-b", "sample")]


def test_embedding_cache_flat_path(tmp_path, monkeypatch):
    import evalforge.rag as rag

    monkeypatch.chdir(tmp_path)
    client = rag.EmbeddingClient(cache_path="cache.json")
    client.cache[rag._embedding_cache_key("sample")] = [1.0]
    client.save_cache()
    assert rag.EmbeddingClient(cache_path="cache.json").cache == client.cache



def test_judge_parse_failures_are_not_treated_as_answer_quality(monkeypatch):
    """An unavailable/malformed judge must not count as a failed RAG answer."""
    judge = RAGJudge.__new__(RAGJudge)
    monkeypatch.setattr(judge, "_call", lambda prompt: "not valid JSON")
    correctness = judge.grade_correctness("Question", "expected", "actual")
    groundedness = judge.grade_groundedness("actual", [{"doc_id": "d", "text": "context"}])
    for result in (correctness, groundedness):
        assert result["parse_ok"] is False
        assert result["score"] is None
        assert result["passed"] is None


def test_judge_api_failure_is_not_treated_as_answer_quality(monkeypatch):
    judge = RAGJudge.__new__(RAGJudge)

    def unavailable(prompt):
        raise RuntimeError("judge API unavailable")

    monkeypatch.setattr(judge, "_call", unavailable)
    result = judge.grade_correctness("Question", "expected", "actual")
    assert result["parse_ok"] is False
    assert result["score"] is None
    assert result["passed"] is None


def test_rag_report_excludes_judge_failures_and_tracks_coverage():
    from types import SimpleNamespace
    from evalforge.rag_runner import RAGEvalRunner

    runner = RAGEvalRunner.__new__(RAGEvalRunner)
    runner.model = "offline-test"
    runner.questions = [{"id": "q"}]
    runner.embedder = SimpleNamespace(api_calls=0, cache_hits=0)

    def row(score, parse_ok):
        return {
            "top_k": 1,
            "generation_error": None,
            "answerable": True,
            "retrieval": {
                "precision_at_k": 1.0, "recall_at_k": 1.0,
                "reciprocal_rank": 1.0, "hit_rate": 1.0,
            },
            "abstention": {"outcome": "correct_answer"},
            "correctness": {"score": score, "parse_ok": parse_ok},
            "groundedness": {"score": score, "parse_ok": parse_ok},
            "latency": {"retrieval_ms": 1, "generation_ms": 1, "total_ms": 2},
        }

    runner.results = [row(None, False), row(0.8, True)]
    generation = runner.generate_report()["by_top_k"]["1"]["generation"]
    assert generation["correctness"] == pytest.approx(0.8)
    assert generation["groundedness"] == pytest.approx(0.8)
    assert generation["correctness_judged"] == 1
    assert generation["groundedness_judged"] == 1
    assert generation["correctness_judge_errors"] == 1
    assert generation["groundedness_judge_errors"] == 1

    runner.results = [row(None, False)]
    generation = runner.generate_report()["by_top_k"]["1"]["generation"]
    assert generation["correctness"] is None
    assert generation["groundedness"] is None
