"""
Graders for RAG evaluation.

Two families:
  1. Retrieval metrics - pure functions over ranked doc IDs. No API calls,
     deterministic, unit-testable in CI.
  2. Generation graders - LLM-as-judge for correctness and groundedness,
     plus deterministic abstention checking for hallucination.
"""

import json
import math
import os
from dotenv import load_dotenv

load_dotenv()

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"


# ---------------------------------------------------------------------------
# Retrieval metrics (pure, deterministic, no API)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids, relevant_ids, k=None):
    """
    Fraction of retrieved documents that are relevant.

    Returns 0.0 when nothing is retrieved. Note the convention: precision
    over an empty retrieval is defined as 0.0 here rather than undefined,
    so aggregate averages stay well-defined.
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    # Count each relevant document at most once. Duplicate retrievals still
    # consume rank slots (and therefore remain in the denominator), but they
    # must not inflate the number of relevant hits.
    hits = len(set(retrieved_ids) & relevant_set)
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids, relevant_ids, k=None):
    """
    Fraction of all relevant documents that were retrieved.

    Returns 1.0 when there are no relevant documents to find, since a
    retriever cannot be penalised for an unanswerable question.
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 1.0
    retrieved_set = set(retrieved_ids)
    # Relevance labels identify documents, not occurrences. Deduplicating the
    # ground-truth set prevents duplicate IDs from distorting recall.
    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Reciprocal of the rank of the first relevant document (1-indexed).

    Returns 0.0 if no relevant document appears in the ranking. Averaged
    over a dataset this is Mean Reciprocal Rank (MRR), which rewards
    putting a relevant chunk near the top rather than merely including it.
    """
    if not relevant_ids:
        return 1.0
    relevant_set = set(relevant_ids)
    for index, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / index
    return 0.0


def hit_rate(retrieved_ids, relevant_ids):
    """1.0 if at least one relevant document was retrieved, else 0.0."""
    if not relevant_ids:
        return 1.0
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


# ---------------------------------------------------------------------------
# Abstention / hallucination (deterministic)
# ---------------------------------------------------------------------------

def did_abstain(answer):
    """True if the model declined to answer for lack of context."""
    return isinstance(answer, str) and answer.strip().casefold() == ABSTAIN_TOKEN.casefold()


def grade_abstention(answer, answerable):
    """
    Compare abstention behaviour against ground truth.

    Four outcomes:
      correct_answer   - answerable, model answered
      correct_abstain  - unanswerable, model abstained (the win condition)
      hallucination    - unanswerable, model answered anyway
      over_abstention  - answerable, model refused unnecessarily

    Hallucination and over-abstention are tracked separately because they
    have opposite fixes: one needs stricter grounding, the other looser.
    """
    abstained = did_abstain(answer)

    if answerable and not abstained:
        outcome = "correct_answer"
    elif not answerable and abstained:
        outcome = "correct_abstain"
    elif not answerable and not abstained:
        outcome = "hallucination"
    else:
        outcome = "over_abstention"

    return {
        "outcome": outcome,
        "abstained": abstained,
        "passed": outcome in ("correct_answer", "correct_abstain"),
    }


# ---------------------------------------------------------------------------
# LLM-as-judge graders (require API)
# ---------------------------------------------------------------------------

class RAGJudge:
    """LLM judge for answer correctness and groundedness."""

    def __init__(self, judge_model="claude"):
        self.judge_model = judge_model
        self.anthropic_client = None
        self.openai_client = None

        if judge_model == "claude":
            try:
                from anthropic import Anthropic
                key = os.getenv("ANTHROPIC_API_KEY")
                if key:
                    self.anthropic_client = Anthropic(api_key=key)
            except Exception as e:
                print(f"[warn] RAGJudge: could not init Anthropic client: {e}")
        elif judge_model == "gpt":
            try:
                from openai import OpenAI
                key = os.getenv("OPENAI_API_KEY")
                if key:
                    self.openai_client = OpenAI(api_key=key)
            except Exception as e:
                print(f"[warn] RAGJudge: could not init OpenAI client: {e}")

    def _call(self, prompt):
        """Send a prompt to the configured judge, return raw text."""
        if self.judge_model == "claude":
            if not self.anthropic_client:
                raise RuntimeError("Anthropic judge not configured")
            msg = self.anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()

        if not self.openai_client:
            raise RuntimeError("OpenAI judge not configured")
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content.strip()

    @staticmethod
    def _parse_score(raw_text):
        """
        Parse {"score": n, "reason": "..."} from a judge response.

        Tolerates markdown code fences, which judges emit often enough that
        naive json.loads fails on otherwise-valid responses. A parse failure
        returns a neutral placeholder with parse_ok=False so callers can exclude
        invalid responses from aggregate quality metrics.
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict) or "score" not in parsed:
                raise ValueError("judge response must contain a score")
            if isinstance(parsed["score"], bool):
                raise ValueError("judge score must be numeric, not boolean")
            score = float(parsed["score"]) / 10.0
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("judge score must be finite and between 0 and 10")
            return {
                "score": score,
                "reason": parsed.get("reason", ""),
                "parse_ok": True,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"score": 0.5, "reason": raw_text[:150], "parse_ok": False}

    def grade_correctness(self, question, expected, actual):
        """Score the answer against the reference answer (0-1)."""
        prompt = (
            "You are grading a question-answering system.\n\n"
            f"QUESTION: {question}\n"
            f"REFERENCE ANSWER: {expected}\n"
            f"ACTUAL ANSWER: {actual}\n\n"
            "Score 0-10 for factual agreement with the reference. Ignore wording "
            "and style differences; judge only whether the substance matches.\n"
            '9-10 fully matches, 7-8 minor omission, 5-6 partially correct, '
            "3-4 mostly wrong, 0-2 contradicts or non-responsive.\n\n"
            'Respond ONLY with JSON: {"score": <0-10>, "reason": "<brief>"}'
        )
        try:
            result = self._parse_score(self._call(prompt))
            return {
                "metric": "correctness",
                "score": result["score"] if result["parse_ok"] else None,
                "passed": result["score"] >= 0.6 if result["parse_ok"] else None,
                "reason": result["reason"],
                "parse_ok": result["parse_ok"],
            }
        except Exception as e:
            return {"metric": "correctness", "score": None, "passed": None,
                    "reason": f"judge error: {str(e)[:80]}", "parse_ok": False}

    def grade_groundedness(self, answer, retrieved_docs):
        """
        Score how well the answer is supported by retrieved context (0-1).

        Deliberately does NOT consider real-world truth - only whether the
        context supports each claim. An answer that is true but unsupported
        scores low here, which is the signal that retrieval underperformed.
        """
        context = "\n\n".join(
            f"[{d['doc_id']}] {d['text']}" for d in retrieved_docs
        )
        prompt = (
            "You are checking whether an answer is grounded in provided context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER: {answer}\n\n"
            "Score 0-10 for how fully the context supports the answer's claims. "
            "Judge ONLY support by the context, not real-world truth. An answer "
            "that is true but unsupported by this context must score low.\n"
            "9-10 every claim supported, 7-8 minor unsupported detail, "
            "5-6 partly supported, 3-4 mostly unsupported, 0-2 fabricated.\n\n"
            'Respond ONLY with JSON: {"score": <0-10>, "reason": "<brief>"}'
        )
        try:
            result = self._parse_score(self._call(prompt))
            return {
                "metric": "groundedness",
                "score": result["score"] if result["parse_ok"] else None,
                "passed": result["score"] >= 0.6 if result["parse_ok"] else None,
                "reason": result["reason"],
                "parse_ok": result["parse_ok"],
            }
        except Exception as e:
            return {"metric": "groundedness", "score": None, "passed": None,
                    "reason": f"judge error: {str(e)[:80]}", "parse_ok": False}
