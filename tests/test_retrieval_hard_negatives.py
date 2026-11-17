"""Adversarial lexical retrieval regression: a distractor must not be credited as evidence.

These are hand-authored synthetic examples, not an external benchmark or LLM test.
"""
from evalforge.custom_gate import gate


def test_relevant_document_survives_keyword_stuffed_distractor():
    docs = [
        {"doc_id": "policy", "title": "Refund eligibility", "text":
         "Refund eligibility requires an original receipt and an order number within thirty days."},
        {"doc_id": "distractor", "title": "Refund eligibility refund eligibility", "text":
         "Refund eligibility refund eligibility: this page discusses marketing search keywords, not the refund procedure."},
    ]
    questions = [
        {"id": "refund", "question": "What is the refund eligibility procedure with receipt and order number?",
         "answerable": True, "relevant_doc_ids": ["policy"]},
        {"id": "weather", "question": "What is tomorrow's weather forecast?",
         "answerable": False, "relevant_doc_ids": []},
    ]
    result = gate(docs, questions, k=1, min_hit_rate=0,
                  min_answerable_questions=1, min_unanswerable_questions=1,
                  min_distinct_relevant_docs=1, include_cases=True)
    assert result["passed"], result["failure_reasons"]
    assert result["observed_lexical_evidence_hit_rate"] == 1.0
    assert result == gate(docs, questions, k=1, min_hit_rate=0,
                          min_answerable_questions=1, min_unanswerable_questions=1,
                          min_distinct_relevant_docs=1, include_cases=True)


def test_keyword_stuffing_does_not_get_mislabeled_as_relevant():
    docs = [
        {"doc_id": "actual", "title": "Reset password", "text": "Use the verified email to reset a password."},
        {"doc_id": "stuffed", "title": "Reset password reset password reset password",
         "text": "Reset password reset password reset password. This is an advertisement, not instructions."},
    ]
    questions = [{"id": "reset", "question": "reset password", "answerable": True,
                  "relevant_doc_ids": ["actual"]}]
    result = gate(docs, questions, k=1, min_hit_rate=1)
    assert not result["passed"]
    assert "lexical_evidence_hit_rate_below_minimum" in result["failure_reasons"]
    assert result["observed_lexical_evidence_hit_rate"] == 0.0
