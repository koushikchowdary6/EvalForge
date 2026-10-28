"""Regression tests for invisible and canonically ambiguous benchmark IDs.

A benchmark label is a join key, not free-form model text. Hidden controls or
non-NFC spellings can split otherwise identical evidence across case IDs and
make evaluation artifacts difficult to audit.
"""

import json

import pytest

from evalforge.agent_runner import validate_agent_tasks
from evalforge.identifiers import valid_benchmark_id
from evalforge.rag import VectorStore
from evalforge.rag_runner import validate_rag_questions


BAD_IDS = [
    "\u0301",  # standalone combining mark: not a visible identifier
    "\u200b",  # zero-width space
    "case\u202e1",  # bidirectional override changes visual order
    "case\ud800",  # lone surrogate cannot be UTF-8 encoded
    "e\u0301",  # visually identical to NFC é, but a distinct byte sequence
    " case ",  # padding creates ambiguous joins
    "case\u00a01",  # non-ASCII whitespace looks like an ordinary space
]


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_rag_corpus_rejects_ambiguous_document_ids_before_embedding(tmp_path, bad_id):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps([{"doc_id": bad_id, "title": "Evidence", "text": "Document"}]),
        encoding="utf-8",
    )
    store = VectorStore(embedding_client=object())
    with pytest.raises(ValueError, match="doc_id"):
        store.load_corpus(corpus)
    assert store.documents == []
    assert store.vectors == {}


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_rag_question_rejects_ambiguous_case_ids(bad_id):
    case = {
        "id": bad_id,
        "question": "Where is the evidence?",
        "expected_answer": "In the document",
        "answerable": True,
        "relevant_doc_ids": ["doc-1"],
    }
    with pytest.raises(ValueError, match="id"):
        validate_rag_questions([case], {"doc-1"})


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_agent_dataset_rejects_ambiguous_case_ids(bad_id):
    with pytest.raises(ValueError, match="id"):
        validate_agent_tasks([{"id": bad_id, "prompt": "Summarize this evidence."}])


def test_rag_references_reject_hidden_document_ids():
    case = {
        "id": "q-1",
        "question": "Where is the evidence?",
        "expected_answer": "In the document",
        "answerable": True,
        "relevant_doc_ids": ["doc\u200b-1"],
    }
    with pytest.raises(ValueError, match="relevant_doc_ids"):
        validate_rag_questions([case], {"doc\u200b-1"})


def test_nfc_unicode_and_internal_plain_spaces_remain_valid():
    for label in ("évidence-1", "caso-東京", "case 1", "document_42"):
        assert valid_benchmark_id(label)
        validate_agent_tasks([{"id": label, "prompt": "Summarize this evidence."}])
