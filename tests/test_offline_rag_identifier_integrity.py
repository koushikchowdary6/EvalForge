"""The offline BM25 baseline must use the same safe join keys as RAG/agent runs."""

import pytest

from evalforge.offline_rag_demo import evaluate


DOCS = [{"doc_id": "doc-1", "title": "Battery", "text": "voltage reading"}]
CASES = [{
    "id": "q-1",
    "question": "battery voltage",
    "answerable": True,
    "relevant_doc_ids": ["doc-1"],
}]

BAD_IDS = [
    "\u0301",           # combining mark alone
    "doc\u200b-1",     # hidden zero-width space
    "doc\u202e-1",     # bidirectional text override
    "e\u0301",         # non-NFC spelling of é
    " doc-1 ",         # padded join key
    "doc\u00a01",      # deceptive Unicode space
    "x" * 257,         # above shared 256-character limit
]


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_offline_corpus_rejects_ambiguous_document_ids(bad_id):
    docs = [dict(DOCS[0], doc_id=bad_id)]
    with pytest.raises(ValueError, match="invalid corpus document"):
        evaluate(docs, CASES, (1,))


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_offline_questions_reject_ambiguous_case_ids(bad_id):
    cases = [dict(CASES[0], id=bad_id)]
    with pytest.raises(ValueError, match="invalid benchmark question"):
        evaluate(DOCS, cases, (1,))


def test_valid_nfc_unicode_and_internal_spaces_remain_usable():
    docs = [dict(DOCS[0], doc_id="évidence 1")]
    cases = [dict(CASES[0], id="caso-東京", relevant_doc_ids=["évidence 1"])]
    report = evaluate(docs, cases, (1,))
    assert report["cases"][0]["retrieved_ids"] == ["évidence 1"]
    assert report["by_top_k"]["1"]["hit_rate"] == 1.0


def test_shared_identifier_length_boundary_is_accepted():
    doc_id = "x" * 256
    docs = [dict(DOCS[0], doc_id=doc_id)]
    cases = [dict(CASES[0], relevant_doc_ids=[doc_id])]
    assert evaluate(docs, cases, (1,))["by_top_k"]["1"]["hit_rate"] == 1.0
