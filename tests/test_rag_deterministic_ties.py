"""Retrieval tie-breaking must not depend on JSON corpus insertion order."""

from evalforge.rag import VectorStore
from evalforge.rag_graders import average_precision_at_k, ndcg_at_k


class FixedEmbedder:
    def embed(self, text):
        return [1.0, 0.0]


def make_store(ids, scores):
    store = VectorStore(embedding_client=FixedEmbedder())
    store.documents = [
        {"doc_id": doc_id, "title": doc_id, "text": "evidence"}
        for doc_id in ids
    ]
    store.vectors = {doc_id: vector for doc_id, vector in scores.items()}
    return store


def test_tied_cosine_scores_have_stable_rank_across_corpus_orders():
    vectors = {
        "z-last": [1.0, 0.0],
        "a-first": [1.0, 0.0],
        "m-middle": [1.0, 0.0],
    }
    expected = ["a-first", "m-middle", "z-last"]
    for order in (list(vectors), list(reversed(vectors))):
        store = make_store(order, vectors)
        ids = [doc["doc_id"] for doc in store.search("query", top_k=3)]
        assert ids == expected
        # Ranking-sensitive metrics must not change when only corpus order does.
        assert average_precision_at_k(ids, ["m-middle"], k=3) == 0.5
        assert ndcg_at_k(ids, ["m-middle"], k=3) > 0


def test_score_dominates_tie_break_and_cutoff_is_respected():
    store = make_store(
        ["z-strong", "a-tied", "b-tied"],
        {
            "z-strong": [1.0, 0.0],
            "a-tied": [0.0, 1.0],
            "b-tied": [0.0, 1.0],
        },
    )
    assert [doc["doc_id"] for doc in store.search("query", top_k=2)] == [
        "z-strong", "a-tied",
    ]
