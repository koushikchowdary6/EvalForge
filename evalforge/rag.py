"""
RAG pipeline for EvalForge.

Embeds a document corpus, performs vector search by cosine similarity,
and generates grounded answers from retrieved context.

Design note: this deliberately avoids FAISS/Chroma. For a corpus of this
size an exact cosine scan is both faster (no index build) and dependency-free,
which keeps the project installable on any Python version. The similarity
math is identical to what a vector DB computes; the only thing given up is
approximate-nearest-neighbour scaling, which matters at ~100k+ vectors.
"""

import json
import math
import os
import time
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_CACHE_PATH = "results/embedding_cache.json"


def cosine_similarity(vec_a, vec_b):
    """
    Cosine similarity between two equal-length vectors.

    Measures the angle between vectors, ignoring magnitude, so document
    length does not distort the score. Returns 0.0 for a zero vector
    rather than raising, so a degenerate embedding cannot crash a run.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


class EmbeddingClient:
    """Wraps the OpenAI embeddings API with an on-disk cache."""

    def __init__(self, cache_path=EMBED_CACHE_PATH):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.api_calls = 0
        self.cache_hits = 0

        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key) if api_key else None
        except Exception as e:
            print(f"[warn] EmbeddingClient: could not init OpenAI client: {e}")
            self.client = None

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    def embed(self, text):
        """
        Embed a single string, using the cache when possible.

        Caching matters here: re-running the eval with a different top-k
        must not re-pay for embeddings, otherwise comparing retrieval
        settings costs money proportional to the number of settings.
        """
        if text in self.cache:
            self.cache_hits += 1
            return self.cache[text]

        if not self.client:
            raise RuntimeError(
                "OpenAI client unavailable - set OPENAI_API_KEY in .env to embed."
            )

        response = self.client.embeddings.create(model=EMBED_MODEL, input=text)
        vector = response.data[0].embedding
        self.api_calls += 1
        self.cache[text] = vector
        return vector


class VectorStore:
    """In-memory vector store over the document corpus."""

    def __init__(self, embedding_client=None):
        self.embedder = embedding_client or EmbeddingClient()
        self.documents = []
        self.vectors = {}

    def load_corpus(self, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        print(f"Loaded {len(self.documents)} documents from {corpus_path}")
        return self.documents

    def build_index(self):
        """Embed every document. Cached embeddings make repeat runs free."""
        print(f"Embedding {len(self.documents)} documents...")
        for doc in self.documents:
            content = f"{doc['title']}\n{doc['text']}"
            self.vectors[doc["doc_id"]] = self.embedder.embed(content)
        self.embedder.save_cache()
        print(
            f"Index ready. {self.embedder.api_calls} API calls, "
            f"{self.embedder.cache_hits} cache hits."
        )

    def search(self, query, top_k=3):
        """
        Return the top_k most similar documents to the query.

        Returns a list of dicts with doc_id, title, text, and score,
        ordered by descending cosine similarity.
        """
        if not self.vectors:
            raise RuntimeError("Index is empty - call build_index() first.")

        query_vector = self.embedder.embed(query)

        scored = []
        for doc in self.documents:
            score = cosine_similarity(query_vector, self.vectors[doc["doc_id"]])
            scored.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": score,
            })

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:top_k]


class RAGPipeline:
    """Retrieve-then-generate pipeline with an explicit grounding instruction."""

    def __init__(self, vector_store, model="claude"):
        self.store = vector_store
        self.model = model

        self.anthropic_client = None
        self.openai_client = None

        if model == "claude":
            try:
                from anthropic import Anthropic
                key = os.getenv("ANTHROPIC_API_KEY")
                if key:
                    self.anthropic_client = Anthropic(api_key=key)
            except Exception as e:
                print(f"[warn] RAGPipeline: could not init Anthropic client: {e}")
        elif model == "gpt":
            try:
                from openai import OpenAI
                key = os.getenv("OPENAI_API_KEY")
                if key:
                    self.openai_client = OpenAI(api_key=key)
            except Exception as e:
                print(f"[warn] RAGPipeline: could not init OpenAI client: {e}")

    @staticmethod
    def build_prompt(question, retrieved_docs):
        """
        Build a grounded-answer prompt.

        The INSUFFICIENT_CONTEXT instruction is what makes hallucination
        measurable: without an explicit escape hatch, a model that answers
        an unanswerable question cannot be distinguished from one that was
        never given the option to abstain.
        """
        context_blocks = [
            f"[{d['doc_id']}] {d['title']}\n{d['text']}" for d in retrieved_docs
        ]
        context = "\n\n".join(context_blocks)

        return (
            "Answer the question using ONLY the context below. "
            "If the context does not contain enough information to answer, "
            "reply with exactly: INSUFFICIENT_CONTEXT\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )

    def generate(self, question, retrieved_docs):
        """Generate an answer from retrieved context. Returns text + latency."""
        prompt = self.build_prompt(question, retrieved_docs)
        start = time.time()

        try:
            if self.model == "claude":
                if not self.anthropic_client:
                    return {"answer": "[Anthropic API not configured]",
                            "latency_ms": 0, "error": "missing api key"}
                message = self.anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = message.content[0].text

            elif self.model == "gpt":
                if not self.openai_client:
                    return {"answer": "[OpenAI API not configured]",
                            "latency_ms": 0, "error": "missing api key"}
                completion = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = completion.choices[0].message.content

            else:
                return {"answer": f"[Unknown model: {self.model}]",
                        "latency_ms": 0, "error": "unknown model"}

            return {"answer": answer.strip(),
                    "latency_ms": (time.time() - start) * 1000}

        except Exception as e:
            return {
                "answer": f"[Generation error: {str(e)[:100]}]",
                "latency_ms": (time.time() - start) * 1000,
                "error": str(e),
            }

    def query(self, question, top_k=3):
        """Full RAG turn: retrieve, then generate. Latencies reported separately."""
        retrieval_start = time.time()
        retrieved = self.store.search(question, top_k=top_k)
        retrieval_ms = (time.time() - retrieval_start) * 1000

        generation = self.generate(question, retrieved)

        return {
            "question": question,
            "retrieved": retrieved,
            "answer": generation["answer"],
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation["latency_ms"],
            "total_ms": retrieval_ms + generation["latency_ms"],
            "error": generation.get("error"),
        }
