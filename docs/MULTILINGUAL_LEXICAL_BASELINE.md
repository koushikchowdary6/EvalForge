# Multilingual lexical retrieval sanity fixture

This **hand-authored, deterministic** fixture checks whether the offline BM25
baseline can retrieve literal word matches in English, Hindi (Devanagari),
Japanese, Arabic, and canonically equivalent accented French. It includes one
Japanese unanswerable query with no lexical match. It is **not** independently
sampled, not cross-lingual translation, and not an LLM/embedding accuracy
benchmark. The relevance labels were written specifically for these examples.

Run both baselines on the exact same inputs:

```bash
python -m evalforge.offline_rag_demo \
  --corpus datasets/multilingual_lexical_corpus.json \
  --questions datasets/multilingual_lexical_questions.json \
  --k 1 --tokenizer ascii > /tmp/multilingual-ascii.json

python -m evalforge.offline_rag_demo \
  --corpus datasets/multilingual_lexical_corpus.json \
  --questions datasets/multilingual_lexical_questions.json \
  --k 1 --tokenizer unicode > /tmp/multilingual-unicode.json

pytest -q tests/test_multilingual_lexical_fixture.py
```

Expected evidence-only behavior at k=1: the ASCII baseline finds the English
literal match but no positive-scoring document for the four other answerable
queries; the Unicode baseline finds all five literal matches. A zero-score
document-ID tie can yield an apparently correct top-1 ID but **must not** be
counted as lexical evidence. The unanswerable query has no positive lexical
match in either mode. The test verifies these invariants and deterministic CLI
output; the JSON reports contain corpus/question SHA-256 fingerprints.

## Interpretation and limitations

- Unicode mode preserves attached combining marks and normalizes canonical
  accents. It does not stem words, segment Japanese compounds, translate
  between languages, or match semantic paraphrases.
- The deliberately simple labels and isolated vocabulary make this a
  **regression fixture**, not a statistical estimate of multilingual retrieval
  performance. A realistic evaluation requires independently collected queries,
  representative negative examples, verified relevance judgments, and a
  held-out protocol.
- This is retrieval-only: no answer generation, hallucination grading, agent
  behavior, or guardrail effectiveness is measured here.

## Query vocabulary coverage diagnostics

The offline JSON report now distinguishes two reasons for absent lexical
retrieval evidence. `query_has_no_selected_tokens` means the selected
tokenizer extracted no words at all; `query_all_selected_terms_oov` means it
extracted words, but none appear in the indexed corpus. For each query,
`query_unique_selected_term_count`, `query_oov_term_count`, and
`query_oov_term_fraction` describe the vocabulary gap using **unique**
selected query terms, matching BM25's query-term set semantics. A query with
zero selected terms has a `null` OOV fraction, not a misleading 0% or 100%.

`query_vocabulary_diagnostics` aggregates fully and partially OOV queries;
the `by_top_k` report separates answerable and unanswerable fully OOV
queries. These values are descriptive corpus/tokenizer coverage checks,
**not** semantic out-of-distribution detection, model accuracy, or proof that
a retrieved passage answers the question.


## Query-dependent relevance recall ceiling

The report distinguishes **corpus indexability** from **query overlap**.
`relevant_indexable_recall_ceiling` is the fraction of labeled-relevant
documents with any tokens under the selected tokenizer. A document can be
indexable yet have *zero* positive BM25 score for a particular question.
`relevant_positive_score_document_count` and
`relevant_positive_score_recall_ceiling` count only labeled-relevant
documents with positive scores for that exact question. This is the maximum
evidence-only recall obtainable at any cutoff with the current query and
tokenizer; it is not an achieved recall, and a lexical match does not prove
that the passage answers the question.

The `query_relevance_coverage` section identifies answerable questions with
zero or partially covered relevance labels. Each `by_top_k` summary includes
the mean query-specific ceiling and the number of answerable cases with a
relevance gap. An unanswerable-only benchmark reports a `null` mean rather
than a fabricated zero. These diagnostics do not change rankings or labels
and are not measures of semantic recall or independently sampled accuracy.

## Diagnosing *why* labeled relevance is lexically unreachable

The report now assigns each **answerable** question exactly one
`query_relevance_gap_reason` and groups case IDs under
`query_relevance_coverage.reason_case_ids`. The mutually exclusive reasons
are:

- `full_positive_relevance`: every labeled-relevant document has a positive
  BM25 score for this query (a ceiling, not an achieved top-k score).
- `no_selected_query_tokens`: the selected tokenizer extracted no terms.
- `all_selected_query_terms_oov`: query terms exist, but none are indexed
  anywhere in the corpus.
- `no_positive_relevant_overlap`: some query terms occur in the corpus, but
  none of the labeled-relevant documents receive a positive score.
- `partial_positive_relevant_overlap`: some, but not all, labeled-relevant
  documents receive positive scores.

Unanswerable cases have a `null` reason; they are never assigned a fabricated
relevance denominator. Reasons are **cutoff-independent** and distinguish
tokenization, corpus vocabulary, and relevance-label overlap failures without
claiming semantic retrieval, translation, or independently validated accuracy.
The per-case ceiling and zero-score evidence metrics remain authoritative.
