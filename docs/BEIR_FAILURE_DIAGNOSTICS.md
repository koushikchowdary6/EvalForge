# Debugging missed BEIR retrievals

`evalforge-beir-failures` explains **which labeled queries a local BM25 baseline misses**, rather than only reporting aggregate hit rates. It uses the same BEIR corpus, queries, and qrels inputs as `evalforge-benchmark-beir` and does not download or redistribute the dataset.

```bash
evalforge-beir-failures \
  --corpus /path/to/scifact/corpus.jsonl \
  --queries /path/to/scifact/queries.jsonl \
  --qrels /path/to/scifact/qrels/test.tsv \
  --tokenizer unicode --k 5 --limit 20 \
  --output missed-queries.json
```

The report includes counts for **all** missed queries, even when `--limit` restricts the displayed examples. `failure_reason_counts` separates four diagnostic categories:

- `no_query_tokens`: the tokenizer produced no query terms
- `all_query_terms_oov`: none of the query terms appear in the corpus vocabulary
- `no_positive_relevant_overlap`: no positively labeled document received a positive BM25 score
- `relevant_docs_outside_top_k`: a relevant document scored positively but ranked below the requested cutoff

For the last category, `first_positive_relevant_rank` and `first_positive_relevant_score` show the earliest missed relevant document under the deterministic score/document-ID ordering. Other categories return null for these fields. `top_k_positive_results` shows the observed ranked IDs, scores, and relevance flags for each displayed failure.

**Interpretation:** A low-ranked relevant document suggests a ranking or tokenization issue worth investigating; out-of-vocabulary failures suggest a lexical coverage problem. These categories do not prove semantic relevance, answer correctness, or production safety. BM25 score ties can make the deterministic rank sensitive to document-ID tie-breaking; use the main benchmark's conservative tie-aware metrics when making acceptance decisions. Only queries with positive qrels are evaluated. Preserve the report's source SHA-256 fingerprints when comparing runs.

The CLI exits 0 for successful analysis and 2 for invalid inputs or failed writes. Existing output files are never overwritten.
