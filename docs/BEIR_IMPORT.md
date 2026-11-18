# Import an independently sourced BEIR-style retrieval dataset

EvalForge can convert **locally obtained** BEIR-format datasets into its labeled
retrieval format without a network call or API key. This is a format adapter,
**not** a bundled third-party benchmark or a claim of independent benchmark results.

Obtain a dataset from its original publisher and check its license, permitted
uses, provenance, split definition, and any personal-data restrictions. Do not
commit third-party data or generated outputs unless redistribution is permitted.

Expected local files:

- `corpus.jsonl`: one object per line with `_id`, `title` (optional), `text`
- `queries.jsonl`: one object per line with `_id`, `text`
- `qrels/test.tsv`: tab-separated `query-id`, `corpus-id`, `score` columns

After `pip install .` (or installing the project from GitHub):

```bash
evalforge-import-beir \
  --corpus /path/to/corpus.jsonl \
  --queries /path/to/queries.jsonl \
  --qrels /path/to/qrels/test.tsv \
  --output-corpus /tmp/beir-corpus.json \
  --output-questions /tmp/beir-questions.json

evalforge-dataset-audit \
  --corpus /tmp/beir-corpus.json \
  --questions /tmp/beir-questions.json --strict

evalforge-bm25 --corpus /tmp/beir-corpus.json \
  --questions /tmp/beir-questions.json --k 1 3 5 --summary-only
```

The importer retains **only queries with at least one positive relevance
judgment** (`score > 0`). Missing qrels and nonpositive judgments do not
prove a query is unanswerable. The importer excludes these queries rather
than inventing negative labels. It rejects duplicate IDs, duplicate qrel
pairs, unknown references, invalid scores, and datasets without positive
labels. Results are deterministically sorted by ID. It refuses to overwrite
existing output files.

**Benchmark reporting:** Always record the upstream dataset name/version,
split, license, download date, preprocessing, corpus and query counts, excluded
queries, tokenizer, top-k, and resulting SHA-256 fingerprints. Compare only
like-for-like runs. BM25 measures lexical retrieval; it is not an LLM
generation, hallucination, or safety evaluation. A genuine benchmark result
must be measured on the actual externally obtained data, not inferred from
this adapter or the repository's hand-authored examples.
