# Paired tokenizer retrieval failure analysis

Use this command to compare **the same BEIR corpus, queries, and relevance labels**
under EvalForge's ASCII and Unicode BM25 tokenizers. It reports exactly which
labeled queries gain or lose a positive-score relevant retrieval hit.

Install from this repository (Python 3.10+):

```bash
python -m pip install "git+https://github.com/koushikchowdary6/EvalForge.git"
evalforge-compare-tokenizers \
  --corpus /path/to/corpus.jsonl \
  --queries /path/to/queries.jsonl \
  --qrels /path/to/qrels/test.tsv \
  --k 5 --output tokenizer-comparison.json
```

Use BEIR-format data you have obtained and are permitted to process.
The command never downloads datasets or overwrites existing output.
Its report includes source SHA-256 fingerprints, paired counts of queries
recovered/lost, failures common to both tokenizers, and up to 20 query-level
changes (customize with `--limit`). Counts always include the full dataset,
even when the displayed query list is truncated.

**Interpretation:** a positive-score relevant hit is a lexical retrieval
diagnostic, not Recall@k, nDCG, semantic relevance, generated-answer accuracy,
or an AI safety measure. Comparing ASCII and Unicode tokenization is not
a comparison with dense embeddings or modern neural retrieval. No real-world
SciFact result is asserted until the command is actually run on independently
obtained data and its measurements are published.
