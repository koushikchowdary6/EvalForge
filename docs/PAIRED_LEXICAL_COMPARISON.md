# Paired ASCII vs Unicode BM25 retrieval audit

This API-free audit evaluates **both tokenizers on exactly the same corpus,
questions, relevance labels, and top-k cutoffs**. It makes improvements and
regressions visible instead of comparing only headline scores from separate
runs. The report is deterministic, with corpus and question SHA-256
fingerprints to support reproduction.

## Run

```bash
python -m evalforge.compare_lexical_baselines --k 1 3 5
python -m pytest tests/test_paired_lexical_comparison.py -q
```

The default input is the hand-authored multilingual lexical sanity fixture.
To use your own benchmark, pass `--corpus path.json --questions path.json`.
The same bounded JSON loader and benchmark-label validation used by the
offline BM25 demo reject malformed, duplicate-key, or unsafe inputs.

## Read the report

For each cutoff, `by_top_k` reports how many **answerable** queries both
tokenizers hit, only ASCII hits, only Unicode hits, or neither hits. Hit
means at least one labeled-relevant document received a **positive** BM25
score within top-k; zero-score document-ID ties never count as evidence.

The paired hit-rate and mean-recall deltas are **Unicode minus ASCII**. A
negative delta is a regression, not an error to hide. `cases` includes
per-question positive-score retrieved IDs and recall deltas, making
unexpected rank shifts inspectable.

Unanswerable questions are **excluded from retrieval-quality denominators**.
Their positive lexical-match counts are shown separately; a lexical match
does not mean the query is answerable. The paired unanswerable matrix reports
`unanswerable_both_match`, `unanswerable_ascii_only_match`,
`unanswerable_unicode_only_match`, and `unanswerable_neither_match`. The
per-case Unicode-minus-ASCII match delta is -1, 0, or +1. Separate match
rates and the paired rate delta expose **offsetting regressions and gains**
that equal aggregate counts would conceal. These are lexical-evidence flags,
**not** hallucination or answer-quality rates. When there are no unanswerable
questions, these rates are JSON `null`; an unanswerable-only benchmark likewise
uses `null` for undefined answerable hit/recall rates.

## Interpretation limits

- This is a paired, descriptive comparison of lexical BM25 tokenization,
  not an embedding, cross-lingual semantic, generative, or agent benchmark.
- The default fixture was hand-authored for regression coverage, **not**
  independently sampled. The deltas are not production effectiveness
  estimates, statistical significance tests, or generalization claims.
- More Unicode lexical overlap can **hurt** top-k relevance when
  nonrelevant documents also gain matching tokens. Inspect both gains and
  regressions, and validate with independently judged held-out data before
  making accuracy claims.
