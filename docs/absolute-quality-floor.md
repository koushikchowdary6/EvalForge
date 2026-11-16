# Absolute quality floors for offline lexical retrieval

A regression-only check can pass when both baseline and candidate perform equally poorly. EvalForge's artifact comparator supports an **independent minimum evidence-hit-rate gate at k=3**.

```bash
evalforge-demo --json --output baseline.json
evalforge-demo --json --output candidate.json
evalforge-compare-artifacts baseline.json candidate.json \
  --tolerance 0.01 --min-hit-rate 0.50 \
  --output comparison.json --github-annotations --github-step-summary
```

The candidate must meet a BM25 lexical evidence hit rate of at least 0.50 at k=3, **and** avoid a decrease greater than 0.01 at any reported cutoff. The floor is inclusive; an exact 0.50 passes. The baseline is not subject to the floor.

- Exit 0: both gates pass.
- Exit 1: a regression or an absolute-floor failure (or both); the JSON report remains available.
- Exit 2: invalid inputs or output/summary errors.

The report includes `minimum_hit_rate_failed`, `min_hit_rate`, `bm25_regression_cutoffs`, and the absolute and delta scores. If you do not specify `--min-hit-rate`, the default is 0, preserving the previous regression-only gate.

These numbers are from deterministic **offline lexical retrieval fixtures**. A hit rate here is not an LLM correctness, prompt-injection resilience, agent safety, or production RAG quality score. Choose your floor from an independently validated dataset and document the rationale rather than treating 0.50 as a universal benchmark.
