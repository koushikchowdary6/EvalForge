# GitHub Actions evaluation summary

EvalForge can publish a compact, human-readable comparison table in a workflow run's **Summary** tab while preserving its JSON output and quality-gate exit code.

After installing `evalforge-ai`, generate baseline and candidate artifacts from the **same dataset** and compare them:

```bash
evalforge-compare-artifacts baseline.json candidate.json \
  --tolerance 0.01 \
  --github-annotations \
  --github-step-summary
```

GitHub Actions automatically supplies `GITHUB_STEP_SUMMARY`. The CLI **appends** a Markdown table with BM25 lexical evidence hit rates at k=1,3,5, baseline/candidate values, deltas and a PASS/FAIL gate. It returns 1 for regressions beyond tolerance, 0 for passing comparisons and 2 for invalid inputs or summary output failures. The optional `--github-annotations` adds failure annotations.

For machine-readable evidence, also pass `--output comparison.json` (the output path must not already exist). Upload the JSON with `actions/upload-artifact` using `if: always()` so it survives a failing gate.

**Scope:** This gate compares deterministic *lexical retrieval evidence hit rate* on identical fixtures. It is not an LLM hallucination, safety, agent-quality, or production RAG guarantee. Review fixture representativeness before treating a passing result as a release decision.

Outside GitHub Actions, set `GITHUB_STEP_SUMMARY` to a writable file path before using `--github-step-summary`, or omit that flag.
