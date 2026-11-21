# EvalForge

EvalForge is a Python toolkit for evaluating retrieval and, optionally, LLM and agent systems. The core offline path measures BM25 lexical retrieval on labeled document sets. It needs no API key and makes no model calls.

The bundled benchmarks use small, hand-authored datasets. They measure lexical retrieval, not generated-answer correctness, hallucination resistance, or production LLM or agent safety.

Requires Python 3.10 or newer. Licensed under MIT.

## Install

From a clone:

```bash
git clone https://github.com/koushikchowdary6/EvalForge.git
cd EvalForge
python -m pip install -e ".[test]"
```

Or install the package directly from GitHub:

```bash
python -m pip install "git+https://github.com/koushikchowdary6/EvalForge.git"
```

This installs the `evalforge-*` command-line tools. Provider-backed LLM and agent evaluation also needs API keys and the optional extra:

```bash
python -m pip install -e ".[providers]"
```

## Run the offline retrieval demo

No API keys required:

```bash
evalforge-demo
evalforge-bm25 --k 1 3 5 --summary-only
```

Write a JSON report and check it against a hit-rate floor. The floor applies only to lexical evidence at k=3 on the bundled fixture. It is not a production acceptance threshold. Output commands refuse to overwrite an existing file.

```bash
evalforge-demo --json --output evaluation.json
evalforge-validate evaluation.json --min-hit-rate 0.50
```

From a source checkout you can also run:

```bash
python -m evalforge.offline_rag_demo --k 1 3 5 --summary-only
python examples/quickstart_demo.py
python -m pytest -q
```

`evalforge-bm25` reports ranked document IDs plus precision@k, recall@k, nDCG@k, AP@k, MRR, and hit rate on the bundled 15-document, 14-question set. These are lexical retrieval measurements, not LLM accuracy claims.

## Evaluate your own labeled dataset

Custom corpora need `doc_id`, `title`, and `text`. Questions need `id`, `question`, `answerable`, and `relevant_doc_ids`. Answerable questions need at least one relevant document ID; unanswerable questions need none.

```bash
evalforge-dataset-audit \
  --corpus examples/customer_support_corpus.json \
  --questions examples/customer_support_questions.json \
  --strict
evalforge-custom-gate \
  --corpus examples/customer_support_corpus.json \
  --questions examples/customer_support_questions.json \
  --k 3 --min-hit-rate 0.50 \
  --output gate-evidence.json
evalforge-custom-summary gate-evidence.json --output gate-summary.md
```

Choose thresholds from your own labeled data, not from these example fixtures. Do not upload sensitive datasets to public CI artifacts.

## SciFact and BEIR

EvalForge can import and run a local BEIR SciFact directory (`evalforge-import-beir`, `evalforge-benchmark-beir`, `evalforge-run-scifact`, `evalforge-verify-scifact`, `evalforge-compare-beir`). The SciFact corpus is not bundled, and this repository does not contain a completed SciFact benchmark report. SciFact support exists, but it has not been benchmarked here yet. See `docs/REAL_BEIR_SCIFACT.md` for how to obtain the dataset and run it yourself.

## Other commands

These CLIs are also installed with the package:

- `evalforge-compare-lexical` compares ASCII and Unicode lexical BM25 baselines
- `evalforge-compare-artifacts` compares two saved offline evaluation reports
- `evalforge-report-md` writes a markdown summary of a report
- `evalforge-verify-evidence` checks a downloaded CI evidence bundle against its SHA-256 manifest (integrity only, not a signature or a quality judgement)

## Results from committed runs

The numbers below come from scripts and output files already in this repository. They are small, hand-authored evaluations, not independent public benchmarks. Reproduce them by running the same scripts.

**LLM evaluation** (`python main.py`, output in `results/report.json`): 19 test cases, Claude Haiku 4.5 vs GPT-4o-mini, 38 evaluations total.

| Model | Exact match | Rule-based | LLM judge |
| --- | --- | --- | --- |
| Claude | 0.0% | 57.9% | 68.4% |
| GPT | 0.0% | 47.4% | 68.4% |

This path needs provider API keys.

**RAG evaluation** (`python main_rag.py`, output in `results/rag_report.json`): a 3-question, k=3 smoke run, not the full 14-question top-k sweep the harness can run. At k=3 the committed report records precision@3 of 44.4%, recall@3 of 100%, MRR of 1.0, and hit rate of 1.0. This path also needs API keys.

**Agent evaluation** (`python main_agent.py`, output in `results/agent_report.json`): a 6-task injection-resistance ablation with the injection-defense prompt disabled. It is not a result from the full 24-task suite. This path also needs API keys.

If a metric is not in these files, it is not a reported result of this project.

## Limitations

- Bundled datasets are small and hand-authored. Scores on them do not establish production retrieval quality or model safety.
- The default offline path is BM25 lexical retrieval. It does not score generated answers, groundedness, or hallucinations.
- Provider-backed LLM, RAG generation, and agent runs require credentials, cost money, and are separate from the API-free demo.
- SciFact and other BEIR datasets are not shipped with the repo, and no SciFact scores are committed here.
- Artifact verification checks file hashes against a local manifest. It does not prove benchmark quality or dataset authenticity.
- There is no published PyPI release claimed here. Install from this GitHub repository or a local checkout.
- EvalForge is an evaluation toolkit still in early development (package version 0.1.0).

## License

MIT. See [LICENSE](LICENSE).
