# Run EvalForge on an independently sourced BEIR dataset

This recipe uses **SciFact**, a public retrieval benchmark distributed by the
[BEIR project](https://github.com/beir-cellar/beir). The dataset is **not**
bundled with EvalForge. Review its upstream license and provenance before use.
This document does not claim that a benchmark was run or that any score was
achieved.

## 1. Obtain the dataset

Download SciFact using the [BEIR dataset instructions](https://github.com/beir-cellar/beir#datasets).
Extract it locally into a directory named `scifact` (do not commit the
downloaded corpus or queries to this repository). The expected layout is:

```text
scifact/
  corpus.jsonl
  queries.jsonl
  qrels/
    test.tsv
```

The paths and split are explicit: `qrels/test.tsv` is the **test** split,
not a training or development split. Record the upstream archive URL, download
date, dataset version if supplied, and any upstream checksum separately.
EvalForge records SHA-256 digests of the **extracted input files** in each
benchmark report; these are not an upstream archive authenticity check.

## 2. Install and run the baseline

From a clean Python 3.10+ environment:

```bash
python -m pip install "git+https://github.com/koushikchowdary6/EvalForge.git"
evalforge-benchmark-beir \
  --corpus scifact/corpus.jsonl \
  --queries scifact/queries.jsonl \
  --qrels scifact/qrels/test.tsv \
  --tokenizer ascii --k 1 3 5 10 \
  --output scifact-ascii.json
```

The command writes a JSON report or exits nonzero on invalid inputs. Keep
`scifact-ascii.json` as evidence. Output paths must be new: the CLI refuses
to overwrite existing reports. Do not report a metric until the command
actually completes successfully.

## 3. Compare a controlled tokenizer change

```bash
evalforge-benchmark-beir \
  --corpus scifact/corpus.jsonl \
  --queries scifact/queries.jsonl \
  --qrels scifact/qrels/test.tsv \
  --tokenizer unicode --k 1 3 5 10 \
  --output scifact-unicode.json

evalforge-compare-beir scifact-ascii.json scifact-unicode.json \
  --all-k --max-drop 0.02 --output scifact-comparison.json
```

The comparison gate can exit **1** if a cutoff regresses beyond the chosen
tolerance; this is a result to investigate, not a script failure to conceal.
It exits **2** for invalid or incompatible inputs. The `0.02` threshold
is an **illustrative policy**, not a validated SciFact acceptance criterion.
Do not use `--require-same-tokenizer` here: the tokenizer difference is
intentional. Use that flag when the experiment is supposed to hold tokenization
constant.

## 4. Report results responsibly

When publishing an actual run, include the source URL and download date,
dataset split, input SHA-256 digests from the reports, EvalForge commit SHA,
Python version, tokenizer, cutoffs, query inclusion/exclusion counts, and
the unedited JSON reports. Disclose any missing upstream checksum.

**Scope and limitations:** EvalForge currently evaluates offline **BM25
lexical retrieval**, not neural retrievers, generated answers, LLM safety,
or statistical significance. It scores queries with positive relevance
judgments; excluded queries must not be silently treated as failures or
successes. A SciFact run would provide independently sourced *retrieval*
evidence, not proof of production RAG quality or generalization to other
datasets.

## Optional: run both baselines with a provenance manifest

The standard-library runner automates the two CLI invocations and comparison,
while recording the input SHA-256 hashes, declared EvalForge revision, Python
version, source URL, download date, cutoffs, and tokenizer configurations.
It **does not download the dataset**, verify the declared revision, or invent
benchmark results. Use a fresh, empty output directory:

```bash
mkdir scifact-evidence
python scripts/run_scifact.py \
  --dataset scifact --output-dir scifact-evidence \
  --evalforge-revision "$(git rev-parse HEAD)" \
  --source-url "https://github.com/beir-cellar/beir" \
  --download-date YYYY-MM-DD --max-drop 0.02
```

Replace `YYYY-MM-DD` with the actual date the data was obtained. The Git
revision must identify the **installed** EvalForge version; if installing from
a different checkout, supply that exact revision instead. The runner refuses
nonempty output directories to protect existing artifacts and preserves its
manifest even when a subprocess fails. A comparison exit code of **1** is a
quality-gate failure with retained evidence, not proof of a broken runner.
The optional `--max-drop` sets the maximum absolute drop permitted at each cutoff\n(default `0.02`, inclusive range `0` to `1`). The chosen tolerance is saved in\n`manifest.json` and forwarded to the comparison CLI. Set it from a defensible\nbaseline or deployment policy; it is not a statistically validated threshold.\n\nNo independently sourced benchmark results are claimed until the commands
are executed and the artifacts reviewed.
