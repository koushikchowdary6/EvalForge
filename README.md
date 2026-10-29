# EvalForge

An LLM and AI Agent evaluation platform. Run tests against AI models and agents, grade responses, identify failures, compare performance, and generate insightful reports.

## What is EvalForge?

EvalForge automates the evaluation of AI systems. Instead of manually checking whether an LLM answered correctly, whether an agent used the right tools, or whether a RAG system found the right information, EvalForge does it for you — fast, consistently, and repeatably.

Think of it like a grading system for AI homework. You write the test questions, tell EvalForge what correct answers look like, and it runs the tests against different models, grades the results, and tells you which models performed best and why others failed.

## Why does evaluation matter?

- **Iteration**: Re-run the same prompt/model test cases automatically; speedups depend on the model, dataset, and judge configuration
- **Confidence**: Know which changes actually improved your system (vs. gut feeling)
- **Reproducibility**: Versioned test cases and deterministic rule-based graders support repeatable comparisons; LLM judges can still vary and inherit bias
- **Debugging**: Understand *why* something failed, not just that it did

## Project Status

**v0.2**: LLM evaluation (dual-model: Claude vs GPT)
- Real LLMJudgeGrader with actual Anthropic/OpenAI API calls
- Dual-model comparison on 19 test cases
- RuleBasedGrader with stopword filtering
- 14 comprehensive unit tests

**v0.3**: RAG evaluation (retrieval + generation; full benchmark suite implemented, tracked artifact below is a smoke run)
- Security-themed 15-document corpus, 14 questions (incl. 2 unanswerable)
- Embedding cache + cosine similarity (zero new dependencies)
- Retrieval metrics: precision@k, recall@k, **nDCG@k**, **MAP@k**, MRR, hit rate
- LLM-judged correctness and groundedness; hallucination detection via abstention
- Top-k sweep support across k=1,3,5 to quantify precision/recall tradeoffs; precision@k uses the requested k as its denominator, including when fewer documents are returned
- Binary-relevance nDCG@k discounts lower-ranked evidence, counts repeated document IDs only once, and is included in per-case JSON, aggregate reports, and the terminal summary; unanswerable cases are excluded from retrieval averages
- Binary average precision at k (AP@k) measures precision at each newly found relevant document; duplicate hits consume rank slots without earning credit, and missed relevant documents reduce the denominator-normalized score. The report averages AP@k over answerable queries as MAP@k, excluding retrieval failures and unanswerable cases.
- 38 comprehensive unit tests

**v0.4**: Agent evaluation (tool-calling + safety; full task suite implemented, tracked artifact below is an ablation run)
- 24 tool-calling tasks over 5 categories: single-tool, multi-tool, no-tool, destructive-refusal, injection-resistance
- Trajectory grading: unsafe paths fail even with correct answers
- **Tool-sequence accuracy**: measures whether expected tools were called in the required order, separately from set-based tool-selection accuracy; per-category and overall summaries exclude provider errors
- Safety metrics: injection resistance, destructive refusal, unnecessary call rate
- Failure categorization: 6 categories with safety violations ranked first
- Token-based cost estimation (Claude vs GPT pricing)
- Ablation: measures whether injection-defense system prompt has measurable effect
- 58 comprehensive unit tests

**v1.0 (Planned)**: Multi-agent systems, advanced safety evaluations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create a .env file with your API keys
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo "OPENAI_API_KEY=your_key_here" >> .env

# Run an evaluation
python main.py
```

## Reproducible embedding evaluation

RAG embedding cache keys are SHA-256 digests of the **embedding model
and input text**, rather than raw document/query strings. This prevents
plaintext corpus content from appearing in JSON cache keys while still
avoiding cross-model vector reuse. Existing plaintext-key cache entries are
ignored; delete old `results/embedding_cache.json` files to remove previously
written plaintext, or run a fresh evaluation and save the migrated cache.
Re-embedding discarded entries requires a configured OpenAI API key.

**This is not encryption:** predictable text may be guessed from its hash, and
the embedding vectors themselves may be sensitive. Protect cache files with
appropriate filesystem permissions and do not commit them. Offline tests
cover cache hits, model isolation, legacy-key rejection, and non-disclosure of
raw text in newly written keys. Embedding API responses must contain exactly
one result for the single requested input (with index 0, when supplied);
missing, duplicate, or misindexed results fail before entering the cache or
retrieval ranking. These response-contract tests use mocked providers and do
not claim any measured production failure rate.

---

## Embedding request preflight

Each RAG document is validated as **title + newline + text** before an index
build starts. Inputs over **12,000 characters**, blank/invisible-only strings,
and invalid Unicode surrogates are rejected before cache lookup or a paid
embedding request. The same boundary applies to direct embedding calls and
queries, including when a custom embedding client is supplied. Invalid corpus
reloads preserve the previous in-memory corpus/index.

This is a conservative character-based resource guard, **not** a provider
token-count guarantee or proof of model accuracy. A larger corpus may require
chunking documents rather than raising the limit. Offline regression tests
exercise oversized later documents, query validation, cache/API non-use, and
last-good index preservation.

## Bounded RAG generation and judge evidence

The RAG generator rejects malformed JSON evidence (including non-finite
numbers, circular values, and incomplete document records) and serialized
question/document payloads above **40,000 characters**, *before* a paid
generation API call. These become generation errors excluded from answer
quality metrics, not model failures or hallucinations. API-free tests cover
both providers and valid small requests.

Separately, correctness and groundedness judges reject serialized untrusted
evidence above **40,000 characters** before making a paid judge API call.
These cases are **unscored** (`score: null`, `parse_ok: false`) rather than
silently truncating documents or assigning a low-quality grade. Aggregate
coverage must be considered alongside scores. Both limits are practical
character-based resource guards, **not** guarantees that shorter prompts fit
every provider's context window.

## RAG benchmark provenance

Newly generated RAG summary reports include a `benchmark_provenance` object
with SHA-256 fingerprints of the **validated, parsed** corpus and question
datasets, plus the embedding and judge model identifiers. The fingerprints
use canonical JSON (sorted object keys, preserved array order), so changing
JSON indentation does not make the benchmark appear new, but editing evidence,
case labels, or case order does. Failed validation or index construction
cannot retain a previous dataset's fingerprints.

Compare these identifiers alongside the generation model and top-k settings
before attributing score changes to an AI system. Hashes are identifiers,
**not proof of ground-truth quality, model determinism, or data secrecy**.
Historical tracked reports predate this feature and must not be retroactively
described as provenance-verified.

## RAG corpus integrity

Corpus JSON is decoded with duplicate-key rejection at every object depth.
Without this check, standard JSON decoding silently takes the last value for
repeated `doc_id`, `title`, or `text` keys, making retrieved evidence and
evaluation results ambiguous. Corpus titles and document text must contain
visible characters and valid Unicode: lone UTF-16 surrogates and invisible-only
evidence fail at load time, before embedding or cache writes. Visible Unicode
and adversarial text containing hidden controls remain supported for security
evaluations. An invalid reload preserves the last-good in-memory corpus/index.
This is an offline input-integrity control, not validation of whether the
corpus evidence is factually correct.

## RAG prompt-injection trust boundaries

Retrieved documents and generated answers can contain adversarial text. The
RAG generator now sends explicit untrusted-context rules in the **system**
message for both Claude and GPT, rather than relying only on instructions
embedded in the same user prompt as the retrieved content. The LLM judge
likewise receives a higher-priority instruction to treat documents and model
answers as evidence, not grading directives. API-free regression tests verify
that these boundaries are present in both providers' request payloads.

These controls are **defense in depth, not proof of prompt-injection immunity**.
The project does not claim measured attack resistance without an adversarial
benchmark and actual model-run results.

## Agent benchmark label validation

The agent runner validates the complete task dataset **before** initializing
a model client or sending paid requests. It rejects duplicate/empty case IDs,
missing prompts, malformed tool lists and argument maps, and contradictory
expected/forbidden tool labels. Injection-resistance and destructive-refusal
cases must name at least one forbidden tool; otherwise a benchmark could
report a misleading 100% safety rate without checking any prohibited action.

For repeated tool calls, `expected_args` can specify a list of argument
objects in invocation order, one per expected occurrence. This prevents two
identical calls from receiving full argument credit when the benchmark expects
different recipients, queries, or other action targets. Existing single-object
argument expectations remain backward compatible. Validation rejects
mismatched list lengths, unknown tool-parameter names, empty expectations,
and non-string argument labels before any model/API requests. This prevents
a benchmark typo from being misreported as an agent's argument failure.

Validation also runs when a previously loaded task list is evaluated, and
failed reloads clear stale benchmark state. The bundled 24-task dataset and
malformed-label cases are covered by offline regression tests.

## Agent cost measurement coverage

Agent cost figures are **static token-based estimates**, not provider invoices.
A run is priced only when both input and output token counts are present,
including explicit zero counts. If any OpenAI round-trip omits usage, the
trajectory remains gradable but its cost is **unavailable**, rather than
silently treating the missing tokens as zero. Reports expose cost coverage,
observed partial spend, and a full total only when every task has a measured
estimate. A provider outage may still leave partial observed spend.

The tracked historical agent ablation report predates this missing-usage
contract and has **not** been rerun or retroactively recalculated. Model
pricing and provider billing may change independently of these estimates.

## Evaluation Methodology

EvalForge uses **three independent grading methods** to triangulate response quality:

1. **Exact Match**: Binary pass/fail on word-for-word comparison (case-insensitive). Best for factual recalls ("capital of Australia" → "Canberra"). Fails for elaborated answers.

2. **Rule-Based**: Flexible scoring that extracts key concepts (nouns, verbs, important adjectives) from the expected answer and checks if the model's response contains them. Filters common stopwords to avoid penalizing for articles/prepositions. Better for real-world answers that elaborate.

3. **LLM Judge**: Makes real API calls (Claude or GPT) with a category-specific rubric (factual/reasoning/instruction-following). The judge grades quality on a 0-10 scale converted to 0-1. Most sophisticated but slower and costlier.

### General-purpose LLM judge coverage (current runner)

The general LLM judge now marks each grade with `parse_ok`. Malformed,
truncated, unavailable, or provider-error judgments are **missing measurements**,
not failed model answers. In new `results/report.json` summaries,
`llm_judge.evaluated` counts valid grades, `unavailable` counts missing
grades, and the pass rate uses **evaluated**, not total cases, as its
denominator. When no valid judge result exists, `pass_rate` is `N/A`
rather than a misleading 0%. Exact-match and rule-based rates still use
all completed model responses.

General judge evidence is JSON-serialized with non-finite values rejected
and capped at **40,000 characters** before paid requests. Invalid evidence
is marked `parse_ok: false`; it is not a model-quality measurement. This
limit is a resource guard, not a guarantee of provider context-window fit.
The tracked v0.2 historical report below predates these coverage fields;
its numbers have **not** been retroactively recomputed.

## Results (v0.2) — Claude vs GPT

Ran a head-to-head evaluation of **Claude Haiku 4.5** and **GPT-4o-mini** on the same 19 test cases (38 total evaluations). All numbers come from an actual `python main.py` run committed to `results/report.json` — reproduce them yourself.

| Grader | Claude Haiku 4.5 | GPT-4o-mini |
|--------|------------------|-------------|
| **Exact Match** | 0/19 (0.0%) | 0/19 (0.0%) |
| **Rule-Based** | 11/19 (57.9%) | 9/19 (47.4%) |
| **LLM Judge** | 13/19 (68.4%) | 13/19 (68.4%) |
| **Avg Latency** | 1558ms | 1443ms |

**What the head-to-head shows:**

1. **Exact match is useless for conversational models** — both score 0%. Both models elaborate ("The capital of Australia is Canberra..." instead of "Canberra"). This is expected, not a defect: it's exactly why rule-based and LLM-judge graders exist.

2. **Claude edges out GPT on rule-based matching** (57.9% vs 47.4%). Claude's phrasing more often contains the expected key terms. This is a grading-method artifact, not necessarily a quality gap — the rule-based grader rewards lexical overlap, not correctness.

3. **The two models tie on the LLM judge** (both 68.4%) — the grader that actually assesses answer quality rather than word overlap. When judged on substance, Claude and GPT-4o-mini are neck-and-neck on this dataset.

4. **GPT is slightly faster on average** (1443ms vs 1558ms), though both sit in the 0.5–4s range per call.

**Why the LLM Judge and Rule-Based graders disagree:** rule-based punishes stylistic variation (a correct answer worded differently fails); the LLM judge rewards conceptual correctness. The gap between them (e.g. Claude 57.9% rule-based vs 68.4% judge) is the single most useful signal here — it quantifies how much a keyword grader *under*-credits models that paraphrase. **No single grader is sufficient; the divergence between them is the finding.**

> Note: `results/*.json` is gitignored to avoid committing API responses, but aggregate reports (`report.json`, `rag_report.json`, `agent_report.json`) are tracked so all numbers are auditable.

**Artifact integrity:** General, RAG, and agent evaluation runners publish each JSON
artifact through a flushed, fsynced temporary file and atomic replacement.
Serialization errors (including `NaN`/`Infinity`) or a failed replacement do
not truncate the previous artifact. A general evaluation aborted by a provider
error removes its stale report when results are saved. The results and report
are **separate atomic files**, not a cross-file transaction; consumers should
not assume a pair was published together.

## Results (v0.3) — RAG Evaluation

The tracked artifact in `results/rag_report.json` is a **3-question k=3 smoke run**, not the full 14-question top-k sweep implemented by the evaluation harness. Publishing the artifact exactly as it exists keeps the benchmark provenance auditable.

| Metric | Tracked k=3 smoke run |
|--------|-----------------------|
| Questions / evaluations | 3 / 3 |
| Precision@3 | 44.4% |
| Recall@3 | 100.0% |
| MRR | 1.000 |
| Hit Rate | 100.0% |
| Correctness | 90.0% |
| Groundedness | 93.3% |
| Avg total latency | 1904ms |

The tracked smoke-run artifact predates the fixed-k precision denominator change. Its historical values have not been recalculated; it also predates nDCG@k and MAP@k. Rerun the benchmark before comparing historical values to new retrieval reports.

This sample is useful as an end-to-end pipeline check, but **three questions are too few for comparative retrieval claims**. The larger 14-question corpus and k=1/3/5 sweep remain implemented in the project; a full reproducible run should be committed before treating those results as benchmark evidence.

## Results (v0.4) — Agent Evaluation

The tracked `results/agent_report.json` is a targeted **6-task injection-resistance ablation with the explicit injection-defense prompt disabled**. It is not a 24-task full-suite result.

| Metric | Tracked ablation |
|--------|------------------|
| Tasks | 6 injection-resistance tasks |
| Task Success Rate | 100.0% (6/6) |
| Tool Selection Accuracy | 100.0% |
| Argument Accuracy | 100.0% |
| Injection Resistance | 100.0% |
| Unnecessary Call Rate | 0.0% |
| Avg Latency | 2576ms |
| Avg Turns | 2.0 |
| Estimated Cost | $0.0167 |

**What this run supports:** all six tested payloads were resisted even with the dedicated injection-defense instruction disabled. **What it does not support:** broad claims about destructive-action refusal, no-tool behavior, or the complete 24-task suite, because those categories are absent from this tracked artifact. The result also suggests these six payloads are not sufficiently discriminative for the tested model; stronger adversarial cases are needed for a meaningful defense ablation.

## Project Structure

```
EvalForge/
├── README.md                      # This file
├── main.py                        # v0.2: LLM evaluation entry point
├── main_rag.py                    # v0.3: RAG evaluation entry point
├── main_agent.py                  # v0.4: Agent evaluation entry point
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Test configuration
├── evalforge/
│   ├── __init__.py
│   ├── runner.py                  # v0.2: LLM eval runner + model API clients
│   ├── graders.py                 # v0.2: Grading logic (ExactMatch, RuleBased, LLMJudge)
│   ├── rag.py                     # v0.3: RAG pipeline (embedding, retrieval, generation)
│   ├── rag_graders.py             # v0.3: Retrieval metrics + LLM graders
│   ├── rag_runner.py              # v0.3: RAG evaluation runner
│   ├── agent.py                   # v0.4: Tool-calling agent
│   ├── agent_tools.py             # v0.4: Mock tool registry
│   ├── agent_graders.py           # v0.4: Trajectory graders
│   └── agent_runner.py            # v0.4: Agent evaluation runner
├── datasets/
│   ├── questions.json             # v0.2: LLM eval dataset (19 cases)
│   ├── rag_corpus.json            # v0.3: RAG corpus (15 security docs)
│   ├── rag_questions.json         # v0.3: RAG questions (14 cases)
│   └── agent_tasks.json           # v0.4: Agent tasks (24 cases)
├── results/
│   ├── report.json                # v0.2: LLM eval results (tracked)
│   ├── rag_report.json            # v0.3: RAG eval results (tracked)
│   └── agent_report.json          # v0.4: Agent eval results (tracked)
├── tests/
│   ├── test_graders.py            # v0.2: Grader unit tests (14 tests)
│   ├── test_rag_graders.py        # v0.3: RAG grader unit tests (38 tests)
│   └── test_agent_graders.py      # v0.4: Agent grader unit tests (58 tests)
└── .github/workflows/
    └── tests.yml                  # CI: runs pytest on push (Python 3.10/3.11/3.12)
```

## Contributing

Built as a learning project to demonstrate LLM-evaluation engineering. Contributions and issues welcome.

## License

MIT
