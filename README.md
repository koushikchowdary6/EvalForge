# EvalForge

An LLM and AI Agent evaluation platform. Run tests against AI models and agents, grade responses, identify failures, compare performance, and generate insightful reports.

## What is EvalForge?

EvalForge automates the evaluation of AI systems. Instead of manually checking whether an LLM answered correctly, whether an agent used the right tools, or whether a RAG system found the right information, EvalForge does it for you — fast, consistently, and repeatably.

Think of it like a grading system for AI homework. You write the test questions, tell EvalForge what correct answers look like, and it runs the tests against different models, grades the results, and tells you which models performed best and why others failed.

## Why does evaluation matter?

- **Speed**: Iterate on prompts and models 10x faster with automated feedback
- **Confidence**: Know which changes actually improved your system (vs. gut feeling)
- **Reproducibility**: Same test, same grading, every run — no human bias
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
- Retrieval metrics: precision@k, recall@k, MRR, hit rate
- LLM-judged correctness and groundedness; hallucination detection via abstention
- Top-k sweep support across k=1,3,5 to quantify precision/recall tradeoffs
- 38 comprehensive unit tests

**v0.4**: Agent evaluation (tool-calling + safety; full task suite implemented, tracked artifact below is an ablation run)
- 24 tool-calling tasks over 5 categories: single-tool, multi-tool, no-tool, destructive-refusal, injection-resistance
- Trajectory grading: unsafe paths fail even with correct answers
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

## Evaluation Methodology

EvalForge uses **three independent grading methods** to triangulate response quality:

1. **Exact Match**: Binary pass/fail on word-for-word comparison (case-insensitive). Best for factual recalls ("capital of Australia" → "Canberra"). Fails for elaborated answers.

2. **Rule-Based**: Flexible scoring that extracts key concepts (nouns, verbs, important adjectives) from the expected answer and checks if the model's response contains them. Filters common stopwords to avoid penalizing for articles/prepositions. Better for real-world answers that elaborate.

3. **LLM Judge**: Makes real API calls (Claude or GPT) with a category-specific rubric (factual/reasoning/instruction-following). The judge grades quality on a 0-10 scale converted to 0-1. Most sophisticated but slower and costlier.

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
