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

**v0.2 (Current)**: 
- Real LLMJudgeGrader with actual Anthropic/OpenAI API calls (not fake heuristics)
- Dual-model comparison: `main.py` runs Claude Haiku 4.5 and GPT-4o-mini head-to-head on the same dataset
- RuleBasedGrader with stopword filtering for realistic key-term extraction
- 25+ comprehensive unit tests with pytest (API tests marked and skipped in CI)
- GitHub Actions CI/CD pipeline (Python 3.10/3.11/3.12)

**v0.3 (Planned)**: Agent evaluation with tool-call tracing.

**v1.0 (Planned)**: RAG evaluation, multi-agent systems, safety evaluations.

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

> Note: `results/*.json` is gitignored to avoid committing API responses, but `results/report.json` (aggregate stats only, no raw model outputs) is tracked so these numbers are auditable.

## Project Structure

```
EvalForge/
├── README.md                # This file
├── main.py                  # Entry point — runs the full evaluation
├── requirements.txt         # Python dependencies
├── pytest.ini               # Test configuration
├── evalforge/               # Main package
│   ├── __init__.py
│   ├── runner.py            # Evaluation runner + model API clients (RealModelClient)
│   └── graders.py           # Grading logic (ExactMatch, RuleBased, LLMJudge)
├── datasets/
│   └── questions.json       # Test dataset
├── results/
│   └── report.json          # Aggregate stats (tracked; raw responses gitignored)
├── tests/
│   └── test_graders.py      # Unit tests
└── .github/workflows/
    └── tests.yml            # CI: runs pytest on push
```

## Contributing

Built as a learning project to demonstrate LLM-evaluation engineering. Contributions and issues welcome.

## License

MIT
