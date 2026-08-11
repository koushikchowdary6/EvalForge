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
- Dual-model support: Claude and GPT with identical category-specific rubrics
- RuleBasedGrader with stopword filtering for realistic key-term extraction
- 25+ comprehensive unit tests with pytest
- GitHub Actions CI/CD pipeline

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

3. **LLM Judge**: Makes real API calls to Claude with a category-specific rubric (factual/reasoning/instruction-following). Claude grades quality on a 0-10 scale converted to 0-1. Most sophisticated but slower and costlier.

## Results (v0.2)

Evaluated Claude Haiku 4.5 on 19 test cases across three categories. **All numbers regenerated after LLMJudgeGrader rewrite to use real API calls.**

| Metric | Score | Interpretation |
|--------|-------|---|
| **Exact Match** | 0/19 (0.0%) | Too strict; Claude elaborates beyond minimal answers |
| **Rule-Based** | 11/19 (57.9%) | Claude gets key concepts in ~58% of cases |
| **LLM Judge** | 16/19 (84.2%) | Claude's reasoning mostly sound; factual answers strong |
| **Avg Latency** | 1584ms | Haiku model is fast (~1-4s per call; includes LLM judge API overhead) |

**By Category:**

| Category | Exact | Rule | LLM Judge |
|---|---|---|---|
| **Factual (8)** | 0/8 (0%) | 8/8 (100%) | 8/8 (100%) |
| **Reasoning (5)** | 0/5 (0%) | 0/5 (0%) | 4/5 (80%) |
| **Instruction (6)** | 0/6 (0%) | 3/6 (50%) | 4/6 (67%) |

**Grader Agreement Analysis:**

1. **Exact Match is too strict across all categories** (0% pass rate). Claude elaborates beyond minimal answers—this isn't a failure, just the nature of conversational AI. Exact match is only useful for raw fact recall (capitals, symbols), not reasoning or instruction following.

2. **Rule-Based grader is strongest on factual questions** (8/8) where key terms are easily identifiable. Fails completely on reasoning (0/5) because answers use different words than expected, and partial on instruction-following (50%) when instructions require exact formats.

3. **LLM Judge (real API calls) is most forgiving** but also most realistic:
   - **Factual**: 8/8 (100%) — Claude knows facts and explains them clearly
   - **Reasoning**: 4/5 (80%) — Fails on one logical fallacy question (wet ground → must have rained) where Claude explained the fallacy; LLM judge wanted simpler answer
   - **Instruction**: 4/6 (67%) — Fails when exact matching is required (one question needed number "42", got "7"; another needed three specific colors, got "red, blue, green")

4. **Key divergence**: Rule-Based vs LLM Judge on reasoning (0% vs 80%). The real API judge recognizes that Claude's responses demonstrate sound reasoning even when they don't contain exact expected terms. The rule-based grader punishes stylistic variation; the LLM judge rewards conceptual correctness.

**What this means**: No single grader captures the full picture. Exact Match fails on elaboration. Rule-Based fails on flexibility. LLM Judge requires API calls but gives nuanced scoring. **Using all three together provides richer signal.**

## Project Structure

```
evalforge/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── evalforge/          # Main package
│   ├── __init__.py
│   ├── runner.py       # Evaluation runner
│   ├── graders.py      # Grading logic
│   └── models.py       # Model integrations
├── datasets/           # Test datasets
├── results/            # Evaluation results
└── tests/              # Unit tests
```

## Contributing

We're building this as a learning journey. All phases documented in `docs/phases.md`.

## License

MIT
