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

**v0.1 (Current)**: Basic LLM evaluation with exact-match, rule-based, and LLM-as-judge grading. Model comparison. Cost and latency tracking.

**v0.2 (Planned)**: Agent evaluation with tool-call tracing.

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

## Results (v0.1)

Evaluated Claude Haiku 4.5 on 19 test cases across three categories:

| Metric | Score | Interpretation |
|--------|-------|---|
| **Exact Match** | 0/19 (0.0%) | Too strict; Claude elaborates beyond minimal answers |
| **Rule-Based** | 11/19 (57.9%) | Claude gets key concepts in ~58% of cases |
| **LLM Judge** | 19/19 (100.0%) | Claude's responses are coherent and reasonable quality |
| **Avg Latency** | 1512ms | Haiku model is fast (~1-4s per call) |

**Key Insight:** The divergence between graders is itself valuable. Exact Match fails almost everywhere because real models elaborate. Rule-Based and LLM Judge track together much better, suggesting **no single grader is sufficient** — evaluation quality improves with multiple approaches.

**Grader Agreement Analysis:**
- **Exact vs Rule-Based**: Exact is stricter (0% vs 57.9%); perfect correlation: when Exact passes, Rule-Based always passes.
- **Rule-Based vs LLM Judge**: High disagreement (57.9% vs 100%). LLM judge is more forgiving; 8 cases fail rule-based but pass LLM judge, likely because the model captured concepts despite not containing exact key terms.
- **Pattern**: Factual and instruction-following categories show highest exact-match failure (0/8, 0/6). Reasoning questions fail rule-based more often (1/5 pass) because elaboration changes sentence structure.

**By Category:**
- **Factual (8 cases)**: 0/8 exact match, 4/8 rule-based, 8/8 LLM judge. Models know facts but phrase differently.
- **Reasoning (5 cases)**: 0/5 exact match, 1/5 rule-based, 5/5 LLM judge. Complex logic fails rule extraction; LLM judge sees reasoning quality.
- **Instruction Following (6 cases)**: 0/6 exact match, 6/6 rule-based, 6/6 LLM judge. Models follow instructions well when evaluated fairly.

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
