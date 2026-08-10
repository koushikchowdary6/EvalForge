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

## Results (v0.1)

Evaluated Claude Haiku 4.5 on 19 test cases across three categories:

| Metric | Score |
|--------|-------|
| **Exact Match** | 0/19 (0.0%) |
| **Rule-Based** | 11/19 (57.9%) |
| **LLM Judge** | 19/19 (100.0%) |
| **Avg Latency** | 1512ms |

**Key Finding:** Claude produces high-quality, coherent responses (100% LLM judge pass rate), but rarely matches expected answers word-for-word (0% exact match). The rule-based grader (checking for key terms) is a better fit for real-world evaluation — 57.9% pass rate shows Claude gets the core concepts right but often elaborates beyond the minimal expected answer.

**By Category:**
- Factual questions: 0/8 exact match, but 4/8 have key terms
- Reasoning questions: Strong conceptual answers, penalized for format
- Instruction following: Mixed results on format compliance

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
