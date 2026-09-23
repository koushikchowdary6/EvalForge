# EvalForge — Engineering Brief

## Problem

LLM applications are difficult to improve safely when evaluation is manual, inconsistent, or limited to answer correctness. EvalForge treats evaluation as an engineering system: repeatable datasets, multiple graders, retrieval metrics, agent-trajectory checks, safety tests, latency measurements, and cost reporting.

## What is implemented

- LLM evaluation with exact-match, rule-based, and LLM-judge graders.
- RAG evaluation with precision@k, recall@k, MRR, hit rate, correctness, groundedness, and abstention/hallucination checks.
- Agent evaluation across single-tool, multi-tool, no-tool, destructive-refusal, and prompt-injection scenarios.
- Trajectory grading so an unsafe intermediate action can fail an evaluation even when the final answer looks correct.
- Reproducible JSON reports committed for inspection.
- Automated tests and GitHub Actions across supported Python versions.

## Engineering decisions

**Multiple graders instead of one score.** Exact match, lexical rules, and an LLM judge fail in different ways. Keeping them separate exposes disagreement rather than hiding it in a single aggregate number.

**Trajectory-aware agent grading.** Agent safety is about actions, not just final prose. EvalForge therefore evaluates tool selection, arguments, unnecessary calls, destructive-action refusal, and injection resistance.

**Ablations are treated as results.** The initial injection-resistance suite produced 100% with and without the defensive system instruction. Rather than presenting that as proof of robust security, the project records the conclusion that the current adversarial dataset is not discriminative enough.

**Auditable reports.** Aggregate evaluation reports are versioned so a reviewer can inspect the measurements behind README claims.

## Current limitations

The datasets are intentionally small and are not presented as industry benchmarks. LLM-judge scores inherit judge-model bias. Agent tools are simulated, so the suite measures decision trajectories without risking real destructive actions. The next meaningful step is a harder adversarial corpus with indirect and context-dependent injection attempts.

## What this project demonstrates

Python engineering, evaluation design, LLM/RAG/agent testing, safety-oriented reasoning, metrics, experiment interpretation, automated testing, CI, and the ability to distinguish a promising result from an insufficient benchmark.
