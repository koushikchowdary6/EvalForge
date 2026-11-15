"""Validate a saved EvalForge offline demo artifact before using it in CI.

Usage:
    evalforge-demo --json --output evaluation.json
    evalforge-validate evaluation.json --min-hit-rate 0.5

This is a lexical-fixture gate, not a production RAG or LLM-safety benchmark.
"""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate(report: dict, threshold: float, expected_version: str | None = None) -> None:
    # Library callers must receive the same gate validation as CLI callers.
    # NaN comparisons are always false and could otherwise bypass the gate.
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Minimum hit rate must be finite and between 0 and 1")
    if type(report) is not dict or report.get("schema_version") != 1 or type(report["schema_version"]) is not int:
        raise ValueError("Unsupported report schema version")
    if report.get("benchmark_type") != "offline_lexical_retrieval_only":
        raise ValueError("Not an offline lexical retrieval report")
    if expected_version is not None and report.get("evalforge_version") != expected_version:
        raise ValueError("Report package version differs from installed EvalForge")
    if not isinstance(report.get("limitations"), str) or not report["limitations"]:
        raise ValueError("Missing benchmark limitations")
    for section, metric in (
        ("bm25", "lexical_evidence_hit_rate_by_k"),
        ("paired_tokenizer", "recall_delta_unicode_minus_ascii_by_k"),
    ):
        part = report.get(section)
        if type(part) is not dict or type(part.get("question_count")) is not int or part["question_count"] <= 0:
            raise ValueError(f"Invalid {section} question count")
        for name in ("corpus_sha256", "questions_sha256"):
            if not isinstance(part.get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", part[name]):
                raise ValueError(f"Invalid {section} {name}")
        scores = part.get(metric)
        if type(scores) is not dict or set(scores) != {"1", "3", "5"}:
            raise ValueError(f"Invalid {section} cutoff set")
        for k, score in scores.items():
            if score is not None and (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not (-1 <= score <= 1 if section == "paired_tokenizer" else 0 <= score <= 1)
            ):
                raise ValueError(f"Invalid {section} metric at k={k}")
    hit = report["bm25"]["lexical_evidence_hit_rate_by_k"]["3"]
    if hit is None or hit < threshold:
        raise ValueError(f"Lexical evidence hit rate at k=3 ({hit}) below {threshold}")


# Saved reports are small; reject unexpectedly large inputs before JSON decoding.
# This bounds memory used by a CI gate on untrusted or accidental artifacts.
MAX_REPORT_BYTES = 10 * 1024 * 1024


def _read_report(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("Evaluation artifact exceeds 10 MiB limit")
    return raw.decode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--check-installed-version", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_hit_rate) or not 0 <= args.min_hit_rate <= 1:
        parser.error("--min-hit-rate must be finite and between 0 and 1")
    try:
        expected = version("evalforge-ai") if args.check_installed_version else None
        report = json.loads(_read_report(args.report),
                            object_pairs_hook=_unique_keys,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {x}")))
        validate(report, args.min_hit_rate, expected)
    except (ValueError, OSError, UnicodeError, PackageNotFoundError) as exc:
        parser.exit(1, f"EvalForge artifact rejected: {exc}\n")
    print("EvalForge offline lexical artifact: VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
