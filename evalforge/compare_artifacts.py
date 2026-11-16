"""Compare two validated offline EvalForge reports on identical fixtures.

Only deterministic lexical retrieval metrics are compared; no LLM safety claims.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from .artifact_validation import _read_report, _unique_keys, validate


def compare(baseline: dict, candidate: dict, tolerance: float = 0.0) -> dict:
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0 or tolerance > 1:
        raise ValueError("Tolerance must be finite and between 0 and 1")
    validate(baseline, 0.0)
    validate(candidate, 0.0)
    for section in ("bm25", "paired_tokenizer"):
        for field in ("question_count", "corpus_sha256", "questions_sha256"):
            if baseline[section][field] != candidate[section][field]:
                raise ValueError(f"Incomparable {section} {field}: fixtures differ")
    metrics = {}
    absolute_scores = {}
    for section, field in (
        ("bm25", "lexical_evidence_hit_rate_by_k"),
        ("paired_tokenizer", "recall_delta_unicode_minus_ascii_by_k"),
    ):
        metrics[section] = {}
        absolute_scores[section] = {"baseline": {}, "candidate": {}}
        for cutoff in ("1", "3", "5"):
            old = baseline[section][field][cutoff]
            new = candidate[section][field][cutoff]
            if old is None or new is None:
                if old != new:
                    raise ValueError(f"Incomparable undefined {section} metric at k={cutoff}")
                delta = None
            else:
                delta = new - old
            metrics[section][cutoff] = delta
            absolute_scores[section]["baseline"][cutoff] = old
            absolute_scores[section]["candidate"][cutoff] = new
    # Paired Unicode-minus-ASCII delta is diagnostic, not a quality score:
    # its sign cannot be interpreted as an improvement or regression.
    regressions = [
        k for k, delta in metrics["bm25"].items()
        if delta is not None and delta < -tolerance
    ]
    # Include validated provenance so a CI decision is auditable after renaming files.
    provenance = {
        "baseline_version": baseline.get("evalforge_version"),
        "candidate_version": candidate.get("evalforge_version"),
        "datasets": {
            section: {
                field: baseline[section][field]
                for field in ("question_count", "corpus_sha256", "questions_sha256")
            }
            for section in ("bm25", "paired_tokenizer")
        },
    }
    return {"bm25_regression_cutoffs": regressions, "metric_deltas": metrics,
            "absolute_scores": absolute_scores, "tolerance": tolerance,
            "provenance": provenance}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="Write JSON report to a new file (never overwrite)")
    args = parser.parse_args(argv)
    try:
        def load(path):
            return json.loads(
                _read_report(path), object_pairs_hook=_unique_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Invalid JSON constant: {value}")
                ),
            )
        result = compare(load(args.baseline), load(args.candidate), args.tolerance)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, f"EvalForge comparison rejected: {exc}\n")
    payload = json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        try:
            # Stage and fsync the full report before publishing it. An exclusive
            # hard link prevents overwrites and avoids exposing partial CI evidence.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                prefix=".evalforge-compare-", suffix=".tmp",
                dir=args.output.parent, delete=False,
            ) as staged:
                staged_path = Path(staged.name)
                try:
                    staged.write(payload)
                    staged.flush()
                    os.fsync(staged.fileno())
                except BaseException:
                    staged_path.unlink(missing_ok=True)
                    raise
            try:
                os.link(staged_path, args.output)
            finally:
                staged_path.unlink(missing_ok=True)
        except OSError as exc:
            parser.exit(2, f"EvalForge comparison output rejected: {exc}\n")
    else:
        print(payload, end="")
    return 1 if result["bm25_regression_cutoffs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
