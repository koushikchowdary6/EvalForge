"""Compare two validated offline EvalForge reports on identical fixtures.

Only deterministic lexical retrieval metrics are compared; no LLM safety claims.
"""
from __future__ import annotations

import argparse
import json
import math
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
    for section, field in (
        ("bm25", "lexical_evidence_hit_rate_by_k"),
        ("paired_tokenizer", "recall_delta_unicode_minus_ascii_by_k"),
    ):
        metrics[section] = {}
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
    # Paired Unicode-minus-ASCII delta is diagnostic, not a quality score:
    # its sign cannot be interpreted as an improvement or regression.
    regressions = [
        k for k, delta in metrics["bm25"].items()
        if delta is not None and delta < -tolerance
    ]
    return {"bm25_regression_cutoffs": regressions, "metric_deltas": metrics}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.0)
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
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result["bm25_regression_cutoffs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
