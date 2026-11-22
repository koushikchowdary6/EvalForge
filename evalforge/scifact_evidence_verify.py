"""Verify completed SciFact evidence against the runner's input and output hashes.

Integrity is not authenticity: the manifest and reports may all be replaced by
an attacker. Compare fingerprints with a trusted acquisition record.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

REPORTS = ("scifact-ascii.json", "scifact-unicode.json", "scifact-comparison.json")
INPUTS = {"corpus": "corpus.jsonl", "queries": "queries.jsonl", "qrels": "qrels/test.tsv"}


def _read_object(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked evidence file: {path.name}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    data = json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=unique, parse_constant=reject_constant)
    def reject_nonfinite(value):
        if isinstance(value, float) and not __import__("math").isfinite(value):
            raise ValueError(f"Non-finite JSON number: {path.name}")
        if isinstance(value, dict):
            for item in value.values():
                reject_nonfinite(item)
        elif isinstance(value, list):
            for item in value:
                reject_nonfinite(item)

    reject_nonfinite(data)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return data


def _check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid SHA-256 digest: {path.name}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked evidence file: {path.name}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def verify(directory, dataset=None):
    directory = Path(directory)
    manifest = _read_object(directory / "manifest.json")
    if manifest.get("dataset") != "BEIR SciFact" or manifest.get("split") != "test":
        raise ValueError("Not a SciFact test-split manifest")
    outputs = manifest.get("output_sha256")
    inputs = manifest.get("input_sha256")
    if not isinstance(outputs, dict) or set(outputs) != set(REPORTS):
        raise ValueError("Manifest lacks finalized SciFact output hashes")
    if not isinstance(inputs, dict) or set(inputs) != set(INPUTS):
        raise ValueError("Manifest lacks SciFact input hashes")
    reports = {}
    for name in REPORTS:
        _check_hash(directory / name, outputs[name])
        reports[name] = _read_object(directory / name)

    # Hashes alone cannot establish that the files are SciFact reports.
    # Check the provenance and metric schema before declaring evidence valid.
    for tokenizer in ("ascii", "unicode"):
        report = reports[f"scifact-{tokenizer}.json"]
        source = report.get("source")
        evaluation = report.get("evaluation")
        if (type(report.get("schema_version")) is not int
                or report["schema_version"] != 1
                or report.get("benchmark_scope") != "offline_lexical_bm25_positive_qrels_only"
                or not isinstance(source, dict)
                or source.get("source_sha256") != inputs
                or not isinstance(evaluation, dict)):
            raise ValueError(f"Invalid SciFact {tokenizer} report schema or provenance")
        parameters = evaluation.get("parameters")
        mode = (parameters.get("tokenizer") if isinstance(parameters, dict)
                else evaluation.get("tokenizer_mode"))
        expected = {"ascii": "ascii_alphanumeric", "unicode": "unicode_alphanumeric"}[tokenizer]
        if mode not in (tokenizer, expected):
            raise ValueError(f"Invalid SciFact {tokenizer} tokenizer")
        if not isinstance(evaluation.get("by_top_k"), dict) or set(evaluation["by_top_k"]) != {"1", "3", "5", "10"}:
            raise ValueError(f"Missing SciFact {tokenizer} cutoffs")

    comparison = reports["scifact-comparison.json"]
    by_k = comparison.get("by_top_k")
    if (type(comparison.get("schema_version")) is not int
            or comparison["schema_version"] != 1
            or comparison.get("comparison_scope") != "same_source_beir_lexical_evidence_all_cutoffs"
            or not isinstance(by_k, dict)
            or set(by_k) != {"1", "3", "5", "10"}
            or type(comparison.get("passed")) is not bool):
        raise ValueError("Invalid SciFact comparison schema")
    for cutoff, item in by_k.items():
        if (not isinstance(item, dict)
                or item.get("source_sha256") != inputs
                or type(item.get("k")) is not int or item["k"] != int(cutoff)
                or type(item.get("passed")) is not bool):
            raise ValueError(f"Invalid SciFact comparison evidence at k={cutoff}")
    # Bind comparison scores to the actual hashed benchmark reports. Otherwise a
    # self-consistent comparison JSON can invent scores without changing either run.
    for cutoff in ("1", "3", "5", "10"):
        item = by_k[cutoff]
        for tokenizer, field in (("ascii", "baseline_hit_lower_bound"),
                                 ("unicode", "candidate_hit_lower_bound")):
            report = reports[f"scifact-{tokenizer}.json"]
            summary = report["evaluation"]["by_top_k"][cutoff]
            if not isinstance(summary, dict):
                raise ValueError(f"Invalid SciFact {tokenizer} cutoff summary at k={cutoff}")
            score = summary.get("lexical_evidence_hit_lower_bound")
            if (type(score) not in (int, float) or not math.isfinite(score)
                    or not 0 <= score <= 1):
                raise ValueError(f"Invalid SciFact {tokenizer} retrieval score at k={cutoff}")
            if (type(item.get(field)) not in (int, float)
                    or not math.isclose(item[field], score, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"SciFact comparison score disagrees with {tokenizer} report at k={cutoff}")

    # Recompute the quality gate; a matching hash does not prove arithmetic integrity.
    threshold = comparison.get("max_allowed_drop")
    # A valid comparison must use the tolerance recorded before the run.
    # Otherwise edited reports can silently relax the declared regression gate.
    if type(manifest.get("comparison_max_drop")) not in (int, float) or not math.isfinite(manifest["comparison_max_drop"]):
        raise ValueError("Invalid SciFact manifest comparison threshold")
    if threshold != manifest["comparison_max_drop"]:
        raise ValueError("SciFact comparison threshold differs from run manifest")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid SciFact comparison drop threshold")
    for cutoff, item in by_k.items():
        baseline = item.get("baseline_hit_lower_bound")
        candidate = item.get("candidate_hit_lower_bound")
        delta = item.get("delta")
        allowed = item.get("max_allowed_drop")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (baseline, candidate, delta, allowed)):
            raise ValueError(f"Invalid SciFact comparison scores at k={cutoff}")
        if not (0 <= baseline <= 1 and 0 <= candidate <= 1 and allowed == threshold):
            raise ValueError(f"Invalid SciFact comparison bounds at k={cutoff}")
        if not math.isclose(delta, candidate - baseline, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Inconsistent SciFact comparison delta at k={cutoff}")
        if item["passed"] != (baseline - candidate <= allowed + 1e-12):
            raise ValueError(f"Inconsistent SciFact comparison gate at k={cutoff}")
    if comparison["passed"] != all(item["passed"] for item in by_k.values()):
        raise ValueError("SciFact comparison outcome disagrees with cutoff results")
    for key, value in inputs.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid input SHA-256 digest: {key}")
    if dataset is not None:
        dataset = Path(dataset)
        for key, name in INPUTS.items():
            _check_hash(dataset / name, inputs[key])
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Completed SciFact evidence directory")
    parser.add_argument("--dataset", type=Path, help="Optional original extracted SciFact dataset to fingerprint")
    args = parser.parse_args(argv)
    try:
        verify(args.directory, args.dataset)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2, f"SciFact evidence verification failed: {exc}\n")
    print("SciFact report hashes verified" + (" against dataset inputs" if args.dataset else "") + ".")
    print("Integrity only; this does not authenticate the source or validate retrieval quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
