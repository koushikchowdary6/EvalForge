#!/usr/bin/env python3
"""Run a reproducible, local SciFact BM25 tokenizer comparison.

Dataset acquisition and licensing remain the operator's responsibility.
This runner never downloads or redistributes benchmark data.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path):
    """Reject non-object or nonstandard JSON masquerading as benchmark evidence."""
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as source:
        data = json.load(source, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object report")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Extracted BEIR SciFact directory")
    parser.add_argument("--output-dir", type=Path, help="Existing empty output directory (required for full runs, not preflight)")
    parser.add_argument("--evalforge-revision", required=True, help="Installed EvalForge commit or release identifier")
    parser.add_argument("--source-url", required=True, help="Upstream dataset source URL")
    parser.add_argument("--download-date", required=True, help="Date dataset was obtained (YYYY-MM-DD)")
    parser.add_argument("--max-drop", type=float, default=0.02,
                        help="Maximum permitted absolute lexical hit-rate drop at each cutoff (default: 0.02)")
    parser.add_argument("--timeout-seconds", type=float, default=1800,
                        help="Per-command timeout in seconds (default: 1800)")
    parser.add_argument("--preflight-validate-schema", action="store_true",
                        help="With --preflight-only, parse BEIR inputs and check document, query and qrel references (uses memory proportional to dataset size)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate input paths and print source SHA-256 fingerprints without running benchmarks or writing artifacts")
    args = parser.parse_args(argv)
    if args.preflight_validate_schema and not args.preflight_only:
        parser.error("--preflight-validate-schema requires --preflight-only")

    import math
    if not math.isfinite(args.max_drop) or not 0 <= args.max_drop <= 1:
        parser.error("--max-drop must be a finite number in [0, 1]")

    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be a positive finite number")

    files = {
        "corpus": args.dataset / "corpus.jsonl",
        "queries": args.dataset / "queries.jsonl",
        "qrels": args.dataset / "qrels" / "test.tsv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        parser.error("Missing SciFact input files: " + ", ".join(missing))
    if not args.preflight_only:
        if args.output_dir is None or not args.output_dir.is_dir():
            parser.error("--output-dir must be an existing directory for full runs")
        if any(args.output_dir.iterdir()):
            parser.error("Output directory must be empty to avoid overwriting evidence")
    # date.fromisoformat also accepts compact YYYYMMDD on supported Python
    # versions. Require the documented format for stable provenance metadata.
    import re
    from datetime import date
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", args.download_date):
        parser.error("--download-date must use YYYY-MM-DD")
    try:
        date.fromisoformat(args.download_date)
    except ValueError:
        parser.error("--download-date must be a valid calendar date")

    try:
        input_fingerprints = {key: sha256(path) for key, path in files.items()}
    except OSError as exc:
        print(f"Cannot fingerprint SciFact inputs: {exc}", file=sys.stderr)
        return 2

    if args.preflight_only:
        schema_report = None
        if args.preflight_validate_schema:
            from .beir_import import convert
            try:
                _, _, schema_report = convert(files["corpus"], files["queries"], files["qrels"])
            except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
                print(f"SciFact schema preflight rejected: {exc}", file=sys.stderr)
                return 2
            # Detect input mutation between fingerprinting and schema validation.
            try:
                if any(sha256(path) != input_fingerprints[key] for key, path in files.items()):
                    print("SciFact inputs changed during schema preflight", file=sys.stderr)
                    return 2
            except OSError as exc:
                print(f"SciFact input unreadable during schema preflight: {exc}", file=sys.stderr)
                return 2
        print(json.dumps({
            "dataset": "BEIR SciFact",
            "split": "test",
            "source_sha256": input_fingerprints,
            "source_url": args.source_url,
            "download_date": args.download_date,
            "status": "inputs_schema_validated" if schema_report is not None else "inputs_fingerprinted_only",
            **({"schema_validation": schema_report} if schema_report is not None else {}),
        }, sort_keys=True))
        return 0

    manifest = {
        "dataset": "BEIR SciFact",
        "split": "test",
        "source_url": args.source_url,
        "download_date": args.download_date,
        "evalforge_revision": args.evalforge_revision,
        "python_version": platform.python_version(),
        "input_sha256": input_fingerprints,
        "cutoffs": [1, 3, 5, 10],
        "tokenizers": ["ascii", "unicode"],
        "comparison_max_drop": args.max_drop,
        "command_timeout_seconds": args.timeout_seconds,
        "comparison_threshold_note": "Illustrative tolerance, not a validated SciFact acceptance criterion",
    }
    # Persist provenance before running so a failed command is auditable.
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def inputs_unchanged():
        """Do not accept evidence if the benchmark inputs changed mid-run."""
        for key, path in files.items():
            try:
                current = sha256(path)
            except OSError as exc:
                print(f"SciFact {key} input became unreadable: {exc}", file=sys.stderr)
                return False
            if current != manifest["input_sha256"][key]:
                print(f"SciFact {key} input changed during evaluation; evidence is not comparable", file=sys.stderr)
                return False
        return True

    reports = {}
    for tokenizer in ("ascii", "unicode"):
        output = args.output_dir / f"scifact-{tokenizer}.json"
        command = [
            "evalforge-benchmark-beir",
            "--corpus", str(files["corpus"]),
            "--queries", str(files["queries"]),
            "--qrels", str(files["qrels"]),
            "--tokenizer", tokenizer,
            "--k", "1", "3", "5", "10",
            "--output", str(output),
        ]
        try:
            completed = subprocess.run(command, check=False, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"{tokenizer} benchmark exceeded {args.timeout_seconds:g}s; evidence retained", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"Could not launch {tokenizer} benchmark: {exc}", file=sys.stderr)
            return 2
        if not inputs_unchanged():
            return 2
        if completed.returncode:
            print(f"{tokenizer} benchmark failed with exit {completed.returncode}", file=sys.stderr)
            return 2
        if not output.is_file():
            print(f"{tokenizer} benchmark exited successfully but did not write {output}", file=sys.stderr)
            return 2
        try:
            report = load_json_object(output)
            source = report.get("source")
            if not isinstance(source, dict) or source.get("source_sha256") != manifest["input_sha256"]:
                raise ValueError("Benchmark source SHA-256 does not match the run manifest")
            if (report.get("schema_version") != 1
                    or report.get("benchmark_scope") != "offline_lexical_bm25_positive_qrels_only"):
                raise ValueError("Benchmark schema or scope differs from the requested run")
            evaluation = report.get("evaluation")
            if not isinstance(evaluation, dict):
                raise ValueError("Benchmark evaluation must be an object")
            expected_tokenizer = {"ascii": "ascii_alphanumeric",
                                  "unicode": "unicode_alphanumeric"}[tokenizer]
            # Installed BM25 reports identify the tokenizer in parameters.tokenizer.
            # Older subprocess fixtures use evaluation.tokenizer_mode instead.
            parameters = evaluation.get("parameters")
            reported_tokenizer = (
                parameters.get("tokenizer") if isinstance(parameters, dict)
                else evaluation.get("tokenizer_mode")
            )
            if reported_tokenizer not in (tokenizer, expected_tokenizer):
                raise ValueError(f"Benchmark tokenizer differs from requested {tokenizer}")
            by_k = evaluation.get("by_top_k")
            if not isinstance(by_k, dict) or set(by_k) != {"1", "3", "5", "10"}:
                raise ValueError("Benchmark did not report all requested cutoffs")
        except (OSError, ValueError) as exc:
            print(f"{tokenizer} benchmark produced unreadable JSON: {exc}", file=sys.stderr)
            return 2
        reports[tokenizer] = output

    command = [
        "evalforge-compare-beir", str(reports["ascii"]), str(reports["unicode"]),
        "--all-k", "--max-drop", str(args.max_drop),
        "--output", str(args.output_dir / "scifact-comparison.json"),
    ]
    try:
        completed = subprocess.run(command, check=False, timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"Comparison exceeded {args.timeout_seconds:g}s; evidence retained", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Could not launch comparison CLI: {exc}", file=sys.stderr)
        return 2
    if not inputs_unchanged():
        return 2
    if completed.returncode in (0, 1):
        comparison_path = args.output_dir / "scifact-comparison.json"
        if not comparison_path.is_file():
            print("Comparison exited without writing JSON evidence", file=sys.stderr)
            return 2
        try:
            evidence = load_json_object(comparison_path)
            if (evidence.get("schema_version") != 1
                    or evidence.get("comparison_scope") != "same_source_beir_lexical_evidence_all_cutoffs"
                    or evidence.get("max_allowed_drop") != args.max_drop):
                raise ValueError("Comparison schema, scope, or threshold differs from the requested run")
            by_k = evidence.get("by_top_k")
            if not isinstance(by_k, dict) or set(by_k) != {"1", "3", "5", "10"}:
                raise ValueError("Comparison did not report every requested cutoff")
            for cutoff, item in by_k.items():
                if (not isinstance(item, dict)
                        or item.get("source_sha256") != manifest["input_sha256"]
                        or item.get("k") != int(cutoff)
                        or item.get("baseline_tokenizer") not in ("ascii", "ascii_alphanumeric")
                        or item.get("candidate_tokenizer") not in ("unicode", "unicode_alphanumeric")
                        or item.get("max_allowed_drop") != args.max_drop
                        or type(item.get("passed")) is not bool):
                    raise ValueError(f"Comparison evidence inconsistent at k={cutoff}")
            if (type(evidence.get("passed")) is not bool
                    or evidence["passed"] != all(item["passed"] for item in by_k.values())
                    or evidence["passed"] != (completed.returncode == 0)):
                raise ValueError("Comparison exit status disagrees with the reported gate outcome")
        except (OSError, ValueError, TypeError) as exc:
            print(f"Comparison produced invalid JSON evidence: {exc}", file=sys.stderr)
            return 2
    if completed.returncode in (0, 1):
        # Bind the final reports to the input provenance, including valid gate failures.
        # A partial or invalid run keeps its original input-only manifest.
        try:
            manifest["output_sha256"] = {
                name: sha256(args.output_dir / name)
                for name in ("scifact-ascii.json", "scifact-unicode.json", "scifact-comparison.json")
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Cannot finalize SciFact evidence manifest: {exc}", file=sys.stderr)
            return 2

    if completed.returncode == 1:
        print("Comparison quality gate failed; evidence retained (this may be a valid finding).", file=sys.stderr)
    elif completed.returncode:
        print("Comparison could not be completed; inspect retained evidence.", file=sys.stderr)
    return completed.returncode if completed.returncode in (0, 1) else 2


if __name__ == "__main__":
    sys.exit(main())
