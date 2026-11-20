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
    parser.add_argument("--output-dir", type=Path, required=True, help="Existing empty output directory")
    parser.add_argument("--evalforge-revision", required=True, help="Installed EvalForge commit or release identifier")
    parser.add_argument("--source-url", required=True, help="Upstream dataset source URL")
    parser.add_argument("--download-date", required=True, help="Date dataset was obtained (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    files = {
        "corpus": args.dataset / "corpus.jsonl",
        "queries": args.dataset / "queries.jsonl",
        "qrels": args.dataset / "qrels" / "test.tsv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        parser.error("Missing SciFact input files: " + ", ".join(missing))
    if not args.output_dir.is_dir():
        parser.error("Output directory must already exist")
    if any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty to avoid overwriting evidence")
    try:
        from datetime import date
        date.fromisoformat(args.download_date)
    except ValueError:
        parser.error("--download-date must be a valid ISO date")

    manifest = {
        "dataset": "BEIR SciFact",
        "split": "test",
        "source_url": args.source_url,
        "download_date": args.download_date,
        "evalforge_revision": args.evalforge_revision,
        "python_version": platform.python_version(),
        "input_sha256": {key: sha256(path) for key, path in files.items()},
        "cutoffs": [1, 3, 5, 10],
        "tokenizers": ["ascii", "unicode"],
        "comparison_max_drop": 0.02,
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
            completed = subprocess.run(command, check=False)
        except FileNotFoundError:
            print("Install EvalForge first: benchmark CLI was not found", file=sys.stderr)
            return 2
        if not inputs_unchanged():
            return 2
        if completed.returncode:
            print(f"{tokenizer} benchmark failed with exit {completed.returncode}", file=sys.stderr)
            return completed.returncode
        if not output.is_file():
            print(f"{tokenizer} benchmark exited successfully but did not write {output}", file=sys.stderr)
            return 2
        try:
            load_json_object(output)
        except (OSError, ValueError) as exc:
            print(f"{tokenizer} benchmark produced unreadable JSON: {exc}", file=sys.stderr)
            return 2
        reports[tokenizer] = output

    command = [
        "evalforge-compare-beir", str(reports["ascii"]), str(reports["unicode"]),
        "--all-k", "--max-drop", "0.02",
        "--output", str(args.output_dir / "scifact-comparison.json"),
    ]
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError:
        print("Install EvalForge first: comparison CLI was not found", file=sys.stderr)
        return 2
    if not inputs_unchanged():
        return 2
    if completed.returncode in (0, 1):
        comparison_path = args.output_dir / "scifact-comparison.json"
        if not comparison_path.is_file():
            print("Comparison exited without writing JSON evidence", file=sys.stderr)
            return 2
        try:
            load_json_object(comparison_path)
        except (OSError, ValueError) as exc:
            print(f"Comparison produced unreadable JSON evidence: {exc}", file=sys.stderr)
            return 2
    if completed.returncode == 1:
        print("Comparison quality gate failed; evidence retained (this may be a valid finding).", file=sys.stderr)
    elif completed.returncode:
        print("Comparison could not be completed; inspect retained evidence.", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
