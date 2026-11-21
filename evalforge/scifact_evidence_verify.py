"""Verify completed SciFact evidence against the runner's input and output hashes.

Integrity is not authenticity: the manifest and reports may all be replaced by
an attacker. Compare fingerprints with a trusted acquisition record.
"""
import argparse
import hashlib
import json
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
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return data


def _check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid SHA-256 digest: {path.name}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked evidence file: {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
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
    for name in REPORTS:
        _check_hash(directory / name, outputs[name])
        _read_object(directory / name)
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
