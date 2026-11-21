"""Verify the integrity of a downloaded offline evaluation evidence bundle.

This checks artifact integrity, not the truth of the benchmark claims or the
identity of the workflow that produced the files.
"""
import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = ("baseline.json", "candidate.json", "comparison.json")


def verify_bundle(directory: Path) -> dict:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Evidence manifest is missing or a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported evidence manifest schema")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED):
        raise ValueError("Manifest must hash exactly the three evidence JSON files")
    if not isinstance(manifest.get("git_revision"), str) or len(manifest["git_revision"]) != 40:
        raise ValueError("Missing 40-character Git revision")
    for name in REQUIRED:
        expected = hashes[name]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid SHA-256 digest for {name}")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Evidence file is missing or a symlink: {name}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        json.loads(data)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing manifest.json and three evidence reports")
    args = parser.parse_args(argv)
    try:
        manifest = verify_bundle(args.directory)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Evidence verification failed: {exc}\n")
    print(f"Verified three JSON artifacts for Git revision {manifest['git_revision']}")
    print("Integrity only: this does not authenticate the producer or validate benchmark quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
