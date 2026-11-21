import hashlib
import json

import pytest

from evalforge.evidence_verify import verify_bundle


@pytest.fixture
def bundle(tmp_path):
    hashes = {}
    for name in ("baseline.json", "candidate.json", "comparison.json"):
        data = b'{"ok": true}\n'
        (tmp_path / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "git_revision": "a" * 40,
        "sha256": hashes,
    }))
    return tmp_path


def test_valid_bundle(bundle):
    assert verify_bundle(bundle)["schema_version"] == 1


def test_tampered_report_rejected(bundle):
    (bundle / "candidate.json").write_text('{"ok": false}\n')
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_bundle(bundle)


def test_missing_report_rejected(bundle):
    (bundle / "comparison.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_bundle(bundle)


def test_extra_manifest_entry_rejected(bundle):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sha256"]["../outside.json"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly"):
        verify_bundle(bundle)


def test_corrupt_json_rejected_even_with_matching_hash(bundle):
    data = b"not-json"
    (bundle / "baseline.json").write_bytes(data)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sha256"]["baseline.json"] = hashlib.sha256(data).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(json.JSONDecodeError):
        verify_bundle(bundle)
