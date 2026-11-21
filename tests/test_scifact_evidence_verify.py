"""Regression tests for the independent SciFact evidence verifier."""
import hashlib
import json

import pytest

from evalforge.scifact_evidence_verify import verify, main, REPORTS, INPUTS


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def bundle(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    dataset = tmp_path / "dataset"
    (dataset / "qrels").mkdir(parents=True)
    input_hashes = {}
    for key, name in INPUTS.items():
        data = (key + "\n").encode()
        (dataset / name).write_bytes(data)
        input_hashes[key] = _hash(data)
    output_hashes = {}
    for name in REPORTS:
        if name == "scifact-comparison.json":
            payload = {
                "schema_version": 1,
                "comparison_scope": "same_source_beir_lexical_evidence_all_cutoffs",
                "passed": True,
                "by_top_k": {
                    str(k): {"k": k, "passed": True, "source_sha256": input_hashes}
                    for k in (1, 3, 5, 10)
                },
            }
        else:
            tokenizer = name.removeprefix("scifact-").removesuffix(".json")
            payload = {
                "schema_version": 1,
                "benchmark_scope": "offline_lexical_bm25_positive_qrels_only",
                "source": {"source_sha256": input_hashes},
                "evaluation": {
                    "parameters": {"tokenizer": tokenizer},
                    "by_top_k": {str(k): {} for k in (1, 3, 5, 10)},
                },
            }
        data = json.dumps(payload).encode()
        (evidence / name).write_bytes(data)
        output_hashes[name] = _hash(data)
    (evidence / "manifest.json").write_text(json.dumps({
        "dataset": "BEIR SciFact", "split": "test",
        "input_sha256": input_hashes, "output_sha256": output_hashes,
    }), encoding="utf-8")
    return evidence, dataset


def test_valid_bundle_and_original_inputs(tmp_path):
    evidence, dataset = bundle(tmp_path)
    assert verify(evidence, dataset)["dataset"] == "BEIR SciFact"
    assert main([str(evidence), "--dataset", str(dataset)]) == 0


def test_modified_report_rejected(tmp_path):
    evidence, _ = bundle(tmp_path)
    (evidence / REPORTS[0]).write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(evidence)


def test_modified_source_rejected(tmp_path):
    evidence, dataset = bundle(tmp_path)
    (dataset / INPUTS["qrels"]).write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(evidence, dataset)


def test_partial_manifest_rejected(tmp_path):
    evidence, _ = bundle(tmp_path)
    manifest = json.loads((evidence / "manifest.json").read_text())
    del manifest["output_sha256"]
    (evidence / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finalized"):
        verify(evidence)


def test_symlink_report_rejected_even_with_matching_hash(tmp_path):
    evidence, _ = bundle(tmp_path)
    report = evidence / REPORTS[0]
    external = tmp_path / "external.json"
    external.write_bytes(report.read_bytes())
    report.unlink()
    report.symlink_to(external)
    with pytest.raises(ValueError, match="symlinked"):
        verify(evidence)


def test_duplicate_json_keys_rejected(tmp_path):
    evidence, _ = bundle(tmp_path)
    report = evidence / REPORTS[0]
    payload = b'{"a":1,"a":2}'
    report.write_bytes(payload)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"][REPORTS[0]] = _hash(payload)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        verify(evidence)


@pytest.mark.parametrize("name", REPORTS)
def test_hash_matching_nonbenchmark_json_rejected(tmp_path, name):
    evidence, _ = bundle(tmp_path)
    payload = b'{"unrelated":"valid JSON but not benchmark evidence"}'
    (evidence / name).write_bytes(payload)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"][name] = _hash(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid SciFact"):
        verify(evidence)


def test_hash_matching_comparison_outcome_inconsistency_rejected(tmp_path):
    evidence, _ = bundle(tmp_path)
    name = "scifact-comparison.json"
    report = json.loads((evidence / name).read_text(encoding="utf-8"))
    report["by_top_k"]["5"]["passed"] = False
    payload = json.dumps(report).encode()
    (evidence / name).write_bytes(payload)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"][name] = _hash(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outcome disagrees"):
        verify(evidence)
