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
                "max_allowed_drop": 0.02,
                "by_top_k": {
                    str(k): {"k": k, "passed": True, "source_sha256": input_hashes,
                             "baseline_hit_lower_bound": 0.7,
                             "candidate_hit_lower_bound": 0.7,
                             "delta": 0.0, "max_allowed_drop": 0.02}
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
                    "by_top_k": {str(k): {"lexical_evidence_hit_lower_bound": 0.7} for k in (1, 3, 5, 10)},
                },
            }
        data = json.dumps(payload).encode()
        (evidence / name).write_bytes(data)
        output_hashes[name] = _hash(data)
    (evidence / "manifest.json").write_text(json.dumps({
        "dataset": "BEIR SciFact", "split": "test",
        "input_sha256": input_hashes, "output_sha256": output_hashes,
        "comparison_max_drop": 0.02,
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
    with pytest.raises(ValueError, match="comparison"):
        verify(evidence)


@pytest.mark.parametrize("number", ["1e999", "-1e999"])
def test_hash_matching_overflowed_json_number_rejected(tmp_path, number):
    evidence, _ = bundle(tmp_path)
    name = "scifact-ascii.json"
    report = (evidence / name).read_text(encoding="utf-8")
    payload = report[:-1] + ', "overflowed_metric": ' + number + '}'
    (evidence / name).write_text(payload, encoding="utf-8")
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"][name] = _hash(payload.encode("utf-8"))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite JSON number"):
        verify(evidence)


@pytest.mark.parametrize("field,value", [
    ("delta", 0.3),
    ("candidate_hit_lower_bound", 0.1),
    ("max_allowed_drop", 0.5),
])
def test_hash_matching_fabricated_comparison_rejected(tmp_path, field, value):
    evidence, _ = bundle(tmp_path)
    name = "scifact-comparison.json"
    report = json.loads((evidence / name).read_text(encoding="utf-8"))
    report["by_top_k"]["3"][field] = value
    payload = json.dumps(report).encode()
    (evidence / name).write_bytes(payload)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"][name] = _hash(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SciFact comparison"):
        verify(evidence)


@pytest.mark.parametrize("tokenizer,field", [
    ("ascii", "baseline_hit_lower_bound"),
    ("unicode", "candidate_hit_lower_bound"),
])
def test_hash_matching_fabricated_retrieval_score_rejected(tmp_path, tokenizer, field):
    evidence, _ = bundle(tmp_path)
    name = f"scifact-{tokenizer}.json"
    report = json.loads((evidence / name).read_text(encoding="utf-8"))
    report["evaluation"]["by_top_k"]["5"]["lexical_evidence_hit_lower_bound"] = 0.2
    payload = json.dumps(report).encode()
    (evidence / name).write_bytes(payload)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"][name] = _hash(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison score disagrees"):
        verify(evidence)


def test_hash_matching_missing_retrieval_metric_rejected(tmp_path):
    evidence, _ = bundle(tmp_path)
    name = "scifact-ascii.json"
    report = json.loads((evidence / name).read_text(encoding="utf-8"))
    del report["evaluation"]["by_top_k"]["3"]["lexical_evidence_hit_lower_bound"]
    payload = json.dumps(report).encode()
    (evidence / name).write_bytes(payload)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"][name] = _hash(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="retrieval score"):
        verify(evidence)


@pytest.mark.parametrize("invalid", [None, -0.01, 1.01, "0.02", float("inf")])
def test_manifest_comparison_threshold_invalid_rejected(tmp_path, invalid):
    evidence, _ = bundle(tmp_path)
    path = evidence / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if invalid is None:
        del manifest["comparison_max_drop"]
    else:
        manifest["comparison_max_drop"] = invalid
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest comparison threshold|comparison threshold differs|Non-finite JSON constant"):
        verify(evidence)


def test_matching_hashes_cannot_relax_manifest_comparison_threshold(tmp_path):
    evidence, _ = bundle(tmp_path)
    name = "scifact-comparison.json"
    path = evidence / name
    report = json.loads(path.read_text(encoding="utf-8"))
    report["max_allowed_drop"] = 0.5
    for row in report["by_top_k"].values():
        row["max_allowed_drop"] = 0.5
    payload = json.dumps(report).encode("utf-8")
    path.write_bytes(payload)
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"][name] = _hash(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold differs"):
        verify(evidence)


def test_verifier_streams_hashes_without_loading_entire_files(tmp_path, monkeypatch):
    """Large BEIR corpora should not be loaded into memory just for hashing."""
    from pathlib import Path

    evidence, dataset = bundle(tmp_path)

    def forbid_read_bytes(self):
        raise AssertionError(f"Unbounded read_bytes while hashing: {self}")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    assert verify(evidence, dataset)["dataset"] == "BEIR SciFact"


def test_oversized_evidence_json_rejected_before_parsing(tmp_path):
    from evalforge.scifact_evidence_verify import _read_object

    oversized = tmp_path / "manifest.json"
    # Sparse allocation avoids keeping a 64 MiB fixture in test memory.
    with oversized.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="Oversized SciFact evidence JSON"):
        _read_object(oversized)


def test_deeply_nested_untrusted_json_exits_cleanly(tmp_path, capsys):
    """Malformed evidence must not leak a Python recursion traceback."""
    evidence, _ = bundle(tmp_path)
    manifest = evidence / "manifest.json"
    manifest.write_text('{"nested":' + "[" * 1500 + "0" + "]" * 1500 + "}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main([str(evidence)])
    assert error.value.code == 2
    assert "Excessively nested SciFact evidence JSON" in capsys.readouterr().err
