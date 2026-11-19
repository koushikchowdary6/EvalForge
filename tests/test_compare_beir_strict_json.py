"""Comparison input parsing must not silently rewrite provenance or scores."""
import json

import pytest

from evalforge.compare_beir import main
from tests.test_compare_beir_all_cutoffs import _report


def test_duplicate_provenance_key_is_rejected(tmp_path, capsys):
    report = _report(tmp_path)
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(report), encoding="utf-8")
    raw = json.dumps(report)
    # A repeated nested key could otherwise silently override the trusted digest.
    digest = report["source"]["source_sha256"]["corpus"]
    raw = raw.replace('"corpus": "' + digest + '"',
                      '"corpus": "' + "0" * 64 + '", "corpus": "' + digest + '"', 1)
    candidate.write_text(raw, encoding="utf-8")
    assert main([str(baseline), str(candidate), "--all-k"]) == 2
    assert "duplicate JSON key: corpus" in capsys.readouterr().err


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_are_rejected(tmp_path, capsys, token):
    report = _report(tmp_path)
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(report), encoding="utf-8")
    candidate.write_text(json.dumps(report).replace('"schema_version": 1',
                                                     '"schema_version": 1, "unexpected": ' + token, 1),
                         encoding="utf-8")
    assert main([str(baseline), str(candidate), "--all-k"]) == 2
    assert "non-finite JSON constant" in capsys.readouterr().err


def test_valid_json_still_compares(tmp_path, capsys):
    report = _report(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(report), encoding="utf-8")
    assert main([str(baseline), str(baseline), "--all-k"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
