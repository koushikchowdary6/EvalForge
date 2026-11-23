"""Regression coverage for linked evidence paths.

The verifier must not follow symlinked parent directories even if the
manifest or input file itself is a regular file.
"""
import hashlib
import json

import pytest

from evalforge.scifact_evidence_verify import _check_hash, _read_object


@pytest.mark.parametrize("reader", ["json", "hash"])
def test_symlinked_parent_is_rejected(tmp_path, reader):
    actual = tmp_path / "actual"
    actual.mkdir()
    target = actual / "manifest.json"
    payload = b'{"dataset": "BEIR SciFact"}'
    target.write_bytes(payload)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    path = linked / "manifest.json"

    with pytest.raises(ValueError, match="symlinked"):
        if reader == "json":
            _read_object(path)
        else:
            _check_hash(path, hashlib.sha256(payload).hexdigest())


def test_regular_parent_is_accepted(tmp_path):
    directory = tmp_path / "actual"
    directory.mkdir()
    target = directory / "manifest.json"
    payload = {"dataset": "BEIR SciFact"}
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _read_object(target) == payload
    _check_hash(target, hashlib.sha256(target.read_bytes()).hexdigest())
