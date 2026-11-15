"""Resource-boundary regression tests for the installed artifact validator."""

import pytest

from evalforge.artifact_validation import MAX_REPORT_BYTES, _read_report, main


def test_report_reader_accepts_exact_byte_limit(tmp_path):
    path = tmp_path / "exact.json"
    path.write_bytes(b" " * MAX_REPORT_BYTES)
    assert len(_read_report(path)) == MAX_REPORT_BYTES


def test_report_reader_rejects_oversized_input(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (MAX_REPORT_BYTES + 1))
    with pytest.raises(ValueError, match="10 MiB"):
        _read_report(path)


def test_cli_rejects_oversized_report_without_success(tmp_path, capsys):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (MAX_REPORT_BYTES + 1))
    with pytest.raises(SystemExit) as exit_info:
        main([str(path)])
    assert exit_info.value.code == 1
    assert "exceeds 10 MiB" in capsys.readouterr().err
