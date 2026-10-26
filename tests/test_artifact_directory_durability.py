"""Atomic JSON reports must sync both file data and rename metadata."""

import json
import os
import stat

import pytest

from evalforge import artifacts


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX-specific")
def test_atomic_json_publish_syncs_file_before_rename_and_directory_after(tmp_path, monkeypatch):
    events = []
    real_fsync = artifacts.os.fsync
    real_replace = artifacts.os.replace

    def recording_fsync(fd):
        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        return real_fsync(fd)

    def recording_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "fsync", recording_fsync)
    monkeypatch.setattr(artifacts.os, "replace", recording_replace)

    destination = tmp_path / "benchmark.json"
    artifacts.atomic_json_dump(destination, {"cases": [1, 2, 3]})

    assert events == ["file", "replace", "directory"]
    assert json.loads(destination.read_text(encoding="utf-8")) == {"cases": [1, 2, 3]}
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX-specific")
def test_directory_sync_failure_closes_descriptor_and_cleans_temp(tmp_path, monkeypatch):
    real_fsync = artifacts.os.fsync
    real_close = artifacts.os.close
    directory_descriptors = []

    def failing_directory_sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_descriptors.append(fd)
            raise OSError("simulated directory sync failure")
        return real_fsync(fd)

    closed = []

    def recording_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(artifacts.os, "fsync", failing_directory_sync)
    monkeypatch.setattr(artifacts.os, "close", recording_close)

    destination = tmp_path / "benchmark.json"
    with pytest.raises(OSError, match="simulated directory sync failure"):
        artifacts.atomic_json_dump(destination, {"version": 2})

    # A failed metadata sync occurs after replacement: do not claim rollback.
    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 2}
    assert len(directory_descriptors) == 1
    assert directory_descriptors[0] in closed
    assert not list(tmp_path.glob(".*.tmp"))
