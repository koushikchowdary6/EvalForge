"""Crash-safe JSON artifact publishing for evaluation runners.

Write to a unique sibling temporary file, fsync the complete payload, then
atomically replace the destination. On POSIX, sync the containing directory
so the new filename survives a sudden crash on filesystems supporting fsync.
Failed serialization or replacement must leave the previous artifact intact.
"""

import json
import os
import tempfile


def atomic_json_dump(path, value):
    """Publish only complete, standard JSON; clean up temporary files on error."""
    destination = os.fspath(path)
    directory = os.path.dirname(destination) or "."
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # fsync on the temporary file makes its contents durable, but the
        # rename itself is a directory metadata change. Sync the containing
        # directory on POSIX so a power loss does not lose the new entry.
        # This is best-effort filesystem durability, not a cross-platform
        # transaction across multiple report files.
        if os.name == "posix":
            dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
