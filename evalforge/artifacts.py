"""Crash-safe JSON artifact publishing for evaluation runners.

Write to a unique sibling temporary file, fsync the complete payload, then
atomically replace the destination. Failed serialization or publication must
leave the previous benchmark artifact intact.
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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
