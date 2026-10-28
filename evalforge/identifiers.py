"""Fail-closed validation for benchmark case and document identifiers.

IDs are machine-readable joins between corpus, ground truth, and reports.
Invisible Unicode controls and canonically equivalent spellings can make two
visually identical labels refer to different cases or documents. Reject those
labels rather than silently rewriting them and changing benchmark provenance.
"""

import unicodedata


def valid_benchmark_id(value):
    """Accept visible, unpadded NFC identifiers without hidden controls.

    Ordinary accented Unicode letters are supported. This is identifier
    integrity checking, not a complete Unicode confusable-name detector.
    """
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and unicodedata.is_normalized("NFC", value)
        and any(
            not char.isspace()
            and not unicodedata.category(char).startswith(("C", "M"))
            for char in value
        )
        and all(
            not unicodedata.category(char).startswith("C")
            and (not char.isspace() or char == " ")
            for char in value
        )
    )
