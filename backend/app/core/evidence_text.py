"""Composing an answer out of retrieved evidence blocks.

Retrieval deliberately returns the same fact from several sources — a chunk, a
reconstructed OCR line, an extracted field — because each is separate
provenance for the same claim. Concatenating their text produced answers that
stuttered:

    "Landlord: Brackenfield Estates Landlord: Brackenfield Estates
     Landlord: Brackenfield Estates landlord: Brackenfield Estates ..."

The citations are all real and all worth keeping; the repetition is not. Text
that says the same thing twice is deduplicated here, and the citation list is
left untouched, so provenance is unaffected.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

ANSWER_CHARACTER_LIMIT = 1200

_WHITESPACE = re.compile(r"\s+")


def _comparison_key(value: str) -> str:
    """Case- and whitespace-insensitive identity for a block of evidence text."""
    return _WHITESPACE.sub(" ", value).strip().lower()


def compose_evidence_text(
    blocks: Iterable[str],
    *,
    limit: int = ANSWER_CHARACTER_LIMIT,
) -> str:
    """Joins evidence blocks, dropping repeats and blocks already contained.

    Order is preserved: retrieval ranked these, and the highest-ranked block
    should still lead the answer. A block whose text already appears inside a
    kept block adds nothing, so it is dropped too — that is the common case
    when a chunk and the OCR line inside it are both retrieved.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for raw in blocks:
        text = _WHITESPACE.sub(" ", str(raw)).strip()
        if not text:
            continue
        key = _comparison_key(text)
        if key in seen:
            continue
        if any(key in _comparison_key(existing) for existing in kept):
            continue
        # A longer block may subsume shorter ones already kept.
        kept = [existing for existing in kept if _comparison_key(existing) not in key]
        seen.add(key)
        kept.append(text)
    return " ".join(kept)[:limit]
