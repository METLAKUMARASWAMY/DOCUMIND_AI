"""Whether a candidate value can be the party a "who" question asks for.

A question like "who owns the property being let" is asking for an agent. The
retrieval layers score candidates lexically, so a question sharing one generic
token with an unrelated field ("property" against "property_address") can
outrank the field that actually answers it -- and the answer is then presented
as supported. Answering "who ..." with a street address is a wrong answer, not
a weak one, so such candidates are dropped and the caller abstains.

The decision uses the document's own contextual semantic typing rather than
re-classifying the bare string: classification at extraction time saw the
surrounding line, and is markedly more accurate than classifying a value alone.
Values the document left untyped stay eligible, which matters because personal
and organisation names are frequently typed only as TEXT.
"""

from __future__ import annotations

import re
from typing import Any

#: Semantic types that cannot denote a person or organisation. PERSON,
#: ORGANISATION and TEXT are deliberately absent: the first two are agents, and
#: TEXT is the "not confidently typed" bucket a real name usually lands in.
NON_AGENT_SEMANTIC_TYPES: frozenset[str] = frozenset(
    {
        "ADDRESS",
        "BANK_ACCOUNT",
        "BOOLEAN",
        "CURRENCY",
        "DATE",
        "DATETIME",
        "DOCUMENT_REFERENCE",
        "EMAIL",
        "IDENTIFIER",
        "MONEY",
        "NUMBER",
        "PERCENTAGE",
        "PHONE",
        "PRODUCT",
        "QUANTITY",
        "URL",
    }
)

_AGENT_QUESTION = re.compile(r"(?<![\w-])(?:who|whom|whose)(?![\w-])", re.IGNORECASE)


def asks_for_an_agent(question: str) -> bool:
    """True when the question is asking which party did or owns something."""
    return bool(_AGENT_QUESTION.search(question or ""))


def semantic_types_by_value(document: Any) -> dict[str, str]:
    """Map each typed value in a document to the semantic type it was given."""
    mapping: dict[str, str] = {}
    for entity in getattr(document, "typed_entities", None) or []:
        semantic = getattr(entity, "semantic_type", None)
        if semantic is None:
            continue
        name = str(getattr(semantic, "value", semantic)).upper()
        for attribute in ("normalized_value", "raw_value"):
            value = getattr(entity, attribute, None)
            if value:
                mapping[str(value).strip().lower()] = name
    return mapping


def value_cannot_be_an_agent(value: object, semantic_types: dict[str, str]) -> bool:
    """True when this document typed ``value`` as something no agent can be."""
    text = str(value or "").strip().lower()
    if not text:
        return False
    return semantic_types.get(text) in NON_AGENT_SEMANTIC_TYPES
