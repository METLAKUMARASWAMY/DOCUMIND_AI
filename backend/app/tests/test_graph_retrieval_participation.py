"""D-21: graph retrieval could only reach a hardcoded relationship allowlist.

``GRAPH_QUERY_INTENTS`` was a frozen map whose entries came from one fixture --
including the literal alias ``"type of cigarette"``. Only relationship types
named in it could ever reach Cypher, so the field-derived edges introduced by the
D-05 fix (``HAS_POLICY_NUMBER``, ``HAS_CONSIGNEE_NAME``, ...) were unreachable
through the knowledge graph no matter what the user asked.

Candidates are now derived from the same field vocabulary and naming rule the
graph builder uses to create the edges, so question -> edge resolution works for
schemas nobody enumerated in advance. Injection safety is preserved: candidates
are matched against a strict pattern before reaching Cypher, and user text is
never interpolated into the query.
"""

from __future__ import annotations

import pytest

from app.services.vision.persistence import (
    _SAFE_RELATIONSHIP,
    GRAPH_QUERY_INTENTS,
    graph_relationship_candidates,
)

# ------------------------------------------------------------------ derivation


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what is the sample number", "HAS_SAMPLE_NUMBER"),
        ("what is the purpose", "HAS_PURPOSE"),
        ("what is the batch size", "HAS_BATCH_SIZE"),
    ],
)
def test_previously_hardcoded_intents_still_resolve(question: str, expected: str) -> None:
    assert expected in graph_relationship_candidates(question)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # Fields that exist in the registry but were never in the old frozen
        # intent map, so they were previously unreachable through the graph.
        ("what is the policy number", "HAS_POLICY_NUMBER"),
        ("what is the client number", "HAS_CLIENT_NUMBER"),
        ("what is the facsimile number", "HAS_FACSIMILE_NUMBER"),
    ],
)
def test_unseen_field_questions_reach_the_graph(question: str, expected: str) -> None:
    """The whole point: a field the intent map never listed must be reachable."""

    assert expected in graph_relationship_candidates(question)


def test_unknown_terms_degrade_to_the_broader_relation() -> None:
    """A term no schema defines still resolves to something usable.

    "claim amount" is not a registered field, so no HAS_CLAIM_AMOUNT edge can
    exist. Resolving to the broader HAS_AMOUNT is the correct outcome -- the
    graph is searched with a relation that might actually be present rather than
    one that certainly is not.
    """

    candidates = graph_relationship_candidates("what is the claim amount")

    assert "HAS_AMOUNT" in candidates
    assert "HAS_CLAIM_AMOUNT" not in candidates


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("who requested this sample", "REQUESTED_BY"),
        ("who wrote the specifications", "SPECIFICATIONS_WRITTEN_BY"),
        ("who approved the payment", "APPROVED_BY"),
    ],
)
def test_agent_questions_offer_by_relations(question: str, expected: str) -> None:
    """Irregular verbs make phrase->verb mapping unreliable, so "who ..."
    offers the derivable ``*_BY`` set and Cypher filters to what exists."""

    assert expected in graph_relationship_candidates(question)


def test_stated_intents_still_work() -> None:
    assert "ISSUED_BY" in graph_relationship_candidates("which vendor issued this")
    assert "REFERENCES" in graph_relationship_candidates("what is this document linked to")


def test_fixture_alias_is_gone() -> None:
    """'type of cigarette' was a fixture leak in the shipped intent map."""

    aliases = {alias for values in GRAPH_QUERY_INTENTS.values() for alias in values}
    assert "type of cigarette" not in aliases
    for leaked in ("sample number", "batch size", "lot size"):
        assert leaked not in aliases, f"{leaked!r} should be derived, not hardcoded"


# ------------------------------------------------------------------ safety


def test_no_candidates_for_an_unrelated_question() -> None:
    assert graph_relationship_candidates("tell me a joke") == []
    assert graph_relationship_candidates("") == []
    assert graph_relationship_candidates("   ") == []


@pytest.mark.parametrize(
    "hostile",
    [
        "vendor'] OR 1=1 //",
        "vendor` MATCH (n) DETACH DELETE n //",
        "vendor\nDROP DATABASE neo4j",
        "$(whoami) vendor",
        "vendor{tenant_id: 'other'}",
    ],
)
def test_hostile_input_cannot_shape_the_relationship_filter(hostile: str) -> None:
    """User text must never become a relationship name."""

    for candidate in graph_relationship_candidates(hostile):
        assert _SAFE_RELATIONSHIP.match(candidate), f"unsafe candidate {candidate!r}"


def test_every_candidate_matches_the_safe_pattern() -> None:
    questions = [
        "which vendor issued this",
        "who requested this sample",
        "what is the policy number",
        "what is the total amount and the invoice date and the vendor",
    ]
    for question in questions:
        for candidate in graph_relationship_candidates(question):
            assert _SAFE_RELATIONSHIP.match(candidate), candidate


def test_candidate_count_is_bounded() -> None:
    """A broad question must not widen the Cypher filter without limit."""

    broad = "who " + " ".join(
        ["vendor", "customer", "invoice number", "total amount", "address", "tax", "phone"]
    )
    assert len(graph_relationship_candidates(broad)) <= 40


def test_candidates_are_unique() -> None:
    candidates = graph_relationship_candidates("who approved the vendor invoice payment")
    assert len(candidates) == len(set(candidates))
