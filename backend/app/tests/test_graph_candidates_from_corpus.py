"""Graph queries must reach edges the registry has never heard of.

D-21 replaced a hardcoded relationship allowlist with names derived from the
registered field vocabulary. That works while the documents look like the
registry — and the acceptance fixture does: its ``requested_by`` field yields
``REQUESTED_BY``, which the registry can also name, so the fixture never
exposed the limit.

Documents from a domain the registry has never seen do expose it. Measured,
registry-only derivation against the edges those documents actually hold:

    "which vessel carried the shipment"  -> reaches NONE of the corpus edges
    "what is the haemoglobin level"      -> reaches NONE of the corpus edges
    "who is the reporting pathologist"   -> reaches NONE of the corpus edges

The corpus now decides what exists and the question decides what is relevant,
so ``HAS_VESSEL_NAME`` and ``HAS_HAEMOGLOBIN`` become reachable without anyone
having enumerated them. Candidates are still filtered before they can reach
Cypher: edge types are written from document text, so they are as untrusted as
the text.

Note on how this was found: the first diagnosis here was wrong. A truncated
printout of the candidate list appeared to show ``REQUESTED_BY`` missing when
it was present at index 9, and the control test below is what caught it.
"""

from __future__ import annotations

import pytest

from app.services.vision.persistence import graph_relationship_candidates

# Exactly the edge types the acceptance fixture produces.
FIXTURE_EDGES = [
    "CONTAINS_PAGE",
    "HAS_FIELD",
    "MATCHES",
    "HAS_DATE",
    "HAS_BATCH_SIZE",
    "HAS_PRODUCT_TYPE",
    "HAS_REQUEST_DATE",
    "REQUESTED_BY",
    "HAS_SAMPLE_NUMBER",
    "SPECIFICATIONS_WRITTEN_BY",
]


#: Edges from domains the registered schemas do not cover.
UNSEEN_DOMAIN_EDGES = [
    "HAS_VESSEL_NAME",
    "HAS_PORT_OF_DISCHARGE",
    "HAS_CONTAINER_NUMBER",
    "HAS_GROSS_WEIGHT",
    "HAS_HAEMOGLOBIN",
    "HAS_PLATELET_COUNT",
    "HAS_SPECIMEN_TYPE",
    "REPORTED_BY",
]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("which vessel carried the shipment", "HAS_VESSEL_NAME"),
        ("what is the haemoglobin level", "HAS_HAEMOGLOBIN"),
        ("what is the gross weight", "HAS_GROSS_WEIGHT"),
        ("what is the port of discharge", "HAS_PORT_OF_DISCHARGE"),
    ],
)
def test_a_question_reaches_an_edge_from_a_domain_the_registry_never_saw(question, expected):
    candidates = graph_relationship_candidates(question, present_relationships=UNSEEN_DOMAIN_EDGES)
    assert expected in candidates, candidates


@pytest.mark.parametrize(
    "question",
    [
        "which vessel carried the shipment",
        "what is the haemoglobin level",
        "what is the gross weight",
    ],
)
def test_without_the_corpus_none_of_those_edges_are_reachable(question):
    """Pins *why* the parameter is needed rather than asserting it vacuously.

    If the registry ever grows these terms this test starts failing, which is
    the correct signal: the one above would then be proving nothing.
    """
    candidates = graph_relationship_candidates(question)
    assert not set(candidates) & set(UNSEEN_DOMAIN_EDGES), candidates


def test_the_fixture_edge_stays_reachable_with_the_corpus_supplied():
    """The registry could already name this one; it must not regress."""
    candidates = graph_relationship_candidates(
        "Who made the original request?", present_relationships=FIXTURE_EDGES
    )
    assert "REQUESTED_BY" in candidates, candidates


def test_a_field_question_selects_only_its_own_edge():
    candidates = graph_relationship_candidates(
        "what is the sample number", present_relationships=FIXTURE_EDGES
    )
    assert "HAS_SAMPLE_NUMBER" in candidates
    assert "REQUESTED_BY" not in candidates, candidates


def test_structural_edges_are_never_offered():
    """CONTAINS_PAGE and HAS_FIELD stitch the graph together; they assert nothing."""
    candidates = graph_relationship_candidates(
        "which page contains the field", present_relationships=FIXTURE_EDGES
    )
    for structural in ("CONTAINS_PAGE", "HAS_FIELD", "MATCHES"):
        assert structural not in candidates, candidates


def test_an_unrelated_question_selects_nothing():
    """Offering every edge for any question would make the filter meaningless."""
    assert (
        graph_relationship_candidates("what colour is the sky", present_relationships=FIXTURE_EDGES)
        == []
    )


def test_a_who_question_does_not_pull_in_non_agent_edges():
    candidates = graph_relationship_candidates("who signed it", present_relationships=FIXTURE_EDGES)
    assert "HAS_BATCH_SIZE" not in candidates
    assert "HAS_PRODUCT_TYPE" not in candidates
    assert "REQUESTED_BY" in candidates


def test_candidate_count_stays_bounded():
    """One broad question must not widen the Cypher filter without limit."""
    many = [f"HANDLED_BY_PARTY_{index}" for index in range(200)]
    candidates = graph_relationship_candidates("who handled this", present_relationships=many)
    assert 0 < len(candidates) <= 40, len(candidates)


@pytest.mark.parametrize(
    "hostile",
    [
        "REQUESTED_BY]->() DETACH DELETE n //",
        "requested_by; DROP DATABASE neo4j",
        "  ",
        "'",
        "a" * 200,
    ],
)
def test_a_hostile_relationship_name_never_reaches_cypher(hostile):
    """Edge types come from the graph, but the graph is written from document
    text, so they are still filtered before they can reach a query."""
    candidates = graph_relationship_candidates(
        "who requested this", present_relationships=[hostile, "REQUESTED_BY"]
    )
    assert all(candidate.replace("_", "").isalnum() for candidate in candidates), candidates
    assert hostile.upper() not in candidates


def test_present_relationships_is_optional():
    """Existing callers must keep working unchanged."""
    assert isinstance(graph_relationship_candidates("who issued this invoice"), list)


def test_case_and_whitespace_in_present_relationships_are_normalised():
    candidates = graph_relationship_candidates(
        "who requested this", present_relationships=["  requested_by  "]
    )
    assert "REQUESTED_BY" in candidates
