"""Regression tests for D-05: a fixture-shaped knowledge-graph ontology.

``EvidenceGraphBuilder._add_entity_relationships`` used to enumerate the field
names of one lab-request-form fixture — ``sample_number``, ``product_type``,
``batch_size``, ``requested_by``, ``specifications_written_by``, ``request_date``,
``purpose`` — in order to emit typed edges. A document whose schema used any
other field name got no typed relationship at all.

Those edges are now derived from the field name itself, so:

* every edge the fixture used to produce is still produced, byte for byte, and
* a document with field names nobody anticipated gets the same treatment.
"""

from __future__ import annotations

import pytest

from app.models.schemas import TextAnalysisRequest
from app.services.vision.graph_builder import EvidenceGraphBuilder
from app.services.vision.service import VisionAIService

TENANT = "tenant-graph-ontology"


# ------------------------------------------------------- the naming rule itself


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        # Exactly the edges the hardcoded block used to emit.
        ("sample_number", "HAS_SAMPLE_NUMBER"),
        ("product_type", "HAS_PRODUCT_TYPE"),
        ("batch_size", "HAS_BATCH_SIZE"),
        ("request_date", "HAS_REQUEST_DATE"),
        ("purpose", "HAS_PURPOSE"),
        ("requested_by", "REQUESTED_BY"),
        ("specifications_written_by", "SPECIFICATIONS_WRITTEN_BY"),
    ],
)
def test_previously_hardcoded_edges_are_reproduced(field_name: str, expected: str) -> None:
    assert EvidenceGraphBuilder._derived_relationship(field_name) == expected


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        # Field names from domains this project has never processed.
        ("policy_number", "HAS_POLICY_NUMBER"),
        ("consignee_name", "HAS_CONSIGNEE_NAME"),
        ("net_weight", "HAS_NET_WEIGHT"),
        ("claim_amount", "HAS_CLAIM_AMOUNT"),
        ("approved_by", "APPROVED_BY"),
        ("authorised_by", "AUTHORISED_BY"),
    ],
)
def test_unseen_field_names_receive_typed_edges(field_name: str, expected: str) -> None:
    assert EvidenceGraphBuilder._derived_relationship(field_name) == expected


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("Sample Number", "HAS_SAMPLE_NUMBER"),
        ("sample-number", "HAS_SAMPLE_NUMBER"),
        ("sample  number ", "HAS_SAMPLE_NUMBER"),
        ("__weird__name__", "HAS_WEIRD_NAME"),
    ],
)
def test_field_names_are_normalised(field_name: str, expected: str) -> None:
    assert EvidenceGraphBuilder._derived_relationship(field_name) == expected


@pytest.mark.parametrize("field_name", ["", "   ", "___", "!!!"])
def test_unusable_field_names_produce_no_edge(field_name: str) -> None:
    assert EvidenceGraphBuilder._derived_relationship(field_name) is None


@pytest.mark.parametrize(
    "field_name",
    [
        "sample_number",
        "product_type",
        "batch_size",
        "requested_by",
        "specifications_written_by",
        "request_date",
        "purpose",
    ],
)
def test_fixture_fields_are_not_in_the_enterprise_ontology(field_name: str) -> None:
    """They must be handled by the general rule, not by a special case."""

    assert field_name not in EvidenceGraphBuilder.ONTOLOGY_FIELDS, (
        f"{field_name!r} is still enumerated as a special case"
    )


# ------------------------------------------------------- end-to-end on a document


def _graph_for(text: str, *, filename: str) -> tuple[list, list]:
    service = VisionAIService()
    document = service.process_text(
        TextAnalysisRequest(filename=filename, text=text, metadata={}),
        actor="graph-ontology-test",
        tenant_id=TENANT,
    )
    return document.graph.nodes, document.graph.edges


def test_enterprise_ontology_still_emits_its_named_edges() -> None:
    _, edges = _graph_for(
        "Invoice INV-2026-7788\n"
        "Vendor: Cascade Supplies\n"
        "Invoice Date: 2026-06-01\n"
        "Total Amount: INR 90,000.00\n"
        "PO Number: PO-2026-3344\n",
        filename="ontology-invoice.txt",
    )
    labels = {str(edge.get("type") or "") for edge in edges}

    assert "ISSUED_BY" in labels, f"vendor relation missing from {sorted(labels)}"
    assert "REFERENCES" in labels


def test_a_document_from_an_unanticipated_domain_still_gets_typed_edges() -> None:
    """The acceptance criterion: field names this repo has never seen."""

    _, edges = _graph_for(
        "Marine Cargo Declaration\n"
        "Consignee Name: Harbour Freight Ltd\n"
        "Net Weight: 1,240 kg\n"
        "Port Of Loading: Rotterdam\n"
        "Cleared By: M. Okafor\n",
        filename="unseen-domain-declaration.txt",
    )
    labels = {str(edge.get("type") or "") for edge in edges}

    # Whatever was extracted must be typed, never left as a bare untyped edge.
    typed = {label for label in labels if label.startswith("HAS_") or label.endswith("_BY")}
    assert typed, f"no typed relationship was derived; got {sorted(labels)}"


def test_every_edge_carries_evidence_and_provenance() -> None:
    _, edges = _graph_for(
        "Invoice INV-2026-7788\nVendor: Cascade Supplies\nTotal Amount: INR 90,000.00\n",
        filename="ontology-evidence.txt",
    )

    assert edges, "expected a non-empty graph"
    for edge in edges:
        assert edge.get("type"), f"every edge must be labelled: {edge}"
        assert edge.get("status") in {"verified", "proposed", "machine_observed"}
        assert edge.get("source") and edge.get("target")
