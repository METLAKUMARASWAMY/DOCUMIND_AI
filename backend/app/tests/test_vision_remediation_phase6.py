from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.vision.persistence import VisionPersistenceCoordinator
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _headers() -> dict[str, str]:
    return auth_headers(client)


def _form_graph() -> tuple[dict, dict[str, str]]:
    unique = uuid4().hex[:8]
    headers = _headers()
    created = client.post(
        "/api/v1/vision/documents/analyze",
        headers=headers,
        json={
            "filename": f"graph-registration-{unique}.txt",
            "text": (
                "Organisation Registration Form\n"
                f"Organisation Name: Graph Works {unique}\n"
                "Address: 42 Evidence Road\n"
                "Contact Name: Asha Rao\n"
                "FAX NO.: 212-403-2211\n"
                "Email: asha@example.org\n"
                "Signature: Asha Rao\n"
            ),
            "metadata": {"document_type_hint": "form"},
        },
    )
    assert created.status_code == 200, created.text
    document = created.json()["data"]
    response = client.get(
        f"/api/v1/graph/document/{document['document_id']}",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()["data"], headers


def test_form_graph_has_typed_bounded_nodes_and_source_linked_edges():
    graph, _ = _form_graph()
    node_types = {node["type"] for node in graph["nodes"]}

    assert {
        "Organisation",
        "Document",
        "Page",
        "ContactMethod",
        "ExtractedField",
        "Signature",
    } <= node_types
    assert len(graph["nodes"]) > 6
    assert graph["edges"]
    assert graph["preferred_backend"] == "neo4j"
    assert graph["active_backend"] == "local_evidence_graph"
    assert all(len(node["display_label"]) <= 30 for node in graph["nodes"])

    non_structural = [
        edge
        for edge in graph["edges"]
        if edge["provenance"] != "document_structure"
    ]
    assert non_structural
    assert all(edge["source_linked"] for edge in non_structural)
    assert all(edge["evidence_text"] for edge in non_structural)
    assert all(edge["source_page"] >= 1 for edge in non_structural)


def test_proposed_relationship_verification_is_evidence_gated_and_audited():
    graph, headers = _form_graph()
    relationship = next(
        edge
        for edge in graph["edges"]
        if edge["status"] == "proposed" and edge["verification_eligible"]
    )

    blocked = client.post(
        f"/api/v1/graph/relationships/{relationship['id']}/verify",
        headers=headers,
        json={"comment": "Evidence checked against source.", "evidence_acknowledged": False},
    )
    assert blocked.status_code == 422

    verified = client.post(
        f"/api/v1/graph/relationships/{relationship['id']}/verify",
        headers=headers,
        json={
            "comment": "Evidence checked against the exact source region.",
            "evidence_acknowledged": True,
        },
    )
    assert verified.status_code == 200, verified.text
    assert verified.json()["data"]["status"] == "verified"
    assert verified.json()["data"]["review_status"] == "accepted"
    assert verified.json()["data"]["verification_eligible"] is False
    assert verified.json()["data"]["verified_by"]

    document_id = relationship["document_id"]
    detail = client.get(f"/api/v1/documents/{document_id}", headers=headers)
    audit_types = {event["event_type"] for event in detail.json()["data"]["audit_events"]}
    assert "graph_relationship_verified" in audit_types


def test_neo4j_projection_uses_whitelisted_domain_labels_and_relationships():
    graph, _ = _form_graph()

    class Result:
        def consume(self):
            return None

    class Transaction:
        def __init__(self):
            self.statements: list[str] = []

        def run(self, statement: str, **parameters):
            del parameters
            self.statements.append(statement)
            return Result()

    transaction = Transaction()
    document_id = next(node["id"] for node in graph["nodes"] if node["type"] == "Document")
    VisionPersistenceCoordinator._upsert_graph_tx(
        transaction,
        "tenant-test",
        document_id,
        graph,
    )

    assert any(":Organisation" in statement for statement in transaction.statements)
    assert any("[r:HAS_CONTACT" in statement for statement in transaction.statements)
    assert all("VISION_RELATIONSHIP" not in statement for statement in transaction.statements)
