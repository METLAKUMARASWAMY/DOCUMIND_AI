from urllib.parse import quote

from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())


def _create_api_case() -> dict:
    response = client.post(
        "/api/v1/demo-legacy/documents/analyze",
        json={
            "filename": "invoice-api-9001.txt",
            "text": (
                "Invoice INV-9001\n"
                "Vendor: Aster Analytics Inc\n"
                "Amount Due USD 10,000.00\n"
                "Payment terms Net 30\n"
                "Manual override was applied.\n"
                "New bank account requested for urgent wire."
            ),
            "metadata": {"business_unit": "Finance"},
        },
    )
    assert response.status_code == 200
    return response.json()["data"]


def test_case_graph_evidence_agent_and_finding_endpoints():
    case = _create_api_case()
    case_id = case["case_id"]

    cases = client.get("/api/v1/demo-legacy/cases")
    assert cases.status_code == 200
    assert cases.json()["success"] is True

    detail = client.get(f"/api/v1/demo-legacy/cases/{case_id}")
    assert detail.status_code == 200
    assert detail.json()["case_id"] == case_id

    evidence = client.get(f"/api/v1/demo-legacy/cases/{case_id}/evidence")
    assert evidence.status_code == 200
    assert evidence.json()["data"]["evidence_count"] > 0

    graph = client.get(f"/api/v1/demo-legacy/graph/cases/{case_id}")
    assert graph.status_code == 200
    edge_types = {edge["type"] for edge in graph.json()["edges"]}
    assert {
        "CASE_CONTAINS_DOCUMENT",
        "CASE_HAS_RISK",
        "FINDING_SUPPORTED_BY_EVIDENCE",
        "VENDOR_ISSUED_INVOICE",
    }.issubset(edge_types)

    entity_id = quote("vendor:aster analytics inc", safe="")
    entity = client.get(f"/api/v1/demo-legacy/graph/entities/{entity_id}", params={"case_id": case_id})
    assert entity.status_code == 200
    assert entity.json()["matches"][0]["relationship_count"] >= 1

    risk_paths = client.get("/api/v1/demo-legacy/graph/risk-paths", params={"min_risk_score": 1})
    assert risk_paths.status_code == 200
    assert any(item["case_id"] == case_id for item in risk_paths.json()["risk_paths"])

    projected = client.post(f"/api/v1/demo-legacy/graph/project/{case_id}")
    assert projected.status_code == 200
    assert projected.json()["status"] == "projected"

    compliance = client.get("/api/v1/demo-legacy/compliance/findings", params={"case_id": case_id})
    assert compliance.status_code == 200
    assert compliance.json()["total"] >= 1

    fraud = client.get("/api/v1/demo-legacy/fraud/findings", params={"case_id": case_id})
    assert fraud.status_code == 200
    assert fraud.json()["total"] >= 1

    agents = client.get(f"/api/v1/demo-legacy/agents/runs/{case_id}")
    assert agents.status_code == 200
    assert agents.json()["agent_count"] >= 5


def test_readiness_endpoints_cover_later_phases():
    endpoints = [
        "/api/v1/demo-legacy/mlops/readiness",
        "/api/v1/demo-legacy/security/readiness",
        "/api/v1/demo-legacy/deployment/readiness",
        "/api/v1/demo-legacy/testing/readiness",
        "/api/v1/demo-legacy/research/readiness",
        "/api/v1/demo-legacy/portfolio/readiness",
    ]
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json()["status"]
