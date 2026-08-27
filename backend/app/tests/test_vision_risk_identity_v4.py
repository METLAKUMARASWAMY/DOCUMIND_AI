from __future__ import annotations

from app.models.vision import (
    VisionDocumentRecord,
    VisionProcessingStatus,
    VisionRiskSeverity,
)
from app.services.vision.service import VisionAIService


def _document() -> VisionDocumentRecord:
    return VisionDocumentRecord(
        document_id="vision-doc-stable-risk",
        case_id="vision-case-stable-risk",
        filename="stable-risk.txt",
        file_hash="a" * 64,
        status=VisionProcessingStatus.review_required,
        parsed_text="A stable deterministic rule observation.",
        created_at="2026-08-06T00:00:00+00:00",
        updated_at="2026-08-06T00:00:00+00:00",
    )


def test_risk_and_graph_node_identity_is_stable_across_reprocessing() -> None:
    service = VisionAIService()
    document = _document()
    arguments = {
        "title": "Stable validation warning",
        "severity": VisionRiskSeverity.medium,
        "category": "validation",
        "description": "The same rule observation should retain its identity.",
        "recommended_action": "Review the source evidence.",
        "source_text": "A stable deterministic rule observation.",
        "field_names": ["sample_number"],
        "rule_id": "validation.stable_observation",
    }

    first = service._risk(document, **arguments)
    second = service._risk(document, **arguments)
    document.risk_indicators = [first]
    first_graph = service._graph(document)
    document.risk_indicators = [second]
    second_graph = service._graph(document)

    assert first.risk_id == second.risk_id
    assert {node["id"] for node in first_graph.nodes} == {node["id"] for node in second_graph.nodes}
    assert len({node["id"] for node in second_graph.nodes}) == len(second_graph.nodes)
