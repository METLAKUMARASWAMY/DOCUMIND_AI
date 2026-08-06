from __future__ import annotations

from uuid import uuid4

from app.core.security import hash_document, utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionAskRequest,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionProcessingStatus,
)
from app.services.vision.graph_builder import evidence_graph_builder
from app.services.vision.persistence import VisionPersistenceCoordinator
from app.services.vision.service import RETRIEVAL_SOURCE_SEQUENCE, VisionAIService


def _field(
    document_id: str,
    name: str,
    value: str,
    source_text: str,
    *,
    status: ReviewerStatus = ReviewerStatus.accepted,
) -> ExtractedVisionField:
    return ExtractedVisionField(
        name=name,
        value=value,
        confidence=0.94,
        calibrated=True,
        reviewer_status=status,
        validation_status="valid",
        extractor="deterministic-schema-extractor-v3",
        evidence=VisionEvidence(
            document_id=document_id,
            page_number=1,
            source_text=source_text,
            bounding_box=VisionBoundingBox(x1=10, y1=10, x2=420, y2=42),
            model_confidence=0.94,
            model_name="deterministic-schema-extractor",
            model_version="v3",
            processing_timestamp=utc_now_iso(),
            reviewer_status=status,
        ),
    )


def _business_form() -> VisionDocumentRecord:
    document_id = f"vision-doc-rag-{uuid4().hex[:10]}"
    text = (
        "Date: September 21, 1976\n"
        "Sample No.: 6030\n"
        "Type of Cigarette: 85 mm Filter\n"
        "Batch Size: 50 lbs.\n"
        "Original Request Made By: Dr. A. W. Spears\n"
        "Sample Specifications Written By: W. E. Routh\n"
    )
    now = utc_now_iso()
    document = VisionDocumentRecord(
        document_id=document_id,
        case_id=f"case-{uuid4().hex[:8]}",
        filename="00093726.png",
        tenant_id="tenant-rag-v4",
        status=VisionProcessingStatus.review_required,
        document_type=VisionDocumentType.business_form,
        file_hash=hash_document(text.encode()),
        parsed_text=text,
        created_at=now,
        updated_at=now,
        metadata={"pipeline_version": "vision-pipeline-v4"},
        extracted_fields=[
            _field(document_id, "date", "September 21, 1976", "Date: September 21, 1976"),
            _field(document_id, "sample_number", "6030", "Sample No.: 6030"),
            _field(document_id, "product_type", "85 mm Filter", "Type of Cigarette: 85 mm Filter"),
            _field(document_id, "batch_size", "50 lbs.", "Batch Size: 50 lbs."),
            _field(
                document_id,
                "requested_by",
                "Dr. A. W. Spears",
                "Original Request Made By: Dr. A. W. Spears",
            ),
            _field(
                document_id,
                "specifications_written_by",
                "W. E. Routh",
                "Sample Specifications Written By: W. E. Routh",
            ),
            _field(
                document_id,
                "document_title",
                "September 21, 1976",
                "Date: September 21, 1976",
                status=ReviewerStatus.unverified,
            ),
        ],
    )
    document.graph = evidence_graph_builder.build(document)
    return document


def test_semantic_graph_ontology_has_source_linked_business_relationships():
    document = _business_form()
    node_types = {node["type"] for node in document.graph.nodes}
    relationships = {edge["type"] for edge in document.graph.edges}

    assert {"Document", "Page", "Date", "Identifier", "Product", "Amount", "Person"} <= node_types
    assert {
        "HAS_DATE",
        "HAS_SAMPLE_NUMBER",
        "HAS_PRODUCT_TYPE",
        "HAS_BATCH_SIZE",
        "REQUESTED_BY",
        "SPECIFICATIONS_WRITTEN_BY",
    } <= relationships
    assert not any(
        node["type"] == "Form" and node.get("value") == "September 21, 1976"
        for node in document.graph.nodes
    )
    for edge in document.graph.edges:
        assert edge["tenant_id"] == document.tenant_id
        assert "source_field_id" in edge
        assert "source_text" in edge
        assert "bounding_box" in edge
        assert "verification_status" in edge
        assert edge["pipeline_version"] == "vision-pipeline-v4"


def test_graphrag_query_returns_readable_path_and_field_evidence():
    service = VisionAIService()
    document = _business_form()
    service.documents[document.document_id] = document

    response = service.ask(
        VisionAskRequest(
            question="Who requested this sample?",
            scope="current_document",
            document_id=document.document_id,
        ),
        document.tenant_id,
        conversation_id=f"graph-query-{uuid4().hex}",
        user_id="reviewer-graph",
    )

    assert response.answer_status == "supported"
    assert "Dr. A. W. Spears" in response.answer
    assert response.retrieval_path == "graph_path_query"
    assert response.graph_paths[0]["relationships"] == ["REQUESTED_BY"]
    assert response.graph_paths[0]["source_field"] == "requested_by"
    assert response.citations[0].source_text == "Original Request Made By: Dr. A. W. Spears"
    assert response.citations[0].bounding_boxes
    assert response.citations[0].evidence_type == "graph_path"


def test_yes_preserves_pending_clarification_until_an_option_is_selected():
    service = VisionAIService()
    document = _business_form()
    service.documents[document.document_id] = document
    conversation_id = f"clarification-{uuid4().hex}"

    ambiguous = service.ask(
        VisionAskRequest(
            question="Is this safe?",
            scope="current_document",
            document_id=document.document_id,
        ),
        document.tenant_id,
        conversation_id=conversation_id,
        user_id="reviewer-conversation",
    )
    reloaded_service = VisionAIService()
    reloaded_service.documents[document.document_id] = document
    acknowledgement = reloaded_service.ask(
        VisionAskRequest(question="yes", conversation_id=conversation_id),
        document.tenant_id,
        user_id="reviewer-conversation",
    )
    selected = reloaded_service.ask(
        VisionAskRequest(
            question="Check approval status",
            conversation_id=conversation_id,
            clarification_option="approval_status",
        ),
        document.tenant_id,
        user_id="reviewer-conversation",
    )

    assert ambiguous.answer_status == "clarification_needed"
    assert len(ambiguous.clarification_options) == 5
    assert acknowledgement.answer == "Please select which type of safety check you want me to perform."
    assert acknowledgement.query_type == "clarification_response"
    assert acknowledgement.retriever == "not_used"
    assert selected.query_type == "approval_question"
    assert selected.resolved_question == "What is the current approval status?"
    assert selected.scope["documents"][0]["document_id"] == document.document_id
    assert selected.citations
    assert selected.retrieval_trace is not None
    assert selected.retrieval_trace.sources_checked == list(RETRIEVAL_SOURCE_SEQUENCE)
    assert selected.retrieval_trace.selected_evidence

    cache_key = reloaded_service._conversation_cache_key(
        conversation_id,
        document.tenant_id,
        "reviewer-conversation",
    )
    state = reloaded_service.conversations[cache_key]
    assert state.pending_clarification == {}
    assert len(state.turns) == 6


def test_neo4j_rebuild_removes_stale_projected_nodes_before_upsert():
    document = _business_form()

    class Result:
        def consume(self):
            return None

    class Transaction:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def run(self, statement: str, **parameters):
            self.calls.append((statement, parameters))
            return Result()

    transaction = Transaction()
    VisionPersistenceCoordinator._upsert_graph_tx(
        transaction,
        document.tenant_id,
        document.document_id,
        document.graph.model_dump(mode="json"),
    )

    stale_cleanup = next(
        (statement, parameters)
        for statement, parameters in transaction.calls
        if "DETACH DELETE node" in statement
    )
    assert set(stale_cleanup[1]["node_ids"]) == {
        node["id"] for node in document.graph.nodes
    }
    assert any("[r:REQUESTED_BY" in statement for statement, _ in transaction.calls)
