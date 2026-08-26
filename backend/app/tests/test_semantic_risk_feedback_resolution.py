from __future__ import annotations

import json

from app.core.config import settings
from app.models.schemas import TextAnalysisRequest
from app.models.vision import (
    ExtractedVisionField,
    GenericDiscoveredField,
    ReviewerStatus,
    TypedVisionEntity,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionSemanticType,
)
from app.services.vision.entity_resolution import ConservativeEntityResolver
from app.services.vision.graph_builder import evidence_graph_builder
from app.services.vision.reviewer_feedback import ReviewerFeedbackDataset
from app.services.vision.semantic_risk import GenericSemanticRiskEngine
from app.services.vision.service import VisionAIService


def _evidence(document_id: str, evidence_id: str, text: str) -> VisionEvidence:
    return VisionEvidence(
        evidence_id=evidence_id,
        document_id=document_id,
        page_number=1,
        source_text=text,
        bounding_box=VisionBoundingBox(x1=10, y1=10, x2=300, y2=30),
        model_confidence=0.91,
        model_name="test-generic-extractor",
        model_version="v1",
        processing_timestamp="2026-08-13T00:00:00+00:00",
    )


def _generic_field(
    document_id: str,
    *,
    field_id: str,
    label: str,
    name: str,
    value: str,
    normalized_value: str,
    semantic_type: VisionSemanticType,
) -> GenericDiscoveredField:
    evidence = _evidence(document_id, f"ev-{field_id}", f"{label}: {value}")
    return GenericDiscoveredField(
        field_id=field_id,
        raw_label=label,
        normalized_label=name,
        raw_value=value,
        normalized_value=normalized_value,
        page=1,
        label_bbox=VisionBoundingBox(x1=10, y1=10, x2=130, y2=30),
        value_bbox=VisionBoundingBox(x1=140, y1=10, x2=300, y2=30),
        confidence=0.91,
        source_engine="test-generic-extractor",
        semantic_type_if_known=semantic_type,
        semantic_mapping_confidence=0.9,
        evidence=evidence,
    )


def _document(
    document_id: str,
    tenant_id: str,
    *,
    generic_fields: list[GenericDiscoveredField] | None = None,
    entities: list[TypedVisionEntity] | None = None,
    extracted_fields: list[ExtractedVisionField] | None = None,
) -> VisionDocumentRecord:
    return VisionDocumentRecord(
        document_id=document_id,
        case_id=f"case-{document_id}",
        filename=f"{document_id}.txt",
        tenant_id=tenant_id,
        document_type=VisionDocumentType.unknown,
        file_hash=(document_id[-1] if document_id else "a") * 64,
        generic_fields=generic_fields or [],
        typed_entities=entities or [],
        extracted_fields=extracted_fields or [],
        created_at="2026-08-13T00:00:00+00:00",
        updated_at="2026-08-13T00:00:00+00:00",
    )


def _entity(
    entity_id: str,
    value: str,
    semantic_type: VisionSemanticType,
    evidence_id: str,
    *,
    confidence: float = 0.91,
    context: dict | None = None,
) -> TypedVisionEntity:
    return TypedVisionEntity(
        entity_id=entity_id,
        raw_value=value,
        normalized_value=value.casefold(),
        semantic_type=semantic_type,
        confidence=confidence,
        evidence_ids=[evidence_id],
        context=context or {},
    )


def _graph_organisation_document(
    document_id: str,
    *,
    resolution: dict,
) -> VisionDocumentRecord:
    evidence = _evidence(
        document_id,
        f"ev-{document_id}-organisation",
        "Organisation: ABC Private Limited",
    )
    return _document(
        document_id,
        "tenant-graph-resolution",
        extracted_fields=[
            ExtractedVisionField(
                field_id=f"field-{document_id}-organisation",
                name="organisation_name",
                value="ABC Private Limited",
                normalized_value="ABC Private Limited",
                confidence=0.91,
                validation_status="valid",
                evidence=evidence,
            )
        ],
        entities=[
            _entity(
                f"entity-{document_id}-organisation",
                "ABC Private Limited",
                VisionSemanticType.organisation,
                evidence.evidence_id,
                context={"entity_resolution": resolution},
            )
        ],
    )


def test_generic_typed_bank_account_drives_safe_evidence_grounded_risk() -> None:
    document_id = "doc-generic-bank"
    account = _generic_field(
        document_id,
        field_id="payment-account",
        label="Requested Payment Account",
        name="requested_payment_account",
        value="ACCT-99999999",
        normalized_value="ACCT99999999",
        semantic_type=VisionSemanticType.bank_account,
    )
    document = _document(document_id, "tenant-risk-a", generic_fields=[account])

    findings = GenericSemanticRiskEngine().evaluate(
        document,
        {"approved_bank_account": "ACCT-11112222"},
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "business_risk.semantic_bank_account_mismatch"
    assert finding.evidence[0].evidence_id == account.evidence.evidence_id
    assert finding.confidence >= 0.8
    assert "human verification" in finding.description
    assert "fraud" not in finding.description.casefold()
    assert "99999999" not in finding.source_text
    assert "11112222" not in finding.source_text


def test_generic_date_aliases_drive_sequence_rule_without_template_fields() -> None:
    document_id = "doc-generic-dates"
    issued = _generic_field(
        document_id,
        field_id="bill-date",
        label="Bill Date",
        name="bill_date",
        value="2026-08-12",
        normalized_value="2026-08-12",
        semantic_type=VisionSemanticType.date,
    )
    deadline = _generic_field(
        document_id,
        field_id="payment-deadline",
        label="Payment Deadline",
        name="payment_deadline",
        value="2026-08-01",
        normalized_value="2026-08-01",
        semantic_type=VisionSemanticType.date,
    )
    findings = GenericSemanticRiskEngine().evaluate(
        _document(document_id, "tenant-risk-a", generic_fields=[issued, deadline]),
        {},
    )

    assert [item.rule_id for item in findings] == [
        "business_risk.semantic_due_date_precedes_issue_date"
    ]
    assert {item.evidence_id for item in findings[0].evidence} == {
        issued.evidence.evidence_id,
        deadline.evidence.evidence_id,
    }


def test_ambiguous_numeric_date_order_abstains_from_sequence_risk() -> None:
    document_id = "doc-ambiguous-dates"
    issued = _generic_field(
        document_id,
        field_id="ambiguous-bill-date",
        label="Bill Date",
        name="bill_date",
        value="03/04/2026",
        normalized_value="03/04/2026",
        semantic_type=VisionSemanticType.date,
    )
    deadline = _generic_field(
        document_id,
        field_id="ambiguous-deadline",
        label="Payment Deadline",
        name="payment_deadline",
        value="04/03/2026",
        normalized_value="04/03/2026",
        semantic_type=VisionSemanticType.date,
    )

    findings = GenericSemanticRiskEngine().evaluate(
        _document(document_id, "tenant-risk-a", generic_fields=[issued, deadline]),
        {},
    )

    assert all(
        item.rule_id != "business_risk.semantic_due_date_precedes_issue_date" for item in findings
    )


def test_duplicate_invoice_fact_is_tenant_scoped_and_cites_both_documents() -> None:
    first_id = "doc-invoice-one"
    second_id = "doc-invoice-two"
    other_tenant_id = "doc-invoice-other-tenant"
    first = _document(
        first_id,
        "tenant-invoice-a",
        generic_fields=[
            _generic_field(
                first_id,
                field_id="invoice-one",
                label="Invoice Number",
                name="invoice_number",
                value="INV-991",
                normalized_value="INV-991",
                semantic_type=VisionSemanticType.document_reference,
            )
        ],
    )
    second = _document(
        second_id,
        "tenant-invoice-a",
        generic_fields=[
            _generic_field(
                second_id,
                field_id="invoice-two",
                label="Invoice Number",
                name="invoice_number",
                value="INV-991",
                normalized_value="INV-991",
                semantic_type=VisionSemanticType.document_reference,
            )
        ],
    )
    other_tenant = _document(
        other_tenant_id,
        "tenant-invoice-b",
        generic_fields=[
            _generic_field(
                other_tenant_id,
                field_id="invoice-other",
                label="Invoice Number",
                name="invoice_number",
                value="INV-991",
                normalized_value="INV-991",
                semantic_type=VisionSemanticType.document_reference,
            )
        ],
    )

    findings = GenericSemanticRiskEngine().evaluate(
        first,
        {},
        [first, second, other_tenant],
    )

    duplicate = next(
        item for item in findings if item.rule_id == "business_risk.duplicate_invoice_identifier"
    )
    assert {item.document_id for item in duplicate.evidence} == {first_id, second_id}
    assert other_tenant_id not in duplicate.source_text


def test_entity_resolution_deduplicates_evidence_and_never_merges_names_only() -> None:
    resolver = ConservativeEntityResolver()
    current = _document(
        "doc-current",
        "tenant-resolution",
        entities=[
            _entity("email-1", "ops@example.test", VisionSemanticType.email, "ev-email-1"),
            _entity("email-2", "OPS@example.test", VisionSemanticType.email, "ev-email-2"),
            _entity("org-1", "ABC Pvt. Ltd.", VisionSemanticType.organisation, "ev-org-1"),
        ],
    )
    candidate = _document(
        "doc-candidate",
        "tenant-resolution",
        entities=[
            _entity("email-3", "ops@example.test", VisionSemanticType.email, "ev-email-3"),
            _entity(
                "org-2",
                "ABC Private Limited",
                VisionSemanticType.organisation,
                "ev-org-2",
            ),
        ],
    )
    other_tenant = _document(
        "doc-other-tenant",
        "tenant-other",
        entities=[_entity("email-4", "ops@example.test", VisionSemanticType.email, "ev-email-4")],
    )

    report = resolver.resolve(current, [current, candidate, other_tenant])

    assert report.input_entity_count == 3
    assert report.deduplicated_entity_count == 2
    email = next(item for item in report.entities if item.semantic_type == VisionSemanticType.email)
    assert set(email.evidence_ids) == {"ev-email-1", "ev-email-2"}
    email_match = next(
        item for item in report.decisions if item.semantic_type == VisionSemanticType.email
    )
    assert email_match.status == "resolved"
    assert email_match.automatic_merge is True
    assert email_match.matched_document_id == candidate.document_id
    organisation = next(
        item for item in report.decisions if item.semantic_type == VisionSemanticType.organisation
    )
    assert organisation.status == "possible_match"
    assert organisation.automatic_merge is False
    assert organisation.method == "legal_suffix_normalization"
    assert all(item.matched_document_id != other_tenant.document_id for item in report.decisions)


def test_graph_keeps_unresolved_name_entities_document_scoped() -> None:
    first = _graph_organisation_document(
        "doc-graph-name-a",
        resolution={
            "canonical_entity_id": None,
            "observation_entity_id": "observation-a",
            "automatic_merge": False,
            "decision_statuses": ["possible_match"],
            "resolution_confidence": 0.76,
            "resolution_methods": ["exact_name_only"],
        },
    )
    second = _graph_organisation_document(
        "doc-graph-name-b",
        resolution={
            "canonical_entity_id": None,
            "observation_entity_id": "observation-b",
            "automatic_merge": False,
            "decision_statuses": ["possible_match"],
            "resolution_confidence": 0.76,
            "resolution_methods": ["exact_name_only"],
        },
    )

    first_graph = evidence_graph_builder.build(first)
    second_graph = evidence_graph_builder.build(second)
    first_node = next(item for item in first_graph.nodes if item["type"] == "Organisation")
    second_node = next(item for item in second_graph.nodes if item["type"] == "Organisation")

    assert first_node["id"] != second_node["id"]
    assert first_node["automatic_entity_merge"] is False
    assert first_node.get("canonical_entity_id") is None
    assert first_node["entity_resolution_status"] == "possible_match"
    assert first_node["entity_resolution_confidence"] == 0.76


def test_graph_uses_reviewer_resolved_canonical_id_only_for_approved_merge() -> None:
    resolution = {
        "canonical_entity_id": "resolved-entity-reviewed-abc",
        "observation_entity_id": "observation-local",
        "automatic_merge": True,
        "decision_statuses": ["resolved"],
        "resolution_confidence": 0.96,
        "resolution_methods": ["reviewed_value_alias"],
    }
    first = _graph_organisation_document("doc-graph-resolved-a", resolution=resolution)
    second = _graph_organisation_document("doc-graph-resolved-b", resolution=resolution)

    first_node = next(
        item for item in evidence_graph_builder.build(first).nodes if item["type"] == "Organisation"
    )
    second_node = next(
        item
        for item in evidence_graph_builder.build(second).nodes
        if item["type"] == "Organisation"
    )

    assert first_node["id"] == second_node["id"]
    assert first_node["automatic_entity_merge"] is True
    assert first_node["canonical_entity_id"] == "resolved-entity-reviewed-abc"
    assert first_node["entity_resolution_method"] == "reviewed_value_alias"


def test_reviewer_approved_value_alias_can_resolve_but_conflicting_feedback_abstains(
    tmp_path,
) -> None:
    dataset = ReviewerFeedbackDataset(tmp_path / "feedback")
    base = {
        "tenant_id": "tenant-feedback",
        "document_id": "doc-feedback",
        "status": ReviewerStatus.edited.value,
        "semantic_type": VisionSemanticType.organisation.value,
        "raw_label": "Supplier Designation",
        "corrected_field_name": "supplier_name",
        "original_value": "ABC Pvt. Ltd.",
        "corrected_value": "ABC Private Limited",
    }
    dataset.append(base)
    reloaded = ReviewerFeedbackDataset(tmp_path / "feedback")

    assert reloaded.approved_label_aliases(
        "tenant-feedback",
        normalizer=lambda value: value.casefold().replace(" ", "_"),
    ) == {"supplier_designation": "supplier_name"}
    aliases = reloaded.reviewed_value_aliases("tenant-feedback")
    assert aliases[VisionSemanticType.organisation.value]["abc pvt ltd"] == ("abc private limited")

    current = _document(
        "doc-reviewed-alias-a",
        "tenant-feedback",
        entities=[
            _entity(
                "org-reviewed-a",
                "ABC Pvt. Ltd.",
                VisionSemanticType.organisation,
                "ev-reviewed-a",
            )
        ],
    )
    candidate = _document(
        "doc-reviewed-alias-b",
        "tenant-feedback",
        entities=[
            _entity(
                "org-reviewed-b",
                "ABC Private Limited",
                VisionSemanticType.organisation,
                "ev-reviewed-b",
            )
        ],
    )
    decision = (
        ConservativeEntityResolver()
        .resolve(
            current,
            [candidate],
            reviewed_value_aliases=aliases,
        )
        .decisions[0]
    )
    assert decision.status == "resolved"
    assert decision.method == "reviewed_value_alias"
    assert decision.automatic_merge is True

    # A corrupt/copied cross-tenant record in this tenant's shard is ignored.
    with dataset.path_for("tenant-feedback").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    **base,
                    "tenant_id": "tenant-other",
                    "corrected_field_name": "poisoned_name",
                }
            )
            + "\n"
        )
        valid_record = dataset.records("tenant-feedback")[0]
        handle.write(
            json.dumps(
                {
                    **valid_record,
                    "corrected_field_name": "tampered_name",
                }
            )
            + "\n"
        )
    assert all(
        item["tenant_id"] == "tenant-feedback" for item in reloaded.records("tenant-feedback")
    )
    assert len(reloaded.records("tenant-feedback")) == 1

    dataset.append({**base, "corrected_field_name": "conflicting_supplier_name"})
    assert (
        reloaded.approved_label_aliases(
            "tenant-feedback",
            normalizer=lambda value: value.casefold().replace(" ", "_"),
        )
        == {}
    )


def test_feedback_alias_survives_service_restart_reprocess_and_tenant_boundary(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    tenant_id = "tenant-feedback-restart"
    source_text = "Mystery Record\nSupplier Designation: Northwind Alpha\n"
    first_service = VisionAIService()
    first = first_service.process_text(
        TextAnalysisRequest(filename="unseen-original.txt", text=source_text),
        tenant_id=tenant_id,
    )
    original = next(item for item in first.extracted_fields if item.name == "supplier_designation")
    assert first.review_task is not None
    first_service.patch_field(
        first.review_task.review_id,
        original.field_id,
        reviewed_value=original.value,
        reviewed_field_name="corporate_contact",
        reviewer_status=ReviewerStatus.edited,
        comment="Reviewed reusable semantic correction.",
        actor="reviewer@example.test",
        tenant_id=tenant_id,
    )

    restarted = VisionAIService()
    after_restart = restarted.process_text(
        TextAnalysisRequest(
            filename="unseen-after-restart.txt",
            text="Another Record\nSupplier Designation: Contoso Beta\n",
        ),
        tenant_id=tenant_id,
    )
    assert any(item.name == "corporate_contact" for item in after_restart.extracted_fields)

    isolated = restarted.process_text(
        TextAnalysisRequest(
            filename="unseen-other-tenant.txt",
            text="Third Record\nSupplier Designation: Fabrikam Gamma\n",
        ),
        tenant_id="tenant-feedback-isolated",
    )
    assert any(item.name == "supplier_designation" for item in isolated.extracted_fields)
    assert all(item.name != "corporate_contact" for item in isolated.extracted_fields)

    reprocess = restarted.queue_reprocess(
        first.document_id,
        source_text.encode(),
        actor="reviewer@example.test",
        tenant_id=tenant_id,
    )
    restarted.process_job(reprocess.job_id, tenant_id)
    reprocessed = restarted.get_document(first.document_id, tenant_id)
    assert any(item.name == "corporate_contact" for item in reprocessed.extracted_fields)
    assert reprocessed.metadata["reprocess_count"] == 1
    summary = restarted.feedback_dataset.summary(tenant_id)
    assert summary["record_count"] == 1
    assert summary["automatic_retraining"] is False
    exported = restarted.active_learning_export(tenant_id)
    assert exported["total_corrections"] == 1
    assert exported["persisted_corrections"] == 1
    assert exported["legacy_live_corrections"] == 0
    assert exported["summary"]["record_count"] == 1
    assert exported["corrections"][0]["field_name"] == "corporate_contact"
    assert exported["corrections"][0]["evidence"]["evidence_id"]
    assert restarted.feedback_dataset.records(tenant_id)[0]["reviewer_comment"] == (
        "Reviewed reusable semantic correction."
    )


def test_active_learning_export_masks_sensitive_persisted_values(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    service = VisionAIService()
    service.feedback_dataset.append(
        {
            "tenant_id": "tenant-sensitive-export",
            "document_id": "doc-sensitive-export",
            "document_type": "invoice",
            "field_id": "field-account",
            "raw_label": "Payment Account",
            "original_field_name": "requested_payment_account",
            "corrected_field_name": "bank_account_number",
            "original_value": "ACCT-11112222",
            "corrected_value": "ACCT-99998888",
            "semantic_type": VisionSemanticType.bank_account.value,
            "source_text": "Payment Account: ACCT-99998888",
            "evidence_id": "ev-sensitive-account",
            "page": 1,
            "status": ReviewerStatus.edited.value,
        }
    )

    exported = service.active_learning_export("tenant-sensitive-export")

    correction = exported["corrections"][0]
    assert "11112222" not in correction["model_value"]
    assert "99998888" not in correction["reviewed_value"]
    assert correction["model_value"].endswith("2222")
    assert correction["reviewed_value"].endswith("8888")
    assert correction["evidence"]["source_text"] == ("Sensitive source text withheld from export.")


def test_service_runs_generic_semantic_risk_and_resolution_hooks(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    service = VisionAIService()
    document = service.process_text(
        TextAnalysisRequest(
            filename="unknown-payment-request.txt",
            text=(
                "Settlement Instruction\n"
                "Requested Payment Account: ACCT-99999999\n"
                "Contact Email: treasury@example.test\n"
            ),
            metadata={"approved_bank_account": "ACCT-11112222"},
        ),
        tenant_id="tenant-generic-risk-hook",
    )

    semantic_risk = next(
        item
        for item in document.risk_indicators
        if item.rule_id == "business_risk.semantic_bank_account_mismatch"
    )
    assert semantic_risk.model_name == "generic-semantic-risk-engine"
    assert semantic_risk.evidence
    assert semantic_risk.affected_fields
    assert document.metadata["entity_resolution"]["tenant_id"] == document.tenant_id
    assert document.metadata["entity_resolution"]["model_version"] == (
        "tenant-entity-resolution-v1"
    )
