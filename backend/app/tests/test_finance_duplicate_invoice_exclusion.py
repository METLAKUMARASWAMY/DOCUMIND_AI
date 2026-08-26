from __future__ import annotations

from app.core.config import settings
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionFinanceNarrativeRequest,
    VisionFinanceNarrativeType,
    VisionProcessingStatus,
    VisionReportRequest,
    VisionReportType,
)
from app.services.vision.finance_narratives import VerifiedFinanceNarrativeService
from app.services.vision.service import VisionAIService


def _field(
    document_id: str,
    name: str,
    value: str,
    *,
    reviewer_status: ReviewerStatus = ReviewerStatus.accepted,
) -> ExtractedVisionField:
    return ExtractedVisionField(
        name=name,
        value=value,
        reviewer_status=reviewer_status,
        evidence=VisionEvidence(
            evidence_id=f"evidence-{document_id}-{name}",
            document_id=document_id,
            source_text=f"{name}: {value}",
            model_confidence=0.99,
            model_name="test-finance-extractor",
            model_version="v1",
            processing_timestamp="2026-08-13T00:00:00+00:00",
            reviewer_status=reviewer_status,
        ),
    )


def _invoice(
    document_id: str,
    invoice_number: str,
    total: str,
    *,
    status: VisionProcessingStatus = VisionProcessingStatus.approved,
    invoice_number_status: ReviewerStatus = ReviewerStatus.accepted,
    vendor: str = "Verified Vendor",
) -> VisionDocumentRecord:
    return VisionDocumentRecord(
        document_id=document_id,
        case_id=f"case-{document_id}",
        filename=f"{document_id}.txt",
        tenant_id="tenant-finance",
        status=status,
        document_type=VisionDocumentType.invoice,
        file_hash=(document_id[-1] * 64),
        extracted_fields=[
            _field(
                document_id,
                "invoice_number",
                invoice_number,
                reviewer_status=invoice_number_status,
            ),
            _field(document_id, "vendor_name", vendor),
            _field(document_id, "total_amount", total),
            _field(document_id, "currency", "USD"),
        ],
        created_at="2026-08-13T00:00:00+00:00",
        updated_at="2026-08-13T00:00:00+00:00",
    )


def _request() -> VisionFinanceNarrativeRequest:
    return VisionFinanceNarrativeRequest(
        narrative_type=VisionFinanceNarrativeType.monthly_invoice_summary,
    )


def test_duplicate_invoice_number_keeps_approved_source_and_exposes_exclusion_trace():
    completed_duplicate = _invoice(
        "vision-doc-a-completed",
        "INV  -  001",
        "USD 100.00",
        status=VisionProcessingStatus.completed,
    )
    approved_source = _invoice("vision-doc-z-approved", "inv-001", "USD 100.00")
    unique_invoice = _invoice("vision-doc-unique", "INV-002", "USD 50.00")

    narrative = VerifiedFinanceNarrativeService().generate(
        _request(),
        [completed_duplicate, approved_source, unique_invoice],
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )

    assert narrative.document_ids == ["vision-doc-unique", "vision-doc-z-approved"]
    assert narrative.source_document_ids == [
        "vision-doc-a-completed",
        "vision-doc-unique",
        "vision-doc-z-approved",
    ]
    assert narrative.excluded_document_ids == ["vision-doc-a-completed"]
    assert "2 verified invoice(s)" in narrative.narrative
    assert "USD 150.00" in narrative.narrative
    assert "USD 250.00" not in narrative.narrative
    assert {citation.document_id for citation in narrative.citations} == {
        "vision-doc-a-completed",
        "vision-doc-z-approved",
        "vision-doc-unique",
    }
    excluded_citation = next(
        item for item in narrative.citations if item.document_id == "vision-doc-a-completed"
    )
    assert excluded_citation.evidence_type == "duplicate_exclusion_evidence"
    assert excluded_citation.extraction_status == "excluded_duplicate_source"

    exclusion = next(
        trace
        for trace in narrative.calculations
        if trace.label == "Duplicate invoice documents excluded"
    )
    assert exclusion.value == "1"
    assert exclusion.source_document_ids == [
        "vision-doc-z-approved",
        "vision-doc-a-completed",
    ]
    assert exclusion.source_evidence_ids == [
        "evidence-vision-doc-a-completed-invoice_number",
        "evidence-vision-doc-z-approved-invoice_number",
    ]
    assert "invoice_number=inv001" in exclusion.calculation_function
    assert "retained_document_id=vision-doc-z-approved" in exclusion.calculation_function
    assert "excluded_document_ids=vision-doc-a-completed" in exclusion.calculation_function


def test_unverified_invoice_number_is_not_used_to_infer_a_duplicate():
    verified = _invoice("vision-doc-verified", "INV-003", "USD 100.00")
    unverified = _invoice(
        "vision-doc-unverified",
        "inv 003",
        "USD 200.00",
        invoice_number_status=ReviewerStatus.unverified,
    )

    narrative = VerifiedFinanceNarrativeService().generate(
        _request(),
        [verified, unverified],
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )

    assert narrative.document_ids == ["vision-doc-unverified", "vision-doc-verified"]
    assert "2 verified invoice(s)" in narrative.narrative
    assert "USD 300.00" in narrative.narrative
    assert all(
        trace.label != "Duplicate invoice documents excluded" for trace in narrative.calculations
    )


def test_duplicate_approved_source_tie_break_is_independent_of_input_order():
    first_by_id = _invoice("vision-doc-a-approved", "INV/004", "USD 40.00")
    later_by_id = _invoice("vision-doc-z-approved", "inv 004", "USD 40.00")

    for documents in (
        [later_by_id, first_by_id],
        [first_by_id, later_by_id],
    ):
        narrative = VerifiedFinanceNarrativeService().generate(
            _request(),
            documents,
            tenant_id="tenant-finance",
            actor="reviewer@documind.test",
        )

        assert narrative.document_ids == ["vision-doc-a-approved"]
        assert "1 verified invoice(s)" in narrative.narrative
        assert "USD 40.00" in narrative.narrative


def test_non_latin_invoice_prefixes_are_not_collapsed_onto_the_same_digits():
    japanese_invoice = _invoice("vision-doc-japanese", "請求-005", "USD 5.00")
    chinese_invoice = _invoice("vision-doc-chinese", "发票-005", "USD 10.00")

    narrative = VerifiedFinanceNarrativeService().generate(
        _request(),
        [japanese_invoice, chinese_invoice],
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )

    assert narrative.document_ids == ["vision-doc-chinese", "vision-doc-japanese"]
    assert "2 verified invoice(s)" in narrative.narrative
    assert "USD 15.00" in narrative.narrative
    assert all(
        trace.label != "Duplicate invoice documents excluded" for trace in narrative.calculations
    )


def test_same_invoice_number_for_different_vendors_is_not_deduplicated():
    alpha = _invoice("vision-doc-alpha", "INV-007", "USD 70.00", vendor="Alpha LLC")
    beta = _invoice("vision-doc-beta", "inv 007", "USD 80.00", vendor="Beta LLC")

    narrative = VerifiedFinanceNarrativeService().generate(
        _request(),
        [beta, alpha],
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )

    assert narrative.document_ids == ["vision-doc-alpha", "vision-doc-beta"]
    assert "2 verified invoice(s)" in narrative.narrative
    assert "USD 150.00" in narrative.narrative


def test_conflicting_duplicate_invoice_values_require_review_instead_of_guessing():
    first = _invoice("vision-doc-first", "INV-008", "USD 80.00")
    conflict = _invoice("vision-doc-conflict", "inv 008", "USD 800.00")

    import pytest

    with pytest.raises(ValueError, match="Conflicting approved invoices"):
        VerifiedFinanceNarrativeService().generate(
            _request(),
            [first, conflict],
            tenant_id="tenant-finance",
            actor="reviewer@documind.test",
        )


def test_conflicting_tax_on_duplicate_invoice_requires_review():
    first = _invoice("vision-doc-tax-first", "INV-011", "USD 110.00")
    conflict = _invoice("vision-doc-tax-conflict", "inv 011", "USD 110.00")
    first.extracted_fields.append(_field(first.document_id, "tax", "USD 10.00"))
    conflict.extracted_fields.append(_field(conflict.document_id, "tax", "USD 20.00"))

    import pytest

    with pytest.raises(ValueError, match="review fields: tax"):
        VerifiedFinanceNarrativeService().generate(
            VisionFinanceNarrativeRequest(
                narrative_type=VisionFinanceNarrativeType.tax_summary,
            ),
            [first, conflict],
            tenant_id="tenant-finance",
            actor="reviewer@documind.test",
        )


def test_workflow_status_summaries_do_not_hide_duplicate_uploads():
    first = _invoice("vision-doc-workflow-a", "INV-009", "USD 90.00")
    second = _invoice("vision-doc-workflow-b", "inv 009", "USD 90.00")
    narrative = VerifiedFinanceNarrativeService().generate(
        VisionFinanceNarrativeRequest(
            narrative_type=VisionFinanceNarrativeType.approval_status_summary,
        ),
        [second, first],
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )

    assert narrative.document_ids == ["vision-doc-workflow-a", "vision-doc-workflow-b"]
    assert narrative.excluded_document_ids == []
    assert "APPROVED: 2" in narrative.narrative


def test_persisted_identity_distinguishes_duplicate_exclusion_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    service = VisionAIService()
    approved_source = _invoice("vision-doc-z-approved", "INV-006", "USD 60.00")
    completed_duplicate = _invoice(
        "vision-doc-a-completed",
        "inv 006",
        "USD 60.00",
        status=VisionProcessingStatus.completed,
    )
    documents = [approved_source, completed_duplicate]
    service.documents = {document.document_id: document for document in documents}
    monkeypatch.setattr(service, "list_documents", lambda tenant_id=None: documents)
    monkeypatch.setattr(service.persistence, "list_narratives", lambda tenant_id: [])
    monkeypatch.setattr(service.persistence, "save_narrative", lambda narrative: None)
    monkeypatch.setattr(service.persistence, "save_document", lambda document: None)

    retained_only = service.generate_finance_narrative(
        VisionFinanceNarrativeRequest(
            narrative_type=VisionFinanceNarrativeType.monthly_invoice_summary,
            document_ids=[approved_source.document_id],
        ),
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )
    with_duplicate = service.generate_finance_narrative(
        VisionFinanceNarrativeRequest(
            narrative_type=VisionFinanceNarrativeType.monthly_invoice_summary,
            document_ids=[approved_source.document_id, completed_duplicate.document_id],
        ),
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )
    repeated = service.generate_finance_narrative(
        VisionFinanceNarrativeRequest(
            narrative_type=VisionFinanceNarrativeType.monthly_invoice_summary,
            document_ids=[completed_duplicate.document_id, approved_source.document_id],
        ),
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )

    assert retained_only.identity_hash != with_duplicate.identity_hash
    assert retained_only.narrative_id != with_duplicate.narrative_id
    assert with_duplicate.request_scope["duplicate_invoice_exclusions"]
    assert repeated.narrative_id == with_duplicate.narrative_id

    version_request = VisionFinanceNarrativeRequest.model_validate(
        {**with_duplicate.request_scope, "generate_new_version": True}
    )
    versioned = service.generate_finance_narrative(
        version_request,
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )
    assert version_request.document_ids == sorted(
        [approved_source.document_id, completed_duplicate.document_id]
    )
    assert versioned.version == 2
    assert versioned.supersedes == with_duplicate.narrative_id
    assert versioned.identity_hash == with_duplicate.identity_hash
    assert versioned.excluded_document_ids == [completed_duplicate.document_id]


def test_reprocessing_an_excluded_duplicate_marks_narrative_and_report_stale(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    service = VisionAIService()
    retained = _invoice("vision-doc-retained", "INV-010", "USD 10.00")
    excluded = _invoice(
        "vision-doc-excluded",
        "inv 010",
        "USD 10.00",
        status=VisionProcessingStatus.completed,
    )
    monkeypatch.setattr(service.persistence, "save_document", lambda document: None)
    monkeypatch.setattr(service.persistence, "save_narrative", lambda narrative: None)
    monkeypatch.setattr(service.persistence, "save_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(service.persistence, "list_documents", lambda tenant_id: [])
    monkeypatch.setattr(service.persistence, "list_narratives", lambda tenant_id: [])
    monkeypatch.setattr(service.persistence, "list_reports", lambda tenant_id: [])
    service.documents = {
        retained.document_id: retained,
        excluded.document_id: excluded,
    }
    service._persist_document(retained)
    service._persist_document(excluded)

    narrative = service.generate_finance_narrative(
        VisionFinanceNarrativeRequest(
            narrative_type=VisionFinanceNarrativeType.monthly_invoice_summary,
            document_ids=[retained.document_id, excluded.document_id],
        ),
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )
    report = service.generate_report(
        VisionReportRequest(
            report_type=VisionReportType.finance_narrative,
            narrative_id=narrative.narrative_id,
        ),
        tenant_id="tenant-finance",
        actor="reviewer@documind.test",
    )

    service.queue_reprocess(
        excluded.document_id,
        b"Invoice: INV-010\nVendor: Verified Vendor\nTotal Amount: USD 10.00",
        actor="reviewer@documind.test",
        tenant_id="tenant-finance",
    )

    assert service.get_finance_narrative(narrative.narrative_id, "tenant-finance").status == (
        "stale"
    )
    assert service.get_report(report.report_id, "tenant-finance").status == "stale"
    assert narrative.narrative_id in excluded.metadata["stale_derived_artifacts"]["narrative_ids"]
    assert report.report_id in excluded.metadata["stale_derived_artifacts"]["report_ids"]
