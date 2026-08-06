from __future__ import annotations

from app.core.security import utc_now_iso
from app.models.vision import (
    VisionDocumentRecord,
    VisionDocumentType,
    VisionFieldCompleteness,
    VisionProcessingStatus,
    VisionReportRequest,
    VisionReportType,
)
from app.services.vision.reporting import VisionReportService


def test_document_report_uses_the_shared_seven_field_completeness_counts():
    now = utc_now_iso()
    document = VisionDocumentRecord(
        document_id="vision-doc-count-report",
        case_id="vision-case-count-report",
        filename="counts.png",
        file_hash="c" * 64,
        status=VisionProcessingStatus.review_required,
        document_type=VisionDocumentType.business_form,
        created_at=now,
        updated_at=now,
        field_completeness=VisionFieldCompleteness(
            schema_available=True,
            document_type=VisionDocumentType.business_form,
            required_expected_count=3,
            required_matched_count=1,
            required_missing_count=2,
            optional_extracted_count=1,
            total_extracted_count=2,
            duplicates_rejected_count=1,
            disputed_count=0,
        ),
    )

    report = VisionReportService().generate(
        VisionReportRequest(
            report_type=VisionReportType.document_intelligence,
            document_id=document.document_id,
        ),
        [document],
        [],
        tenant_id="tenant-count-report",
        actor="qa@example.com",
    )

    payload = report.content["documents"][0]
    assert payload["field_completeness"] == document.field_completeness.model_dump(
        mode="json"
    )
    assert "required expected 3" in payload["summary"]
    assert "required matched 1" in payload["summary"]
    assert "required missing 2" in payload["summary"]
    assert "optional extracted 1" in payload["summary"]
    assert "total extracted 2" in payload["summary"]
    assert "duplicates rejected 1" in payload["summary"]
    assert "disputed 0" in payload["summary"]
