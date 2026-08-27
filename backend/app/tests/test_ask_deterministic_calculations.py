from __future__ import annotations

from uuid import uuid4

from app.core.security import hash_document, utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionAskCalculationOperation,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionProcessingStatus,
)
from app.services.vision.query_router import VisionQueryRouter
from app.services.vision.service import VisionAIService


def _field(
    document_id: str,
    name: str,
    value: str,
    *,
    status: ReviewerStatus = ReviewerStatus.accepted,
) -> ExtractedVisionField:
    return ExtractedVisionField(
        name=name,
        value=value,
        normalized_value=value,
        confidence=0.96,
        calibrated=True,
        reviewer_status=status,
        validation_status="valid",
        extractor="deterministic-schema-extractor-v3",
        evidence=VisionEvidence(
            document_id=document_id,
            page_number=1,
            source_text=f"{name}: {value}",
            model_confidence=0.96,
            model_name="deterministic-schema-extractor",
            model_version="v3",
            processing_timestamp=utc_now_iso(),
            reviewer_status=status,
        ),
    )


def _document(
    document_type: VisionDocumentType,
    fields: dict[str, str],
    *,
    status: VisionProcessingStatus = VisionProcessingStatus.approved,
    unverified: set[str] | None = None,
) -> VisionDocumentRecord:
    document_id = f"vision-doc-calc-{uuid4().hex[:10]}"
    text = "\n".join(f"{name}: {value}" for name, value in fields.items())
    now = utc_now_iso()
    return VisionDocumentRecord(
        document_id=document_id,
        case_id=f"case-{uuid4().hex[:8]}",
        filename=f"{document_type.value}-{document_id[-4:]}.txt",
        tenant_id="tenant-calculation-tests",
        status=status,
        document_type=document_type,
        file_hash=hash_document(text.encode()),
        parsed_text=text,
        created_at=now,
        updated_at=now,
        extracted_fields=[
            _field(
                document_id,
                name,
                value,
                status=(
                    ReviewerStatus.unverified
                    if name in (unverified or set())
                    else ReviewerStatus.accepted
                ),
            )
            for name, value in fields.items()
        ],
    )


def _calculate(question: str, documents: list[VisionDocumentRecord]):
    return VisionAIService()._answer_arithmetic(
        question,
        documents,
        top_k=5,
        tenant_id="tenant-calculation-tests",
    )


def test_router_recognizes_all_required_deterministic_calculation_shapes():
    router = VisionQueryRouter()

    for question in (
        "Sum invoice totals",
        "How many verified invoices are there?",
        "What is the difference between invoice and purchase order?",
        "What is the invoice total minus the PO total?",
        "Calculate the percentage variance between invoice and PO",
        "By what percent does the invoice differ from the PO?",
        "How many days between invoice date and due date?",
        "What is the combined total of all invoices?",
    ):
        assert router.route(question) == "arithmetic_question"


def test_sum_returns_reviewed_decimal_inputs_result_and_citations():
    invoices = [
        _document(
            VisionDocumentType.invoice,
            {"invoice_number": "INV-1", "total_amount": "USD 110.25", "currency": "USD"},
        ),
        _document(
            VisionDocumentType.invoice,
            {"invoice_number": "INV-2", "total_amount": "200.25", "currency": "USD"},
        ),
    ]

    response = _calculate("Sum invoice totals", invoices)

    assert response.answer_status == "supported"
    assert response.answer == "USD 310.50"
    assert len(response.citations) == 2
    assert response.calculation_trace is not None
    assert response.calculation_trace.operation == VisionAskCalculationOperation.sum
    assert [item.normalized_value for item in response.calculation_trace.inputs] == [
        "110.25",
        "200.25",
    ]
    assert response.calculation_trace.result == "310.50"
    assert response.calculation_trace.currency == "USD"


def test_count_verified_invoices_excludes_unreviewed_or_unapproved_records():
    documents = [
        _document(VisionDocumentType.invoice, {"invoice_number": "INV-1"}),
        _document(
            VisionDocumentType.invoice,
            {"invoice_number": "INV-2"},
            status=VisionProcessingStatus.completed,
        ),
        _document(
            VisionDocumentType.invoice,
            {"invoice_number": "INV-3"},
            unverified={"invoice_number"},
        ),
        _document(
            VisionDocumentType.invoice,
            {"invoice_number": "INV-4"},
            status=VisionProcessingStatus.review_required,
        ),
    ]

    response = _calculate("Count verified invoices", documents)

    assert response.answer == "2 verified invoice(s)."
    assert len(response.citations) == 2
    assert response.calculation_trace is not None
    assert (
        response.calculation_trace.operation
        == VisionAskCalculationOperation.count_verified_invoices
    )
    assert response.calculation_trace.result == "2"
    assert {item.role for item in response.calculation_trace.inputs} == {"counted_invoice"}


def test_invoice_po_difference_and_percentage_variance_are_decimal_safe():
    invoice = _document(
        VisionDocumentType.invoice,
        {
            "invoice_number": "INV-5",
            "purchase_order_number": "PO-5",
            "total_amount": "110.00",
            "currency": "USD",
        },
    )
    purchase_order = _document(
        VisionDocumentType.purchase_order,
        {"purchase_order_number": "PO-5", "total_amount": "100.00", "currency": "USD"},
    )

    difference = _calculate(
        "What is the difference between invoice and purchase order?",
        [invoice, purchase_order],
    )
    variance = _calculate(
        "Calculate the percentage variance between invoice and PO",
        [invoice, purchase_order],
    )

    assert difference.calculation_trace is not None
    assert (
        difference.calculation_trace.operation
        == VisionAskCalculationOperation.invoice_po_difference
    )
    assert difference.calculation_trace.result == "10.00"
    assert difference.calculation_trace.currency == "USD"
    assert len(difference.citations) == 2
    assert variance.calculation_trace is not None
    assert variance.calculation_trace.operation == VisionAskCalculationOperation.percentage_variance
    assert variance.calculation_trace.result == "10.00"
    assert variance.calculation_trace.unit == "percent"


def test_date_interval_uses_reviewed_calendar_dates():
    invoice = _document(
        VisionDocumentType.invoice,
        {
            "invoice_number": "INV-6",
            "invoice_date": "2026-08-01",
            "due_date": "2026-08-31",
        },
    )

    response = _calculate("How many days between invoice date and due date?", [invoice])

    assert response.answer_status == "supported"
    assert response.calculation_trace is not None
    assert response.calculation_trace.operation == VisionAskCalculationOperation.date_interval
    assert response.calculation_trace.result == "30"
    assert response.calculation_trace.unit == "days"
    assert [item.normalized_value for item in response.calculation_trace.inputs] == [
        "2026-08-01",
        "2026-08-31",
    ]
    assert len(response.citations) == 2


def test_currency_guardrail_returns_inputs_but_no_calculated_result():
    invoices = [
        _document(
            VisionDocumentType.invoice,
            {"invoice_number": "INV-USD", "total_amount": "100.00", "currency": "USD"},
        ),
        _document(
            VisionDocumentType.invoice,
            {"invoice_number": "INV-EUR", "total_amount": "100.00", "currency": "EUR"},
        ),
    ]

    response = _calculate("Sum invoice totals", invoices)

    assert response.answer_status == "partially_supported"
    assert response.requires_review is True
    assert response.calculation_trace is not None
    assert response.calculation_trace.result is None
    assert response.calculation_trace.result_type == "not_calculated"
    assert {item.currency for item in response.calculation_trace.inputs} == {"USD", "EUR"}
