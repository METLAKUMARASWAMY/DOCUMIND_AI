from app.models.schemas import TextAnalysisRequest
from app.services.pipeline import DocumentIntelligencePipeline


def test_customer_is_not_classified_as_vendor_and_vendor_is_clean():
    pipeline = DocumentIntelligencePipeline()
    analysis = pipeline.process_text(
        TextAnalysisRequest(
            filename="contract.txt",
            text=(
                "Master Services Agreement\n"
                "Customer: Northwind Bank Operations\n"
                "Vendor: Aster Analytics Inc\n"
                "Services: vendor will provide analytics support for enterprise finance operations.\n"
                "Termination: either party may terminate after thirty days written notice.\n"
                "Confidentiality: both parties protect confidential information.\n"
                "Governing Law: New York\n"
                "Audit Rights: customer may audit vendor controls.\n"
            ),
        )
    )
    fields = {
        (field.name, field.value)
        for document in analysis.documents
        for field in document.extracted_fields
    }

    assert ("company", "Northwind Bank Operations") in fields
    assert ("vendor", "Aster Analytics Inc") in fields
    assert ("vendor", "Northwind Bank Operations") not in fields
    assert all("will provide analytics support" not in value for name, value in fields if name == "vendor")


def test_negated_indemnity_is_missing_clause_not_extracted_clause():
    pipeline = DocumentIntelligencePipeline()
    analysis = pipeline.process_text(
        TextAnalysisRequest(
            filename="contract.txt",
            text=(
                "Master Services Agreement\n"
                "Customer: Northwind Bank Operations\n"
                "Vendor: Aster Analytics Inc\n"
                "This contract does not include indemnity or hold harmless language.\n"
                "Termination: either party may terminate after thirty days written notice.\n"
                "Confidentiality: both parties protect confidential information.\n"
                "Governing Law: New York\n"
                "Audit Rights: customer may audit vendor controls.\n"
            ),
        )
    )
    field_names = [
        field.name
        for document in analysis.documents
        for field in document.extracted_fields
    ]
    finding_titles = {finding.title for finding in analysis.compliance_findings}
    evidence_reasons = [
        evidence.reasoning
        for finding in analysis.compliance_findings
        for evidence in finding.evidence
    ]

    assert "indemnity_clause" not in field_names
    assert "Missing indemnity requirement" in finding_titles
    assert "Required indemnity language was not found in extracted text." in evidence_reasons
