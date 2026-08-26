from app.legacy.pipeline import DocumentIntelligencePipeline
from app.models.schemas import TextAnalysisRequest


def test_pipeline_extracts_and_scores_invoice_case():
    service = DocumentIntelligencePipeline()
    response = service.process_text(
        TextAnalysisRequest(
            filename="invoice-1001.txt",
            text=(
                "Invoice INV-1001\nVendor: Acme Finance Ltd\nAmount Due USD 12,500.00\n"
                "Payment terms Net 30\nNew bank account requested for urgent wire."
            ),
        )
    )

    assert response.case_id.startswith("case-")
    assert response.documents[0].document_type == "invoice"
    assert response.fraud_probability > 0
    assert response.workflow.approval_required is True


def test_rag_returns_citation_after_indexing():
    service = DocumentIntelligencePipeline()
    response = service.process_text(
        TextAnalysisRequest(
            filename="contract.txt",
            text="Master Services Agreement with confidentiality, termination, and governing law in New York.",
        )
    )

    from app.legacy.rag import RAGEngine

    answer = RAGEngine(service.retrieval).answer("What law governs the contract?", response.case_id)
    assert answer.citations
    assert "contract.txt" in answer.citations[0].source_document

