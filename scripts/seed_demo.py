import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.models.schemas import TextAnalysisRequest
from app.services.pipeline import DocumentIntelligencePipeline


def main() -> None:
    service = DocumentIntelligencePipeline()
    analysis = service.process_text(
        TextAnalysisRequest(
            filename="demo-enterprise-case.txt",
            text=(
                "Master Services Agreement. Vendor: Aster Analytics Inc. "
                "Invoice INV-4409. Purchase Order PO-4409. Amount Due USD 42,500.00. "
                "Payment terms Net 30. Confidentiality, termination, and governing law apply. "
                "Manual override was applied without approval. New bank account requested for urgent wire."
            ),
        )
    )
    print(analysis.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
