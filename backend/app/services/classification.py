from pathlib import Path
from typing import ClassVar

from app.core.config import settings
from app.models.schemas import DocumentType


class DocumentClassificationService:
    keyword_map: ClassVar[dict[DocumentType, tuple[str, ...]]] = {
        DocumentType.invoice: (
            "invoice",
            "bill to",
            "invoice number",
            "amount due",
            "tax invoice",
            "inv-",
        ),
        DocumentType.purchase_order: (
            "purchase order",
            "purchase_order",
            "po number",
            "po-",
            "ship to",
            "ordered by",
        ),
        DocumentType.contract: (
            "agreement",
            "contract",
            "master services",
            "term",
            "termination",
            "governing law",
        ),
        DocumentType.bank_change_request: (
            "bank account change",
            "vendor bank update",
            "new bank account",
            "old bank account",
            "beneficiary bank",
            "swift",
            "ifsc",
        ),
        DocumentType.bank_statement: (
            "statement",
            "account number",
            "transaction",
            "opening balance",
        ),
        DocumentType.compliance_policy: ("policy", "compliance", "regulation", "audit", "control"),
        DocumentType.medical_report: ("diagnosis", "patient", "clinical", "lab result"),
        DocumentType.legal_document: ("plaintiff", "defendant", "court", "legal notice"),
        DocumentType.resume: ("experience", "education", "skills", "curriculum vitae"),
        DocumentType.email: (
            "from:",
            "to:",
            "subject:",
            "sent:",
            "email",
            "urgent wire",
            "bank account change",
        ),
    }

    def __init__(self, rvl_cdip_path: str | None = None) -> None:
        raw_path = rvl_cdip_path or settings.rvl_cdip_path
        self.rvl_cdip_path = Path(raw_path).expanduser() if raw_path else None
        self.rvl_cdip_available = bool(self.rvl_cdip_path and self.rvl_cdip_path.exists())

    def classify(self, text: str, filename: str = "") -> tuple[DocumentType, float]:
        haystack = f"{filename}\n{text}".lower()
        scored: list[tuple[int, DocumentType]] = []
        for doc_type, keywords in self.keyword_map.items():
            score = sum(1 for keyword in keywords if keyword in haystack)
            scored.append((score, doc_type))
        score, doc_type = max(scored, key=lambda item: item[0])
        if score == 0:
            return DocumentType.unknown, 0.45
        rvl_bonus = (
            0.02
            if self.rvl_cdip_available
            and doc_type
            in {
                DocumentType.invoice,
                DocumentType.email,
                DocumentType.legal_document,
                DocumentType.compliance_policy,
            }
            else 0.0
        )
        return doc_type, min(0.96, 0.58 + score * 0.11 + rvl_bonus)
