import re
from datetime import UTC, datetime
from typing import Any

from app.models.schemas import (
    ExtractedFinanceFields,
    FinanceDocument,
    FinanceDocumentType,
    FinanceNarrative,
    FinanceNarrativeType,
    RiskFlag,
    RiskLevel,
)


class FinanceNarrativeService:
    """Grounded finance narrative generator built from extracted fields only."""

    def generate(
        self,
        document: FinanceDocument,
        extraction: ExtractedFinanceFields,
        narrative_type: FinanceNarrativeType = FinanceNarrativeType.executive_summary,
    ) -> FinanceNarrative:
        fields = extraction.corrected_extraction or extraction.fields
        risk_flags = list(extraction.risk_flags)
        narrative = self._compose(document, fields, extraction, narrative_type)
        unsupported_claims = self.unsupported_claims(narrative, fields, document.text)
        if unsupported_claims:
            risk_flags.append(
                RiskFlag(
                    title="Unsupported narrative claims",
                    severity=RiskLevel.high,
                    category="hallucination",
                    description="The narrative contains values that were not grounded in extraction or source text.",
                    recommended_action="Regenerate or request reviewer correction before approval.",
                    source_references=extraction.source_references[:3],
                )
            )
        groundedness = 1 - (len(unsupported_claims) / max(1, len(self._claims(narrative))))
        return FinanceNarrative(
            document_id=document.document_id,
            narrative_type=narrative_type,
            narrative=narrative,
            confidence_score=round(max(0.1, extraction.confidence_score - len(unsupported_claims) * 0.1), 3),
            groundedness_score=round(max(0, groundedness), 3),
            unsupported_claims=unsupported_claims,
            risk_flags=risk_flags,
            source_references=extraction.source_references,
            reviewer_recommendation=self._reviewer_recommendation(document.document_type, risk_flags, extraction.missing_fields),
            created_at=datetime.now(UTC).isoformat(),
        )

    def unsupported_claims(self, narrative: str, fields: dict[str, Any], source_text: str) -> list[str]:
        supported_text = " ".join([source_text, self._flatten(fields)]).lower()
        unsupported = []
        for claim in self._claims(narrative):
            raw = claim.group(0)
            normalized = raw.lower().replace(",", "")
            supported = supported_text.replace(",", "")
            if normalized not in supported:
                unsupported.append(raw)
        if "approved for payment" in narrative.lower() or "safe to pay" in narrative.lower():
            unsupported.append("payment_approval_claim")
        return sorted(set(unsupported))

    def _compose(
        self,
        document: FinanceDocument,
        fields: dict[str, Any],
        extraction: ExtractedFinanceFields,
        narrative_type: FinanceNarrativeType,
    ) -> str:
        if narrative_type == FinanceNarrativeType.payment_summary:
            return self._payment_summary(document, fields, extraction)
        if narrative_type == FinanceNarrativeType.risk_summary:
            return self._risk_summary(document, fields, extraction)
        if narrative_type == FinanceNarrativeType.compliance_summary:
            return self._compliance_summary(document, fields, extraction)
        if narrative_type == FinanceNarrativeType.exception_summary:
            return self._exception_summary(document, fields, extraction)
        if narrative_type == FinanceNarrativeType.reviewer_notes_draft:
            return self._reviewer_notes(document, fields, extraction)
        return self._executive_summary(document, fields, extraction)

    def _executive_summary(
        self,
        document: FinanceDocument,
        fields: dict[str, Any],
        extraction: ExtractedFinanceFields,
    ) -> str:
        if document.document_type == FinanceDocumentType.invoice:
            return (
                f"Invoice {self._value(fields, 'invoice_number')} from {self._value(fields, 'vendor_name')} "
                f"dated {self._value(fields, 'invoice_date')} requests payment of "
                f"{self._value(fields, 'total_amount')} against PO {self._value(fields, 'po_number')}. "
                f"Payment terms are {self._value(fields, 'payment_terms')}. "
                f"{self._risk_sentence(extraction)} {self._evidence_sentence(extraction)}"
            )
        if document.document_type == FinanceDocumentType.purchase_order:
            return (
                f"Purchase order {self._value(fields, 'po_number')} from {self._value(fields, 'buyer_name')} "
                f"to {self._value(fields, 'supplier_name')} has total value {self._value(fields, 'total_value')} "
                f"and approval status {self._value(fields, 'approval_status')}. "
                f"{self._risk_sentence(extraction)} {self._evidence_sentence(extraction)}"
            )
        if document.document_type == FinanceDocumentType.contract:
            return (
                f"Finance contract between {self._value(fields, 'parties')} runs from "
                f"{self._value(fields, 'contract_start_date')} to {self._value(fields, 'contract_end_date')}. "
                f"Payment terms are {self._value(fields, 'payment_terms')}. "
                f"Missing clauses include {self._value(fields, 'missing_clauses')}. "
                f"{self._risk_sentence(extraction)} {self._evidence_sentence(extraction)}"
            )
        if document.document_type == FinanceDocumentType.bank_change_request:
            return (
                f"Bank-change request for {self._value(fields, 'vendor_name')} asks to move payment details "
                f"from {self._value(fields, 'old_bank_account')} to {self._value(fields, 'new_bank_account')} "
                f"at {self._value(fields, 'bank_name')}. The masked account is "
                f"{self._value(fields, 'account_number_masked')}. This requires human review before payment action. "
                f"{self._risk_sentence(extraction)} {self._evidence_sentence(extraction)}"
            )
        return f"{document.file_name} was processed as an unknown finance document. {self._evidence_sentence(extraction)}"

    def _payment_summary(self, document: FinanceDocument, fields: dict[str, Any], extraction: ExtractedFinanceFields) -> str:
        if document.document_type == FinanceDocumentType.bank_change_request:
            return (
                f"Bank-change payment hold: vendor {self._value(fields, 'vendor_name')} requested a bank update "
                f"from {self._value(fields, 'old_bank_account')} to {self._value(fields, 'new_bank_account')} "
                f"with masked account {self._value(fields, 'account_number_masked')} at "
                f"{self._value(fields, 'bank_name')}. Request date is {self._value(fields, 'request_date')} and "
                f"requester contact is {self._value(fields, 'requester_email_or_contact')}. Payment must remain on hold "
                f"until independent human verification is recorded. {self._risk_sentence(extraction)} "
                f"{self._evidence_sentence(extraction)}"
            )
        return (
            f"Payment details: vendor or supplier {self._value(fields, 'vendor_name', 'supplier_name')}, "
            f"amount {self._value(fields, 'total_amount', 'total_value')}, currency {self._value(fields, 'currency')}, "
            f"terms {self._value(fields, 'payment_terms')}, PO {self._value(fields, 'po_number')}. "
            "Payment must remain pending until reviewer action is recorded. "
            f"{self._risk_sentence(extraction)}"
        )

    def _risk_summary(self, document: FinanceDocument, fields: dict[str, Any], extraction: ExtractedFinanceFields) -> str:
        flag_text = "; ".join(flag.title for flag in extraction.risk_flags) or "No high-risk finance flags detected"
        return (
            f"Risk summary for {document.file_name}: {flag_text}. "
            f"Missing fields: {self._value({'missing_fields': extraction.missing_fields}, 'missing_fields')}. "
            f"Recommended action: {self._reviewer_recommendation(document.document_type, extraction.risk_flags, extraction.missing_fields)}"
        )

    def _compliance_summary(self, document: FinanceDocument, fields: dict[str, Any], extraction: ExtractedFinanceFields) -> str:
        return (
            f"Compliance summary: document type {document.document_type.value}, extraction confidence "
            f"{round(extraction.confidence_score * 100)}%, missing fields "
            f"{self._value({'missing_fields': extraction.missing_fields}, 'missing_fields')}. "
            f"Important claims cite {len(extraction.source_references)} source reference(s)."
        )

    def _exception_summary(self, document: FinanceDocument, fields: dict[str, Any], extraction: ExtractedFinanceFields) -> str:
        exceptions = [flag.description for flag in extraction.risk_flags] + [
            f"Missing required field: {field}" for field in extraction.missing_fields
        ]
        return "Exception summary: " + (" ".join(exceptions) if exceptions else "No exceptions detected.")

    def _reviewer_notes(self, document: FinanceDocument, fields: dict[str, Any], extraction: ExtractedFinanceFields) -> str:
        return (
            f"Reviewer notes draft: verify {document.document_type.value} fields, inspect "
            f"{len(extraction.source_references)} source reference(s), and resolve "
            f"{len(extraction.risk_flags)} risk flag(s). "
            f"Suggested reviewer action: {self._reviewer_recommendation(document.document_type, extraction.risk_flags, extraction.missing_fields)}"
        )

    def _reviewer_recommendation(
        self,
        document_type: FinanceDocumentType,
        risk_flags: list[RiskFlag],
        missing_fields: list[str],
    ) -> str:
        if document_type == FinanceDocumentType.bank_change_request:
            return "Escalate bank-change request and verify through a trusted out-of-band process."
        if any(flag.severity in {RiskLevel.high, RiskLevel.critical} for flag in risk_flags):
            return "Send to human review and hold downstream payment until reviewed."
        if missing_fields:
            return "Request correction for missing fields before approval."
        return "Send to finance reviewer for approval; do not auto-approve payment."

    def _risk_sentence(self, extraction: ExtractedFinanceFields) -> str:
        if not extraction.risk_flags:
            return "No blocking risk flag was generated by the finance extraction service."
        flags = ", ".join(flag.title for flag in extraction.risk_flags[:4])
        return f"Detected risk flags: {flags}."

    def _evidence_sentence(self, extraction: ExtractedFinanceFields) -> str:
        count = len(extraction.source_references)
        return f"The narrative is grounded in {count} source reference(s)."

    def _value(self, fields: dict[str, Any], *names: str) -> str:
        for name in names:
            value = fields.get(name)
            if value not in (None, "", [], {}):
                if isinstance(value, list):
                    return ", ".join(str(item) for item in value[:4])
                if isinstance(value, dict):
                    return ", ".join(f"{key}: {item}" for key, item in value.items() if item) or "not extracted"
                return str(value)
        return "not extracted"

    def _flatten(self, value: Any) -> str:
        if isinstance(value, dict):
            return " ".join(f"{key} {self._flatten(item)}" for key, item in value.items())
        if isinstance(value, list):
            return " ".join(self._flatten(item) for item in value)
        return str(value)

    def _claims(self, narrative: str) -> list[re.Match]:
        claim_pattern = re.compile(
            r"\b(?:INR|USD|EUR|GBP)\s?[\d,]+(?:\.\d{2})?\b"
            r"|\b[A-Z]{2,}-?\d{3,}\b"
            r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b"
            r"|\*{4}\d{4}\b",
            re.IGNORECASE,
        )
        return list(claim_pattern.finditer(narrative))
