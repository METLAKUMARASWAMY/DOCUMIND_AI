import re
from datetime import UTC, datetime
from typing import Any, ClassVar

from app.models.schemas import (
    ExtractedFinanceFields,
    FinanceDocument,
    FinanceDocumentType,
    RiskFlag,
    RiskLevel,
    SourceReference,
)


class FinanceExtractionService:
    """Rule-based finance extraction with source traceability and safe defaults."""

    required_fields: ClassVar[dict[FinanceDocumentType, list[str]]] = {
        FinanceDocumentType.invoice: [
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "due_date",
            "po_number",
            "line_items",
            "subtotal",
            "tax_amount",
            "total_amount",
            "currency",
            "payment_terms",
            "bank_account_details",
        ],
        FinanceDocumentType.purchase_order: [
            "buyer_name",
            "supplier_name",
            "po_number",
            "po_date",
            "delivery_date",
            "line_items",
            "quantity",
            "unit_price",
            "total_value",
            "currency",
            "approval_status",
            "payment_terms",
        ],
        FinanceDocumentType.contract: [
            "parties",
            "contract_start_date",
            "contract_end_date",
            "payment_terms",
            "renewal_clause",
            "termination_clause",
            "penalty_clause",
            "obligations",
        ],
        FinanceDocumentType.bank_change_request: [
            "vendor_name",
            "old_bank_account",
            "new_bank_account",
            "bank_name",
            "account_number_masked",
            "IFSC_or_SWIFT",
            "request_date",
            "requester_email_or_contact",
            "mismatch_indicators",
            "fraud_risk_level",
            "recommended_human_action",
        ],
        FinanceDocumentType.unknown: [],
    }

    prompt_injection_terms = (
        "ignore previous instructions",
        "ignore all instructions",
        "disregard system",
        "reveal prompt",
        "approve payment automatically",
        "bypass review",
        "do not tell the reviewer",
    )

    patterns: ClassVar[dict[str, re.Pattern[str]]] = {
        "vendor_name": re.compile(
            r"(?:vendor|supplier|seller|beneficiary|payee)\s*[:#-]\s*([^\n\r:]{2,100})",
            re.IGNORECASE,
        ),
        "buyer_name": re.compile(
            r"(?:buyer|bill to|customer|ordered by)\s*[:#-]\s*([^\n\r:]{2,100})", re.IGNORECASE
        ),
        "invoice_number": re.compile(
            r"(?:invoice(?: number| no\.?)?|inv)\s*[:#-]\s*([A-Z0-9-]{3,40})", re.IGNORECASE
        ),
        "po_number": re.compile(
            r"(?:purchase order(?: number| no\.?)?|po(?: number| no\.?)?|po)\s*[:#-]\s*([A-Z0-9-]{3,40})",
            re.IGNORECASE,
        ),
        "invoice_date": re.compile(
            r"(?:invoice date|date)\s*[:#-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
        "due_date": re.compile(
            r"(?:due date|payment due)\s*[:#-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
        "po_date": re.compile(
            r"(?:po date|purchase order date|date)\s*[:#-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
        "delivery_date": re.compile(
            r"(?:delivery date|required by|deliver by)\s*[:#-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
        "contract_start_date": re.compile(
            r"(?:start date|effective date|contract start)\s*[:#-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
        "contract_end_date": re.compile(
            r"(?:end date|expiry date|expiration date|contract end)\s*[:#-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
        "subtotal": re.compile(
            r"(?:subtotal|sub total)\s*[:#-]\s*((?:INR|USD|EUR|GBP|\$)?\s?[\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
        "tax_amount": re.compile(
            r"(?:tax|gst|vat)\s*[:#-]\s*((?:INR|USD|EUR|GBP|\$)?\s?[\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
        "total_amount": re.compile(
            r"(?:total amount|amount due|invoice total|grand total|total)\s*[:#-]\s*((?:INR|USD|EUR|GBP|\$)?\s?[\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
        "total_value": re.compile(
            r"(?:total value|po value|approved amount|total)\s*[:#-]\s*((?:INR|USD|EUR|GBP|\$)?\s?[\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
        "unit_price": re.compile(
            r"(?:unit price|rate)\s*[:#-]\s*((?:INR|USD|EUR|GBP|\$)?\s?[\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
        "quantity": re.compile(r"(?:quantity|qty)\s*[:#-]\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
        "payment_terms": re.compile(
            r"(?:payment terms|terms)\s*[:#-]\s*([^\n\r]{2,100})", re.IGNORECASE
        ),
        "approval_status": re.compile(
            r"(?:approval status|approved by|status)\s*[:#-]\s*([^\n\r]{2,80})", re.IGNORECASE
        ),
        "bank_name": re.compile(
            r"(?:bank name|beneficiary bank)\s*[:#-]\s*([^\n\r:]{2,100})", re.IGNORECASE
        ),
        "old_bank_account": re.compile(
            r"(?:old bank account|old account|existing account)\s*[:#-]\s*([A-Z0-9 -]{6,34})",
            re.IGNORECASE,
        ),
        "new_bank_account": re.compile(
            r"(?:new bank account|new account|updated account|account number)\s*[:#-]\s*([A-Z0-9 -]{6,34})",
            re.IGNORECASE,
        ),
        "IFSC_or_SWIFT": re.compile(
            r"(?:ifsc|swift|bic|routing)\s*[:#-]\s*([A-Z0-9-]{6,34})", re.IGNORECASE
        ),
        "request_date": re.compile(
            r"(?:request date|requested on|date)\s*[:#-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
        "requester_email_or_contact": re.compile(
            r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE
        ),
    }

    def extract(self, document: FinanceDocument) -> ExtractedFinanceFields:
        doc_type = (
            document.document_type
            if document.document_type != FinanceDocumentType.unknown
            else self.detect_type(document)
        )
        document.document_type = doc_type
        fields: dict[str, Any] = {}
        references: list[SourceReference] = []
        field_confidence: dict[str, float] = {}

        if doc_type == FinanceDocumentType.invoice:
            fields, references, field_confidence = self._extract_invoice(document)
        elif doc_type == FinanceDocumentType.purchase_order:
            fields, references, field_confidence = self._extract_purchase_order(document)
        elif doc_type == FinanceDocumentType.contract:
            fields, references, field_confidence = self._extract_contract(document)
        elif doc_type == FinanceDocumentType.bank_change_request:
            fields, references, field_confidence = self._extract_bank_change(document)
        else:
            fields = {"raw_text_preview": document.text[:420]}

        required = self.required_fields.get(doc_type, [])
        missing_fields = [field for field in required if self._is_missing(fields.get(field))]
        risk_flags = self._risk_flags(document, doc_type, fields, missing_fields, references)
        fields["field_confidence"] = field_confidence
        fields["document_type"] = doc_type.value
        fields["risk_level"] = self._risk_level(risk_flags).value
        confidence = self._confidence(required, missing_fields, field_confidence, risk_flags)
        return ExtractedFinanceFields(
            document_id=document.document_id,
            document_type=doc_type,
            fields=fields,
            missing_fields=missing_fields,
            risk_flags=risk_flags,
            confidence_score=confidence,
            source_references=references,
            original_extraction={
                key: value for key, value in fields.items() if key != "field_confidence"
            },
            created_at=self._now(),
        )

    def detect_type(self, document: FinanceDocument) -> FinanceDocumentType:
        text = f"{document.file_name}\n{document.text}".lower()
        scores = {
            FinanceDocumentType.bank_change_request: sum(
                term in text
                for term in (
                    "bank account change",
                    "vendor bank update",
                    "bank update request",
                    "new bank account",
                    "old bank account",
                    "same day wire",
                    "do not call the old",
                )
            ),
            FinanceDocumentType.purchase_order: sum(
                term in text
                for term in ("purchase order", "po number", "approved amount", "ship to")
            ),
            FinanceDocumentType.invoice: sum(
                term in text for term in ("invoice", "amount due", "invoice number", "tax")
            ),
            FinanceDocumentType.contract: sum(
                term in text
                for term in ("agreement", "contract", "termination", "renewal", "obligations")
            ),
        }
        doc_type, score = max(scores.items(), key=lambda item: item[1])
        return doc_type if score else FinanceDocumentType.unknown

    def _extract_invoice(
        self, document: FinanceDocument
    ) -> tuple[dict[str, Any], list[SourceReference], dict[str, float]]:
        names = [
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "due_date",
            "po_number",
            "subtotal",
            "tax_amount",
            "total_amount",
            "payment_terms",
            "bank_name",
            "IFSC_or_SWIFT",
        ]
        fields, refs, confidence = self._extract_named(document, names)
        fields["currency"] = self._currency(document.text)
        fields["line_items"] = self._line_items(document)
        fields["bank_account_details"] = self._bank_details(document)
        self._add_confidence(confidence, "currency", fields["currency"])
        self._add_confidence(confidence, "line_items", fields["line_items"], 0.72)
        self._add_confidence(
            confidence, "bank_account_details", fields["bank_account_details"], 0.68
        )
        return fields, refs, confidence

    def _extract_purchase_order(
        self, document: FinanceDocument
    ) -> tuple[dict[str, Any], list[SourceReference], dict[str, float]]:
        fields, refs, confidence = self._extract_named(
            document,
            [
                "buyer_name",
                "vendor_name",
                "po_number",
                "po_date",
                "delivery_date",
                "quantity",
                "unit_price",
                "total_value",
                "payment_terms",
                "approval_status",
            ],
        )
        if fields.get("vendor_name"):
            fields["supplier_name"] = fields["vendor_name"]
            confidence["supplier_name"] = confidence.get("vendor_name", 0.78)
        fields["currency"] = self._currency(document.text)
        fields["line_items"] = self._line_items(document)
        self._add_confidence(confidence, "currency", fields["currency"])
        self._add_confidence(confidence, "line_items", fields["line_items"], 0.72)
        return fields, refs, confidence

    def _extract_contract(
        self, document: FinanceDocument
    ) -> tuple[dict[str, Any], list[SourceReference], dict[str, float]]:
        fields, refs, confidence = self._extract_named(
            document,
            [
                "buyer_name",
                "vendor_name",
                "contract_start_date",
                "contract_end_date",
                "payment_terms",
            ],
        )
        fields["parties"] = [
            value for value in [fields.get("buyer_name"), fields.get("vendor_name")] if value
        ]
        fields["renewal_clause"] = self._clause(document, "renewal")
        fields["termination_clause"] = self._clause(document, "termination")
        fields["penalty_clause"] = self._clause(document, "penalty")
        fields["obligations"] = self._obligations(document)
        fields["missing_clauses"] = [
            clause
            for clause in ["indemnity", "audit rights", "termination", "confidentiality"]
            if clause not in document.text.lower()
        ]
        fields["risky_clauses"] = self._risky_clauses(document)
        for name in [
            "parties",
            "renewal_clause",
            "termination_clause",
            "penalty_clause",
            "obligations",
        ]:
            self._add_confidence(confidence, name, fields.get(name), 0.7)
        return fields, refs, confidence

    def _extract_bank_change(
        self, document: FinanceDocument
    ) -> tuple[dict[str, Any], list[SourceReference], dict[str, float]]:
        fields, refs, confidence = self._extract_named(
            document,
            [
                "vendor_name",
                "old_bank_account",
                "new_bank_account",
                "bank_name",
                "IFSC_or_SWIFT",
                "request_date",
                "requester_email_or_contact",
            ],
        )
        account = (
            fields.get("new_bank_account")
            or fields.get("old_bank_account")
            or self._first_account(document.text)
        )
        fields["account_number_masked"] = self._mask_account(account)
        fields["mismatch_indicators"] = self._bank_mismatch_indicators(document, fields)
        fields["fraud_risk_level"] = "high" if fields["mismatch_indicators"] else "medium"
        fields["recommended_human_action"] = (
            "Escalate to finance reviewer and verify bank change out of band before payment."
        )
        self._add_confidence(
            confidence, "account_number_masked", fields["account_number_masked"], 0.82
        )
        self._add_confidence(confidence, "mismatch_indicators", fields["mismatch_indicators"], 0.74)
        self._add_confidence(confidence, "fraud_risk_level", fields["fraud_risk_level"], 0.76)
        self._add_confidence(
            confidence, "recommended_human_action", fields["recommended_human_action"], 0.9
        )
        return fields, refs, confidence

    def _extract_named(
        self, document: FinanceDocument, names: list[str]
    ) -> tuple[dict[str, Any], list[SourceReference], dict[str, float]]:
        fields: dict[str, Any] = {}
        refs: list[SourceReference] = []
        confidence: dict[str, float] = {}
        for name in names:
            pattern = self.patterns.get(name)
            if not pattern:
                continue
            match = pattern.search(document.text)
            if not match:
                continue
            value = self._clean_value(match.group(1))
            if name in {"old_bank_account", "new_bank_account"}:
                value = self._mask_account(value)
            if not value:
                continue
            fields[name] = value
            confidence[name] = 0.84
            refs.append(self._reference(document, name, match.group(0)))
        return fields, refs, confidence

    def _risk_flags(
        self,
        document: FinanceDocument,
        doc_type: FinanceDocumentType,
        fields: dict[str, Any],
        missing_fields: list[str],
        references: list[SourceReference],
    ) -> list[RiskFlag]:
        flags: list[RiskFlag] = []
        text_lower = document.text.lower()
        if any(term in text_lower for term in self.prompt_injection_terms):
            flags.append(
                RiskFlag(
                    title="Prompt injection attempt detected",
                    severity=RiskLevel.high,
                    category="guardrail",
                    description="The uploaded document contains instruction-like text that could bypass review.",
                    recommended_action="Ignore embedded instructions and require reviewer validation.",
                    source_references=references[:1],
                )
            )
        if missing_fields:
            flags.append(
                RiskFlag(
                    title="Required finance fields missing",
                    severity=RiskLevel.medium,
                    category="data_quality",
                    description=f"Missing fields: {', '.join(missing_fields[:8])}.",
                    recommended_action="Request correction or upload supporting documents before approval.",
                    source_references=references[:1],
                )
            )
        if doc_type == FinanceDocumentType.bank_change_request:
            flags.append(
                RiskFlag(
                    title="Bank change requires human verification",
                    severity=RiskLevel.high,
                    category="fraud",
                    description="Vendor bank-change requests must not be approved automatically.",
                    recommended_action="Escalate and verify old and new bank details through a trusted channel.",
                    source_references=references[:3],
                )
            )
        if fields.get("bank_account_details") or "new bank account" in text_lower:
            flags.append(
                RiskFlag(
                    title="Sensitive bank details present",
                    severity=RiskLevel.medium,
                    category="privacy",
                    description="Bank account details are masked in reviewer-facing outputs.",
                    recommended_action="Restrict access and avoid payment approval without reviewer action.",
                    source_references=references[:3],
                )
            )
        if any(
            term in text_lower
            for term in ("urgent", "manual override", "same day wire", "off-cycle payment")
        ):
            flags.append(
                RiskFlag(
                    title="Urgent or manual payment language",
                    severity=RiskLevel.high,
                    category="payment_risk",
                    description="Urgent payment or manual override wording increases fraud review priority.",
                    recommended_action="Hold payment until finance and risk review are complete.",
                    source_references=references[:3],
                )
            )
        return flags

    def _bank_details(self, document: FinanceDocument) -> dict[str, Any]:
        bank_name = self._match("bank_name", document.text)
        account = self._first_account(document.text)
        swift = self._match("IFSC_or_SWIFT", document.text)
        result = {}
        if bank_name:
            result["bank_name"] = bank_name
        if account:
            result["account_number_masked"] = self._mask_account(account)
        if swift:
            result["IFSC_or_SWIFT"] = swift
        return result

    def _line_items(self, document: FinanceDocument) -> list[dict[str, Any]]:
        items = []
        for line in document.text.splitlines():
            lower = line.lower()
            if not any(
                term in lower
                for term in ("item", "service", "qty", "quantity", "unit price", "amount")
            ):
                continue
            amount = re.search(r"(?:INR|USD|EUR|GBP|\$)?\s?[\d,]+(?:\.\d{2})?", line, re.IGNORECASE)
            quantity = re.search(
                r"(?:qty|quantity)\s*[:#-]?\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE
            )
            if amount or quantity:
                items.append(
                    {
                        "description": self._clean_value(re.sub(r"\s+", " ", line))[:160],
                        "quantity": quantity.group(1) if quantity else None,
                        "amount": amount.group(0).strip() if amount else None,
                    }
                )
        return items[:8]

    def _bank_mismatch_indicators(
        self, document: FinanceDocument, fields: dict[str, Any]
    ) -> list[str]:
        indicators = []
        lower = document.text.lower()
        if (
            fields.get("old_bank_account")
            and fields.get("new_bank_account")
            and fields["old_bank_account"] != fields["new_bank_account"]
        ):
            indicators.append("old_new_account_changed")
        if "urgent" in lower or "same day" in lower:
            indicators.append("urgent_change_request")
        if fields.get("requester_email_or_contact") and not str(
            fields["requester_email_or_contact"]
        ).lower().endswith((".com", ".in", ".org")):
            indicators.append("unusual_requester_contact")
        if "do not call" in lower or "skip verification" in lower:
            indicators.append("verification_bypass_language")
        return indicators

    def _clause(self, document: FinanceDocument, keyword: str) -> str | None:
        for line in document.text.splitlines():
            if keyword in line.lower():
                return self._clean_value(line)
        return None

    def _obligations(self, document: FinanceDocument) -> list[str]:
        return [
            self._clean_value(line)
            for line in document.text.splitlines()
            if any(
                term in line.lower() for term in ("shall", "must", "responsible for", "obligation")
            )
        ][:8]

    def _risky_clauses(self, document: FinanceDocument) -> list[str]:
        lower = document.text.lower()
        risks = []
        if "penalty" in lower and "cap" not in lower:
            risks.append("uncapped_penalty_language")
        if "auto renew" in lower or "automatic renewal" in lower:
            risks.append("automatic_renewal")
        if "indemnity" not in lower and "indemnification" not in lower:
            risks.append("missing_indemnity")
        return risks

    def _match(self, name: str, text: str) -> str | None:
        pattern = self.patterns.get(name)
        match = pattern.search(text) if pattern else None
        return self._clean_value(match.group(1)) if match else None

    def _reference(
        self, document: FinanceDocument, field_name: str, matched_text: str
    ) -> SourceReference:
        index = max(document.text.lower().find(matched_text.lower()), 0)
        start = max(index - 80, 0)
        end = min(index + len(matched_text) + 120, len(document.text))
        chunk_id = document.chunk_ids[0] if document.chunk_ids else f"{document.document_id}-raw-1"
        evidence_text = self._mask_sensitive_text(document.text[start:end].strip() or matched_text)
        return SourceReference(
            document_id=document.document_id,
            source_document=document.file_name,
            page_number=1,
            chunk_id=chunk_id,
            field_name=field_name,
            evidence_text=evidence_text,
            confidence=0.84,
        )

    def _first_account(self, text: str) -> str | None:
        match = re.search(
            r"(?:account number|account|iban)\s*[:#-]\s*([A-Z0-9 -]{6,34})", text, re.IGNORECASE
        )
        return self._clean_value(match.group(1)) if match else None

    def _mask_account(self, value: str | None) -> str | None:
        if not value:
            return None
        digits = re.sub(r"\D", "", value)
        if len(digits) < 4:
            return "****"
        return f"****{digits[-4:]}"

    def _mask_sensitive_text(self, value: str) -> str:
        labeled = re.sub(
            r"((?:old|new|updated|existing)?\s*(?:bank\s*)?(?:account(?: number)?|iban)\s*[:#-]\s*)([A-Z0-9 -]{6,34})",
            lambda match: f"{match.group(1)}{self._mask_account(match.group(2))}",
            value,
            flags=re.IGNORECASE,
        )
        return re.sub(
            r"\b\d{8,18}\b", lambda match: self._mask_account(match.group(0)) or "****", labeled
        )

    def _currency(self, text: str) -> str | None:
        match = re.search(r"\b(INR|USD|EUR|GBP)\b|[$]", text, re.IGNORECASE)
        if not match:
            return None
        return "USD" if match.group(0) == "$" else match.group(0).upper()

    def _risk_level(self, flags: list[RiskFlag]) -> RiskLevel:
        severities = {flag.severity for flag in flags}
        if RiskLevel.critical in severities:
            return RiskLevel.critical
        if RiskLevel.high in severities:
            return RiskLevel.high
        if RiskLevel.medium in severities:
            return RiskLevel.medium
        return RiskLevel.low

    def _confidence(
        self,
        required: list[str],
        missing_fields: list[str],
        field_confidence: dict[str, float],
        risk_flags: list[RiskFlag],
    ) -> float:
        if not required:
            return 0.48
        coverage = (len(required) - len(missing_fields)) / len(required)
        avg_confidence = (
            sum(field_confidence.values()) / len(field_confidence) if field_confidence else 0.5
        )
        high_risk_penalty = (
            0.08
            if any(flag.severity in {RiskLevel.high, RiskLevel.critical} for flag in risk_flags)
            else 0
        )
        return round(
            max(0.1, min(0.98, coverage * 0.58 + avg_confidence * 0.42 - high_risk_penalty)), 3
        )

    def _is_missing(self, value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}

    def _add_confidence(
        self, confidence: dict[str, float], name: str, value: Any, score: float = 0.8
    ) -> None:
        if not self._is_missing(value):
            confidence[name] = score

    def _clean_value(self, value: str | None) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", value).strip(" .,:;-")

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()
