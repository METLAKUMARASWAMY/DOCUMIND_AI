import re
from typing import ClassVar
from uuid import uuid4

from app.models.schemas import CrossDocumentCheck, DocumentRecord, DocumentType, Evidence


class CrossDocumentVerificationService:
    payment_term_patterns: ClassVar[dict[str, re.Pattern[str]]] = {
        "net 15": re.compile(r"\bnet\s*15\b", re.IGNORECASE),
        "net 30": re.compile(r"\bnet\s*30\b", re.IGNORECASE),
        "net 45": re.compile(r"\bnet\s*45\b", re.IGNORECASE),
        "net 60": re.compile(r"\bnet\s*60\b", re.IGNORECASE),
        "due on receipt": re.compile(r"\bdue\s+on\s+receipt\b", re.IGNORECASE),
        "prepayment": re.compile(
            r"\b(?:prepayment|pre-pay(?:ment)?|upfront|advance payment)\b", re.IGNORECASE
        ),
    }
    exception_markers: ClassVar[tuple[str, ...]] = (
        "approved exception",
        "exception approved",
        "policy exception",
        "waiver",
        "approved waiver",
        "temporary override",
        "deviation approved",
        "approved deviation",
    )

    def verify(self, documents: list[DocumentRecord]) -> list[CrossDocumentCheck]:
        checks = []
        checks.extend(self._invoice_po_amount_match(documents))
        checks.extend(self._invoice_po_reference_match(documents))
        checks.extend(self._invoice_contract_vendor_match(documents))
        checks.extend(self._contract_invoice_terms_match(documents))
        checks.extend(self._vendor_bank_details_match(documents))
        checks.extend(self._bank_change_request_checks(documents))
        checks.extend(self._exception_checks(documents))
        return checks

    def _invoice_po_amount_match(self, documents: list[DocumentRecord]) -> list[CrossDocumentCheck]:
        invoices = [doc for doc in documents if doc.document_type == DocumentType.invoice]
        pos = [doc for doc in documents if doc.document_type == DocumentType.purchase_order]
        if not invoices or not pos:
            return []
        invoice_amounts = self._normalized_field_values(invoices, "amount", self._normalize_amount)
        po_amounts = self._normalized_field_values(pos, "amount", self._normalize_amount)
        overlap = invoice_amounts & po_amounts
        if overlap:
            status = "Passed"
            confidence = 0.92
        else:
            status = "Failed"
            confidence = 0.81
        return [
            CrossDocumentCheck(
                check_id=f"xdoc-{uuid4().hex[:10]}",
                title="Invoice amount matches purchase order amount",
                status=status,
                confidence=confidence,
                evidence=self._pair_evidence(
                    invoices,
                    pos,
                    "Invoice and purchase order amounts were compared for a direct match.",
                ),
            )
        ]

    def _contract_invoice_terms_match(
        self, documents: list[DocumentRecord]
    ) -> list[CrossDocumentCheck]:
        contracts = [doc for doc in documents if doc.document_type == DocumentType.contract]
        invoices = [doc for doc in documents if doc.document_type == DocumentType.invoice]
        if not contracts or not invoices:
            return []
        contract_terms = self._payment_terms(contracts)
        invoice_terms = self._payment_terms(invoices)
        overlap = contract_terms & invoice_terms
        if overlap:
            status = "Passed"
            confidence = 0.9
        else:
            status = "Failed"
            confidence = 0.83
        return [
            CrossDocumentCheck(
                check_id=f"xdoc-{uuid4().hex[:10]}",
                title="Contract payment terms align with invoice conditions",
                status=status,
                confidence=confidence,
                evidence=self._pair_evidence(
                    contracts,
                    invoices,
                    "Contract payment terms were compared against invoice conditions.",
                ),
            )
        ]

    def _invoice_po_reference_match(
        self, documents: list[DocumentRecord]
    ) -> list[CrossDocumentCheck]:
        invoices = [doc for doc in documents if doc.document_type == DocumentType.invoice]
        pos = [doc for doc in documents if doc.document_type == DocumentType.purchase_order]
        if not invoices or not pos:
            return []
        invoice_refs = self._normalized_field_values(
            invoices, "purchase_order_number", self._normalize_po
        )
        po_numbers = self._normalized_field_values(pos, "purchase_order_number", self._normalize_po)
        if not invoice_refs or not po_numbers:
            return [
                CrossDocumentCheck(
                    check_id=f"xdoc-{uuid4().hex[:10]}",
                    title="Invoice references purchase order",
                    status="Review Required",
                    confidence=0.69,
                    evidence=self._pair_evidence(
                        invoices,
                        pos,
                        "Invoice and purchase order identifiers were not both available for matching.",
                    ),
                    recommendation="Request invoice PO reference or purchase order document before approval.",
                )
            ]
        overlap = invoice_refs & po_numbers
        return [
            CrossDocumentCheck(
                check_id=f"xdoc-{uuid4().hex[:10]}",
                title="Invoice PO reference matches purchase order",
                status="Passed" if overlap else "Failed",
                confidence=0.9 if overlap else 0.78,
                evidence=self._pair_evidence(
                    invoices,
                    pos,
                    "Invoice PO reference was compared against purchase order number.",
                ),
                left_value=", ".join(sorted(invoice_refs)),
                right_value=", ".join(sorted(po_numbers)),
                recommendation=None if overlap else "Resolve PO reference mismatch before payment.",
            )
        ]

    def _invoice_contract_vendor_match(
        self, documents: list[DocumentRecord]
    ) -> list[CrossDocumentCheck]:
        invoices = [doc for doc in documents if doc.document_type == DocumentType.invoice]
        contracts = [doc for doc in documents if doc.document_type == DocumentType.contract]
        if not invoices or not contracts:
            return []
        invoice_vendors = self._normalized_field_values(invoices, "vendor")
        contract_vendors = self._normalized_field_values(contracts, "vendor")
        if not invoice_vendors or not contract_vendors:
            return []
        overlap = invoice_vendors & contract_vendors
        return [
            CrossDocumentCheck(
                check_id=f"xdoc-{uuid4().hex[:10]}",
                title="Invoice vendor matches contract vendor",
                status="Passed" if overlap else "Failed",
                confidence=0.9 if overlap else 0.79,
                evidence=self._pair_evidence(
                    invoices,
                    contracts,
                    "Invoice vendor was compared against contract vendor.",
                ),
                left_value=", ".join(sorted(invoice_vendors)),
                right_value=", ".join(sorted(contract_vendors)),
                recommendation=None if overlap else "Resolve vendor mismatch before approval.",
            )
        ]

    def _vendor_bank_details_match(
        self, documents: list[DocumentRecord]
    ) -> list[CrossDocumentCheck]:
        vendor_docs = [
            doc
            for doc in documents
            if doc.document_type in {DocumentType.invoice, DocumentType.purchase_order}
        ]
        bank_docs = [doc for doc in documents if doc.document_type == DocumentType.bank_statement]
        if not vendor_docs or not bank_docs:
            return []
        vendors = self._normalized_field_values(vendor_docs, "vendor")
        vendors.update(
            self._text_label_values(vendor_docs, ("vendor", "supplier", "seller", "company"))
        )
        bank_details = self._normalized_bank_values(bank_docs)
        if not vendors or not bank_details:
            return []
        overlap = vendors & bank_details
        if overlap:
            status = "Passed"
            confidence = 0.89
        else:
            status = "Failed"
            confidence = 0.8
        return [
            CrossDocumentCheck(
                check_id=f"xdoc-{uuid4().hex[:10]}",
                title="Vendor name matches bank details",
                status=status,
                confidence=confidence,
                evidence=self._pair_evidence(
                    vendor_docs,
                    bank_docs,
                    "Vendor identity was compared with bank account holder and beneficiary details.",
                ),
            )
        ]

    def _bank_change_request_checks(
        self, documents: list[DocumentRecord]
    ) -> list[CrossDocumentCheck]:
        hits = [
            doc
            for doc in documents
            if "bank account change" in doc.text.lower()
            or "new bank account" in doc.text.lower()
            or "updated banking details" in doc.text.lower()
        ]
        if not hits:
            return []
        return [
            CrossDocumentCheck(
                check_id=f"xdoc-{uuid4().hex[:10]}",
                title="Bank account change request detected",
                status="Review Required",
                confidence=0.86,
                evidence=[
                    self._evidence(
                        doc, "Bank account change language requires independent verification."
                    )
                    for doc in hits[:2]
                ],
                recommendation="Hold payment until bank account change is verified outside email.",
            )
        ]

    def _exception_checks(self, documents: list[DocumentRecord]) -> list[CrossDocumentCheck]:
        hits = [doc for doc in documents if self._has_exception_language(doc.text)]
        if not hits:
            return []
        return [
            CrossDocumentCheck(
                check_id=f"xdoc-{uuid4().hex[:10]}",
                title="Approved exception or waiver detected",
                status="Review Required",
                confidence=0.87,
                evidence=[
                    self._evidence(doc, "Exception or waiver language found in the source text.")
                    for doc in hits[:2]
                ],
            )
        ]

    def _normalized_field_values(
        self,
        documents: list[DocumentRecord],
        field_name: str,
        normalizer=None,
    ) -> set[str]:
        values = set()
        normalizer = normalizer or self._normalize_text
        for document in documents:
            for field in document.extracted_fields:
                if field.name == field_name:
                    normalized = normalizer(field.value)
                    if normalized:
                        values.add(normalized)
        return values

    def _payment_terms(self, documents: list[DocumentRecord]) -> set[str]:
        terms = set()
        for document in documents:
            lower = document.text.lower()
            for label, pattern in self.payment_term_patterns.items():
                if pattern.search(lower):
                    terms.add(label)
            for field in document.extracted_fields:
                if field.name == "payment_terms":
                    normalized = self._normalize_text(field.value)
                    if normalized:
                        terms.add(normalized)
        return terms

    def _bank_detail_values(self, document: DocumentRecord) -> set[str]:
        values = set()
        for field in document.extracted_fields:
            if field.name in {"bank_account_holder", "bank_identifier", "vendor"}:
                normalized = self._normalize_text(field.value)
                if normalized:
                    values.add(normalized)
        return values

    def _normalized_bank_values(self, documents: list[DocumentRecord]) -> set[str]:
        values = set()
        for document in documents:
            values.update(self._bank_detail_values(document))
            lower = document.text.lower()
            if (
                "beneficiary" in lower
                or "account holder" in lower
                or "account name" in lower
                or "payee" in lower
            ):
                values.update(self._extract_labeled_values(document.text))
        return values

    def _extract_labeled_values(self, text: str) -> set[str]:
        values = set()
        for marker in (
            "account holder",
            "account name",
            "beneficiary",
            "beneficiary name",
            "payee",
        ):
            index = text.lower().find(marker)
            if index < 0:
                continue
            candidate = text[index + len(marker) :].lstrip(": -\t\n\r")
            candidate = candidate.splitlines()[0].strip()
            if candidate:
                values.add(self._normalize_text(candidate))
        return values

    def _text_label_values(
        self, documents: list[DocumentRecord], labels: tuple[str, ...]
    ) -> set[str]:
        values = set()
        for document in documents:
            lower = document.text.lower()
            for label in labels:
                index = lower.find(label)
                if index < 0:
                    continue
                candidate = document.text[index + len(label) :].lstrip(": -\t\n\r")
                candidate = candidate.splitlines()[0].strip()
                if candidate:
                    values.add(self._normalize_text(candidate))
        return values

    def _pair_evidence(
        self,
        left_documents: list[DocumentRecord],
        right_documents: list[DocumentRecord],
        reasoning: str,
    ) -> list[Evidence]:
        evidence = []
        for document in left_documents[:1] + right_documents[:1]:
            evidence.append(self._evidence(document, reasoning))
        return evidence

    def _has_exception_language(self, text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in self.exception_markers)

    def _normalize_text(self, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value.strip().lower())
        return cleaned.strip(" .,:;-")

    def _normalize_amount(self, value: str) -> str:
        cleaned = re.sub(r"[^0-9.]", "", value)
        if cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
        return cleaned.lstrip("0") or "0"

    def _normalize_po(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", value).lower()
        cleaned = cleaned.removeprefix("po")
        return cleaned.lstrip("0") or cleaned

    def _evidence(self, document: DocumentRecord, reasoning: str) -> Evidence:
        return Evidence(
            document_id=document.document_id,
            source_document=document.filename,
            page_number=1,
            text=document.text[:360],
            confidence=0.76,
            extraction_method="cross_document_rule",
            reasoning=reasoning,
        )
