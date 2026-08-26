import re
from collections.abc import Iterable
from typing import ClassVar

from app.models.schemas import DocumentRecord, Evidence, ExtractedField


class KnowledgeExtractionService:
    amount_pattern = re.compile(
        r"(?:USD|INR|EUR|GBP|\$)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
        r"|\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b"
        r"|\b\d+\.\d{2}\b",
        re.IGNORECASE,
    )
    date_pattern = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})\b",
        re.IGNORECASE,
    )
    vendor_pattern = re.compile(
        r"\b(?:vendor|supplier|seller|contractor)\s*[:#-]\s*([^\n\r:]{2,100})",
        re.IGNORECASE,
    )
    company_pattern = re.compile(
        r"\b(?:customer|buyer|bill to|company)\s*[:#-]\s*([^\n\r:]{2,100})",
        re.IGNORECASE,
    )
    invoice_pattern = re.compile(r"(?:invoice|inv)[\s#:-]+([A-Z0-9-]{4,40})", re.IGNORECASE)
    po_pattern = re.compile(
        r"(?:purchase order|po reference|references po|po number|po)[\s#:-]+([A-Z0-9-]{4,40})",
        re.IGNORECASE,
    )
    bank_holder_pattern = re.compile(
        r"(?:account holder|account name|beneficiary(?: name)?|payee)[:\s-]+([^\n\r:]{2,80})",
        re.IGNORECASE,
    )
    bank_identifier_pattern = re.compile(
        r"\b(?:iban|swift|bic|routing number|account number)[:\s-]*([A-Z0-9-]{6,34})",
        re.IGNORECASE,
    )

    clause_keywords: ClassVar[dict[str, tuple[str, ...]]] = {
        "termination_clause": ("termination", "terminate", "notice period"),
        "confidentiality_clause": ("confidential", "non-disclosure", "nda"),
        "indemnity_clause": ("indemnify", "indemnification", "hold harmless"),
        "governing_law_clause": ("governing law", "jurisdiction", "venue"),
        "payment_terms": ("net 30", "payment terms", "due date", "amount due"),
        "audit_rights_clause": ("audit rights", "audit cooperation", "right to audit"),
    }

    def extract(self, document: DocumentRecord) -> list[ExtractedField]:
        fields: list[ExtractedField] = []
        text = document.text
        fields.extend(self._extract_many("amount", self.amount_pattern.finditer(text), document))
        fields.extend(self._extract_many("date", self.date_pattern.finditer(text), document))
        fields.extend(
            self._extract_many(
                "vendor",
                self.vendor_pattern.finditer(text),
                document,
                group=1,
                cleaner=self._clean_party_value,
            )
        )
        fields.extend(
            self._extract_many(
                "company",
                self.company_pattern.finditer(text),
                document,
                group=1,
                cleaner=self._clean_party_value,
            )
        )
        fields.extend(self._extract_many("invoice_number", self.invoice_pattern.finditer(text), document, group=1))
        fields.extend(self._extract_many("purchase_order_number", self.po_pattern.finditer(text), document, group=1))
        fields.extend(self._extract_many("bank_account_holder", self.bank_holder_pattern.finditer(text), document, group=1))
        fields.extend(self._extract_many("bank_identifier", self.bank_identifier_pattern.finditer(text), document, group=1))

        lower = text.lower()
        for name, keywords in self.clause_keywords.items():
            match = next((keyword for keyword in keywords if self._affirmed_keyword(lower, keyword)), None)
            if match:
                fields.append(
                    ExtractedField(
                        name=name,
                        value=match,
                        normalized_value=self._normalize(match),
                        confidence=0.79,
                        evidence=self._evidence(document, match, f"Detected {name.replace('_', ' ')}."),
                    )
                )
        return self._dedupe(fields)

    def _extract_many(
        self,
        name: str,
        matches: Iterable[re.Match],
        document: DocumentRecord,
        group: int = 0,
        cleaner=None,
    ) -> list[ExtractedField]:
        fields = []
        for match in list(matches)[:12]:
            value = match.group(group).strip()
            if cleaner:
                value = cleaner(value)
            if len(value) < 2:
                continue
            fields.append(
                ExtractedField(
                    name=name,
                    value=value,
                    normalized_value=self._normalize(value),
                    confidence=0.76,
                    evidence=self._evidence(document, value, f"Extracted {name} using pattern and context."),
                )
            )
        return fields

    def _evidence(self, document: DocumentRecord, value: str, reasoning: str) -> Evidence:
        index = max(document.text.lower().find(value.lower()), 0)
        start = max(index - 90, 0)
        end = min(index + len(value) + 90, len(document.text))
        snippet = document.text[start:end].strip() or document.text[:240]
        return Evidence(
            document_id=document.document_id,
            source_document=document.filename,
            page_number=1,
            text=snippet,
            confidence=0.78,
            extraction_method="regex_rule",
            reasoning=reasoning,
        )

    def _dedupe(self, fields: list[ExtractedField]) -> list[ExtractedField]:
        seen = set()
        result = []
        for field in fields:
            key = (field.name, field.value.lower())
            if key not in seen:
                seen.add(key)
                result.append(field)
        return result

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower()).strip(" .,:;-")

    def _clean_party_value(self, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip(" .,:;-")
        cleaned = re.split(
            r"\b(?:effective date|payment terms|governing law|services|termination|"
            r"confidentiality|audit rights|manual overrides?|purchase order|related contract|"
            r"approved amount|approval status|invoice|bill to|amount due|due date|subject)\b",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,:;-")
        if self._invalid_party_value(cleaned):
            return ""
        return cleaned

    def _invalid_party_value(self, value: str) -> bool:
        lower = value.lower()
        if len(value.split()) > 6:
            return True
        if any(term in lower for term in (" will ", " provide ", " support ", " process payment")):
            return True
        if lower.startswith(("will ", "provide ", "process ")):
            return True
        return False

    def _affirmed_keyword(self, lower_text: str, keyword: str) -> bool:
        start = 0
        while True:
            index = lower_text.find(keyword, start)
            if index < 0:
                return False
            context = lower_text[max(0, index - 90) : index + len(keyword) + 50]
            if not self._negates_keyword(context):
                return True
            start = index + len(keyword)

    def _negates_keyword(self, context: str) -> bool:
        negations = (
            "does not include",
            "doesn't include",
            "does not contain",
            "not contain",
            "not include",
            "without",
            "missing",
            "not found",
            "no ",
            "lacks",
        )
        return any(negation in context for negation in negations)
