"""Evidence-grounded risk rules over schema-independent semantic facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import ClassVar

from app.models.vision import (
    VisionDocumentRecord,
    VisionEvidence,
    VisionRiskSeverity,
    VisionSemanticType,
)
from app.services.vision.semantic_facts import SemanticFact, collect_semantic_facts
from app.services.vision.semantic_values import (
    SemanticContext,
    SemanticValueType,
    semantic_value_registry,
)


@dataclass(frozen=True)
class SemanticRiskFinding:
    rule_id: str
    title: str
    severity: VisionRiskSeverity
    category: str
    description: str
    recommended_action: str
    source_text: str
    affected_fields: tuple[str, ...]
    evidence: tuple[VisionEvidence, ...]
    confidence: float


class GenericSemanticRiskEngine:
    """Run conservative rules without assuming a registered document schema."""

    model_version = "semantic-risk-rules-v1"

    _BANK_NAMES: ClassVar[set[str]] = {
        "account_number",
        "bank_account",
        "bank_account_number",
        "beneficiary_account",
        "beneficiary_bank_account",
        "payment_account",
        "remittance_account",
        "requested_payment_account",
        "settlement_account",
    }
    _INVOICE_NUMBER_NAMES: ClassVar[set[str]] = {
        "bill_number",
        "invoice_id",
        "invoice_no",
        "invoice_number",
        "invoice_reference",
    }
    _INVOICE_DATE_NAMES: ClassVar[set[str]] = {
        "bill_date",
        "document_date",
        "invoice_date",
        "invoice_issued_on",
        "issued_date",
    }
    _DUE_DATE_NAMES: ClassVar[set[str]] = {
        "due_date",
        "due_on",
        "payment_deadline",
        "payment_due_date",
    }
    _AMOUNT_NAMES: ClassVar[dict[str, set[str]]] = {
        "subtotal": {"amount_before_tax", "net_amount", "subtotal"},
        "tax": {"sales_tax", "tax", "tax_amount", "vat_amount"},
        "discount": {"discount", "discount_amount"},
        "total_amount": {
            "amount_due",
            "grand_total",
            "invoice_total",
            "total_amount",
            "total_payable",
        },
    }

    def evaluate(
        self,
        document: VisionDocumentRecord,
        metadata: dict[str, object],
        corpus: list[VisionDocumentRecord] | tuple[VisionDocumentRecord, ...] = (),
    ) -> list[SemanticRiskFinding]:
        facts = collect_semantic_facts(document)
        findings: list[SemanticRiskFinding] = []
        bank = self._bank_account_mismatch(facts, metadata)
        if bank is not None and not self._legacy_bank_rule_covers(document):
            findings.append(bank)
        date_sequence = self._date_sequence(facts)
        if date_sequence is not None and not self._legacy_date_rule_covers(document):
            findings.append(date_sequence)
        amount = self._amount_arithmetic(facts)
        if amount is not None and not self._legacy_amount_rule_covers(document):
            findings.append(amount)
        duplicate = self._duplicate_invoice_identifier(document, facts, corpus)
        if duplicate is not None:
            findings.append(duplicate)
        return findings

    def _bank_account_mismatch(
        self,
        facts: list[SemanticFact],
        metadata: dict[str, object],
    ) -> SemanticRiskFinding | None:
        approved = next(
            (
                str(metadata[name])
                for name in (
                    "approved_bank_identifier",
                    "approved_bank_account",
                    "approved_payment_account",
                )
                if metadata.get(name)
            ),
            None,
        )
        if not approved:
            return None
        candidates = [
            fact
            for fact in facts
            if fact.name in self._BANK_NAMES
            or (
                fact.semantic_type == VisionSemanticType.bank_account
                and any(token in fact.name for token in ("account", "iban", "routing"))
            )
        ]
        observed = self._best(candidates, minimum_confidence=0.65)
        if observed is None:
            return None
        approved_normalized = self._normalize_account(approved, "approved bank account")
        observed_normalized = self._normalize_account(
            observed.normalized_value or observed.value,
            observed.name,
        )
        if not approved_normalized or not observed_normalized:
            return None
        if approved_normalized == observed_normalized:
            return None
        return SemanticRiskFinding(
            rule_id="business_risk.semantic_bank_account_mismatch",
            title="Bank Detail Mismatch",
            severity=VisionRiskSeverity.high,
            category="cross_document",
            description=(
                "A requested payment account differs from the supplied approved account; "
                "this is a suspicious indicator requiring human verification."
            ),
            recommended_action=(
                "Pause payment and independently verify the beneficiary account before approval."
            ),
            source_text=(
                f"document={self._mask_account(observed_normalized)}; "
                f"approved={self._mask_account(approved_normalized)}"
            ),
            affected_fields=(observed.name,),
            evidence=self._evidence(observed),
            confidence=observed.confidence,
        )

    def _date_sequence(self, facts: list[SemanticFact]) -> SemanticRiskFinding | None:
        invoice_fact = self._best(
            [fact for fact in facts if fact.name in self._INVOICE_DATE_NAMES],
            minimum_confidence=0.65,
        )
        due_fact = self._best(
            [fact for fact in facts if fact.name in self._DUE_DATE_NAMES],
            minimum_confidence=0.65,
        )
        if invoice_fact is None or due_fact is None:
            return None
        invoice_date = self._date(invoice_fact.normalized_value or invoice_fact.value)
        due_date = self._date(due_fact.normalized_value or due_fact.value)
        if invoice_date is None or due_date is None or due_date >= invoice_date:
            return None
        return SemanticRiskFinding(
            rule_id="business_risk.semantic_due_date_precedes_issue_date",
            title="Due Date Before Invoice Date",
            severity=VisionRiskSeverity.medium,
            category="validation",
            description=(
                "The observed payment deadline precedes the observed issue date and requires "
                "human verification."
            ),
            recommended_action="Verify the source dates and intended payment sequence.",
            source_text=(
                f"{invoice_fact.name}={invoice_fact.value}; {due_fact.name}={due_fact.value}"
            ),
            affected_fields=(invoice_fact.name, due_fact.name),
            evidence=self._evidence(invoice_fact, due_fact),
            confidence=min(invoice_fact.confidence, due_fact.confidence),
        )

    def _amount_arithmetic(self, facts: list[SemanticFact]) -> SemanticRiskFinding | None:
        selected = {
            concept: self._best(
                [fact for fact in facts if fact.name in names],
                minimum_confidence=0.65,
            )
            for concept, names in self._AMOUNT_NAMES.items()
        }
        if any(selected[name] is None for name in ("subtotal", "tax", "total_amount")):
            return None
        subtotal = self._decimal(selected["subtotal"])
        tax = self._decimal(selected["tax"])
        discount = self._decimal(selected["discount"]) or Decimal("0")
        total = self._decimal(selected["total_amount"])
        if subtotal is None or tax is None or total is None:
            return None
        expected = subtotal + tax - discount
        tolerance = max(Decimal("1.00"), abs(total) * Decimal("0.01"))
        if abs(expected - total) <= tolerance:
            return None
        observed = [fact for fact in selected.values() if fact is not None]
        return SemanticRiskFinding(
            rule_id="business_risk.semantic_amount_arithmetic_mismatch",
            title="Amount Arithmetic Mismatch",
            severity=VisionRiskSeverity.high,
            category="validation",
            description=(
                "The source-backed amount components do not reconcile within the configured "
                "tolerance and require human review."
            ),
            recommended_action="Review subtotal, tax, discount, and total before approval.",
            source_text=f"subtotal + tax - discount = {expected:.2f}; total = {total:.2f}",
            affected_fields=tuple(fact.name for fact in observed),
            evidence=self._evidence(*observed),
            confidence=min(fact.confidence for fact in observed),
        )

    def _duplicate_invoice_identifier(
        self,
        document: VisionDocumentRecord,
        current_facts: list[SemanticFact],
        corpus: list[VisionDocumentRecord] | tuple[VisionDocumentRecord, ...],
    ) -> SemanticRiskFinding | None:
        current = self._best(
            [fact for fact in current_facts if fact.name in self._INVOICE_NUMBER_NAMES],
            minimum_confidence=0.78,
        )
        if current is None or not current.normalized_value:
            return None
        matches: list[tuple[VisionDocumentRecord, SemanticFact]] = []
        current_value = self._identifier(current.normalized_value)
        for candidate in corpus:
            if candidate.document_id == document.document_id:
                continue
            if candidate.tenant_id != document.tenant_id:
                continue
            other = self._best(
                [
                    fact
                    for fact in collect_semantic_facts(candidate)
                    if fact.name in self._INVOICE_NUMBER_NAMES
                ],
                minimum_confidence=0.78,
            )
            if other is not None and self._identifier(other.normalized_value) == current_value:
                matches.append((candidate, other))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0].document_id)
        related, other = matches[0]
        return SemanticRiskFinding(
            rule_id="business_risk.duplicate_invoice_identifier",
            title="Possible Duplicate Invoice Identifier",
            severity=VisionRiskSeverity.medium,
            category="cross_document",
            description=(
                "The same source-backed invoice identifier appears on another document in "
                "this tenant; this is a duplicate concern, not a fraud conclusion."
            ),
            recommended_action="Compare the cited documents and verify whether reuse is expected.",
            source_text=(
                f"invoice_identifier={current.value}; related_document={related.document_id}"
            ),
            affected_fields=(current.name,),
            evidence=self._evidence(current, other),
            confidence=min(current.confidence, other.confidence),
        )

    def _best(
        self,
        facts: list[SemanticFact],
        *,
        minimum_confidence: float,
    ) -> SemanticFact | None:
        eligible = [
            fact
            for fact in facts
            if (fact.verified or fact.confidence >= minimum_confidence)
            and fact.evidence is not None
        ]
        return max(eligible, key=lambda item: (item.verified, item.confidence), default=None)

    def _evidence(self, *facts: SemanticFact) -> tuple[VisionEvidence, ...]:
        retained: dict[str, VisionEvidence] = {}
        for fact in facts:
            if fact.evidence is not None:
                retained.setdefault(fact.evidence.evidence_id, fact.evidence)
        return tuple(retained.values())

    def _normalize_account(self, value: str, label: str) -> str:
        validation = semantic_value_registry.validate(
            SemanticValueType.BANK_ACCOUNT,
            value,
            SemanticContext(label=label),
        )
        if validation.valid and validation.normalized_value:
            return validation.normalized_value
        compact = re.sub(r"[\s-]+", "", value).upper()
        return compact if re.fullmatch(r"[A-Z0-9]{4,34}", compact) else ""

    def _mask_account(self, value: str) -> str:
        return "*" * max(0, len(value) - 4) + value[-4:]

    def _date(self, value: str) -> date | None:
        cleaned = value.strip().rstrip(".")
        numeric = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2}|\d{4})", cleaned)
        if numeric and int(numeric.group(1)) <= 12 and int(numeric.group(2)) <= 12:
            # Without a locale or an unambiguous component, ordering these dates
            # could create a false sequence indicator.  Abstain instead.
            return None
        for date_format in (
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%m/%d/%y",
            "%d/%m/%y",
            "%B %d, %Y",
            "%d %B %Y",
        ):
            try:
                return datetime.strptime(cleaned, date_format).date()
            except ValueError:
                continue
        return None

    def _decimal(self, fact: SemanticFact | None) -> Decimal | None:
        if fact is None:
            return None
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", fact.normalized_value or fact.value)
        if not match:
            return None
        try:
            return Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return None

    def _identifier(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _legacy_bank_rule_covers(self, document: VisionDocumentRecord) -> bool:
        names = {field.name for field in document.extracted_fields}
        return bool(names & {"bank_account_number", "IFSC_or_bank_code"})

    def _legacy_date_rule_covers(self, document: VisionDocumentRecord) -> bool:
        names = {field.name for field in document.extracted_fields}
        return {"invoice_date", "due_date"} <= names

    def _legacy_amount_rule_covers(self, document: VisionDocumentRecord) -> bool:
        names = {field.name for field in document.extracted_fields}
        return {"subtotal", "tax", "total_amount"} <= names


generic_semantic_risk_engine = GenericSemanticRiskEngine()


__all__ = [
    "GenericSemanticRiskEngine",
    "SemanticRiskFinding",
    "generic_semantic_risk_engine",
]
