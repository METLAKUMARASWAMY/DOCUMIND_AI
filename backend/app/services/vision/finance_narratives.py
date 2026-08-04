from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from app.core.security import utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionAskCitation,
    VisionCalculationTrace,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionFinanceNarrative,
    VisionFinanceNarrativeRequest,
    VisionFinanceNarrativeType,
    VisionProcessingStatus,
)

VERIFIED_DOCUMENT_STATES = {
    VisionProcessingStatus.approved,
    VisionProcessingStatus.completed,
}
VERIFIED_FIELD_STATES = {ReviewerStatus.accepted, ReviewerStatus.edited}


class VerifiedFinanceNarrativeService:
    """Build finance narratives from reviewed values and deterministic calculations."""

    def generate(
        self,
        request: VisionFinanceNarrativeRequest,
        documents: list[VisionDocumentRecord],
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionFinanceNarrative:
        selected = self._select(request, documents)
        if not selected:
            raise ValueError(
                "No approved documents with verified fields match the requested narrative scope"
            )

        calculations: list[VisionCalculationTrace] = []
        citations = self._citations(selected)
        date_range = {"from": request.date_from, "to": request.date_to}
        narrative = self._compose(request.narrative_type, selected, calculations, date_range)
        return VisionFinanceNarrative(
            tenant_id=tenant_id,
            narrative_type=request.narrative_type,
            narrative=narrative,
            calculations=calculations,
            citations=citations,
            document_ids=[document.document_id for document in selected],
            all_values_verified=True,
            created_at=utc_now_iso(),
            generated_by=actor,
        )

    def _select(
        self,
        request: VisionFinanceNarrativeRequest,
        documents: list[VisionDocumentRecord],
    ) -> list[VisionDocumentRecord]:
        requested_ids = set(request.document_ids)
        selected = []
        for document in documents:
            if document.status not in VERIFIED_DOCUMENT_STATES:
                continue
            if requested_ids and document.document_id not in requested_ids:
                continue
            fields = self._verified_fields(document)
            vendor = self._value(fields, "vendor_name", "party_name", "supplier_name")
            if request.vendor_name and self._normalise(vendor) != self._normalise(request.vendor_name):
                continue
            document_date = self._document_date(fields)
            if request.date_from and (not document_date or document_date < self._date(request.date_from)):
                continue
            if request.date_to and (not document_date or document_date > self._date(request.date_to)):
                continue
            if fields:
                selected.append(document)
        return selected

    def _compose(
        self,
        narrative_type: VisionFinanceNarrativeType,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        if narrative_type == VisionFinanceNarrativeType.monthly_invoice_summary:
            invoices = [item for item in documents if item.document_type == VisionDocumentType.invoice]
            return self._invoice_summary(invoices, calculations, date_range)
        if narrative_type == VisionFinanceNarrativeType.vendor_payment_summary:
            return self._payment_summary(documents, calculations, date_range)
        if narrative_type == VisionFinanceNarrativeType.tax_summary:
            return self._tax_summary(documents, calculations, date_range)
        if narrative_type == VisionFinanceNarrativeType.outstanding_payment_summary:
            return self._outstanding_summary(documents, calculations, date_range)
        if narrative_type == VisionFinanceNarrativeType.anomaly_summary:
            return self._anomaly_summary(documents, calculations, date_range)
        if narrative_type == VisionFinanceNarrativeType.approval_status_summary:
            return self._approval_summary(documents, calculations, date_range)
        return self._contract_summary(documents, calculations, date_range)

    def _invoice_summary(
        self,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        if not documents:
            raise ValueError("No approved invoices match the requested narrative scope")
        count = self._count_trace("Verified invoices", documents, calculations, date_range)
        totals = self._sum_by_currency(documents, "total_amount", calculations, date_range)
        rendered = self._render_currency_totals(totals) or "no verified total amount"
        return f"The selected period contains {count} verified invoice(s) with combined value {rendered}."

    def _payment_summary(
        self,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        count = self._count_trace("Verified finance documents", documents, calculations, date_range)
        totals = self._sum_by_currency(documents, "total_amount", calculations, date_range)
        vendors = sorted(
            {
                self._value(self._verified_fields(item), "vendor_name", "party_name", "supplier_name")
                for item in documents
            }
            - {""}
        )
        self._trace(
            calculations,
            "Verified vendors",
            str(len(vendors)),
            documents,
            "count(distinct normalised verified vendor_name)",
            date_range,
            field_names=("vendor_name", "party_name", "supplier_name"),
        )
        rendered = self._render_currency_totals(totals) or "no verified payment total"
        return f"Across {len(vendors)} verified vendor(s), {count} document(s) record {rendered}."

    def _tax_summary(
        self,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        totals = self._sum_by_currency(documents, "tax", calculations, date_range)
        if not totals:
            raise ValueError("The approved documents do not contain verified tax values")
        return f"Verified tax across the selected documents totals {self._render_currency_totals(totals)}."

    def _outstanding_summary(
        self,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        outstanding = []
        for document in documents:
            fields = self._verified_fields(document)
            status = self._normalise(self._value(fields, "payment_status", "status"))
            if status in {"outstanding", "unpaid", "overdue", "pending"} or "outstanding_amount" in fields:
                outstanding.append(document)
        if not outstanding:
            raise ValueError(
                "The approved documents do not contain verified outstanding-payment status or amounts"
            )
        count = self._count_trace("Outstanding documents", outstanding, calculations, date_range)
        totals = self._sum_by_currency(
            outstanding,
            "outstanding_amount",
            calculations,
            date_range,
            fallback_field="total_amount",
        )
        return f"{count} verified document(s) are outstanding, representing {self._render_currency_totals(totals)}."

    def _anomaly_summary(
        self,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        affected = [item for item in documents if item.risk_indicators]
        count = self._count_trace("Documents with reviewed indicators", affected, calculations, date_range)
        indicator_count = sum(len(item.risk_indicators) for item in affected)
        self._trace(
            calculations,
            "Risk indicators",
            str(indicator_count),
            affected,
            "sum(count(reviewed risk indicators per approved document))",
            date_range,
        )
        return (
            f"{count} approved document(s) contain {indicator_count} risk-screening indicator(s). "
            "These are review indicators and are not automatic findings of fraud."
        )

    def _approval_summary(
        self,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        counts = Counter(document.status.value for document in documents)
        for status, count in sorted(counts.items()):
            self._trace(
                calculations,
                f"Documents in {status}",
                str(count),
                [item for item in documents if item.status.value == status],
                f"count(document where status == {status})",
                date_range,
            )
        rendered = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))
        return f"Verified document approval states are {rendered}."

    def _contract_summary(
        self,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        contracts = [item for item in documents if item.document_type == VisionDocumentType.contract]
        if not contracts:
            raise ValueError("No approved contracts match the requested narrative scope")
        count = self._count_trace("Verified contracts", contracts, calculations, date_range)
        flagged = [item for item in contracts if item.risk_indicators]
        risk_count = sum(len(item.risk_indicators) for item in flagged)
        self._trace(
            calculations,
            "Contract review indicators",
            str(risk_count),
            flagged,
            "sum(count(contract risk indicators))",
            date_range,
        )
        return (
            f"The selected scope contains {count} verified contract(s) with {risk_count} review "
            "indicator(s). Clause indicators require legal review and are not legal judgements."
        )

    def _count_trace(
        self,
        label: str,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> int:
        count = len(documents)
        self._trace(
            calculations,
            label,
            str(count),
            documents,
            "count(distinct approved document_id)",
            date_range,
        )
        return count

    def _sum_by_currency(
        self,
        documents: list[VisionDocumentRecord],
        field_name: str,
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
        *,
        fallback_field: str | None = None,
    ) -> dict[str, Decimal]:
        totals: dict[str, Decimal] = defaultdict(Decimal)
        sources: dict[str, list[VisionDocumentRecord]] = defaultdict(list)
        field_names = (field_name, fallback_field) if fallback_field else (field_name,)
        for document in documents:
            fields = self._verified_fields(document)
            raw = self._value(fields, *field_names)
            amount = self._decimal(raw)
            if amount is None:
                continue
            currency = self._value(fields, "currency") or self._currency(raw) or "UNSPECIFIED"
            totals[currency] += amount
            sources[currency].append(document)
        for currency, total in sorted(totals.items()):
            self._trace(
                calculations,
                f"Verified {field_name} total",
                format(total, ".2f"),
                sources[currency],
                f"sum(Decimal(verified {field_name})) grouped by verified currency",
                date_range,
                currency=currency,
                field_names=tuple(name for name in field_names if name),
            )
        return dict(totals)

    def _trace(
        self,
        calculations: list[VisionCalculationTrace],
        label: str,
        value: str,
        documents: list[VisionDocumentRecord],
        function: str,
        date_range: dict[str, str | None],
        *,
        currency: str | None = None,
        field_names: tuple[str, ...] = (),
    ) -> None:
        evidence_ids = []
        for document in documents:
            fields = self._verified_fields(document)
            for name in field_names:
                if name in fields:
                    evidence_ids.append(fields[name].evidence.evidence_id)
        calculations.append(
            VisionCalculationTrace(
                label=label,
                value=value,
                currency=currency,
                source_document_ids=[item.document_id for item in documents],
                source_evidence_ids=sorted(set(evidence_ids)),
                date_range=date_range,
                calculation_function=function,
            )
        )

    def _citations(self, documents: list[VisionDocumentRecord]) -> list[VisionAskCitation]:
        citations = []
        for document in documents:
            for field in self._verified_fields(document).values():
                evidence = field.evidence
                citations.append(
                    VisionAskCitation(
                        document_id=document.document_id,
                        page_number=evidence.page_number,
                        source_text=evidence.source_text,
                        bounding_boxes=[evidence.bounding_box] if evidence.bounding_box else [],
                        evidence_id=evidence.evidence_id,
                        model_confidence=evidence.model_confidence,
                        model_name=evidence.model_name,
                        model_version=evidence.model_version,
                        processing_timestamp=evidence.processing_timestamp,
                        reviewer_status=evidence.reviewer_status,
                    )
                )
        unique = {item.evidence_id: item for item in citations if item.evidence_id}
        return list(unique.values())

    def _verified_fields(self, document: VisionDocumentRecord) -> dict[str, ExtractedVisionField]:
        return {
            field.name: field
            for field in document.extracted_fields
            if field.reviewer_status in VERIFIED_FIELD_STATES
        }

    def _document_date(self, fields: dict[str, ExtractedVisionField]) -> date | None:
        raw = self._value(fields, "invoice_date", "approval_date", "effective_date", "agreement_date")
        return self._date(raw) if raw else None

    def _value(self, fields: dict[str, ExtractedVisionField], *names: str | None) -> str:
        for name in names:
            if not name or name not in fields:
                continue
            field = fields[name]
            value = field.reviewed_value if field.reviewed_value is not None else field.value
            if value:
                return value
        return ""

    def _date(self, value: str) -> date:
        candidate = value.strip()[:10]
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            for pattern in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
                try:
                    return datetime.strptime(value.strip(), pattern).date()
                except ValueError:
                    continue
        raise ValueError(f"Invalid narrative date filter: {value}")

    def _decimal(self, value: str) -> Decimal | None:
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value or "")
        if not match:
            return None
        try:
            return Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return None

    def _currency(self, value: str) -> str | None:
        match = re.search(r"\b(?:INR|USD|EUR|GBP|JPY|AUD|CAD)\b", value or "", re.IGNORECASE)
        if match:
            return match.group(0).upper()
        if "₹" in (value or ""):
            return "INR"
        if "$" in (value or ""):
            return "USD"
        return None

    def _render_currency_totals(self, totals: dict[str, Decimal]) -> str:
        return ", ".join(
            self._format_currency(currency, value)
            for currency, value in sorted(totals.items())
        )

    def _format_currency(self, currency: str, value: Decimal) -> str:
        symbols = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
        fixed = f"{abs(value):.2f}"
        integer, decimals = fixed.split(".", 1)
        if currency == "INR" and len(integer) > 3:
            tail = integer[-3:]
            head = integer[:-3]
            groups = []
            while head:
                groups.append(head[-2:])
                head = head[:-2]
            integer = f"{','.join(reversed(groups))},{tail}"
        else:
            integer = f"{int(integer):,}"
        sign = "-" if value < 0 else ""
        symbol = symbols.get(currency)
        readable = (
            f"{sign}{symbol}{integer}.{decimals}"
            if symbol
            else f"{sign}{integer}.{decimals} {currency}"
        )
        return f"{readable} ({currency} {value:.2f})"

    def _normalise(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


verified_finance_narrative_service = VerifiedFinanceNarrativeService()
