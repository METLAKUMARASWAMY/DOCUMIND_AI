from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
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
DEDUPE_NARRATIVE_TYPES = {
    VisionFinanceNarrativeType.monthly_invoice_summary,
    VisionFinanceNarrativeType.vendor_payment_summary,
    VisionFinanceNarrativeType.tax_summary,
    VisionFinanceNarrativeType.outstanding_payment_summary,
}


@dataclass(frozen=True)
class _DuplicateInvoiceExclusion:
    normalized_invoice_number: str
    normalized_vendor: str
    retained: VisionDocumentRecord
    excluded: tuple[VisionDocumentRecord, ...]


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
        selected, duplicate_exclusions = self._select_with_duplicate_exclusions(
            request,
            documents,
        )
        if not selected:
            raise ValueError(
                "No approved documents with verified fields match the requested narrative scope"
            )

        calculations: list[VisionCalculationTrace] = []
        excluded_documents = sorted(
            (item for exclusion in duplicate_exclusions for item in exclusion.excluded),
            key=lambda item: item.document_id,
        )
        citations = self._citations(selected)
        citations.extend(self._duplicate_exclusion_citations(excluded_documents))
        citations = sorted(
            {citation.evidence_id: citation for citation in citations}.values(),
            key=lambda citation: (
                citation.document_id,
                citation.page_number,
                citation.evidence_id or "",
            ),
        )
        date_range = {"from": request.date_from, "to": request.date_to}
        self._trace_duplicate_invoice_exclusions(
            calculations,
            duplicate_exclusions,
            date_range,
        )
        narrative = self._compose(request.narrative_type, selected, calculations, date_range)
        return VisionFinanceNarrative(
            tenant_id=tenant_id,
            narrative_type=request.narrative_type,
            narrative=narrative,
            calculations=calculations,
            citations=citations,
            document_ids=[document.document_id for document in selected],
            source_document_ids=sorted(
                {document.document_id for document in [*selected, *excluded_documents]}
            ),
            excluded_document_ids=[document.document_id for document in excluded_documents],
            all_values_verified=True,
            created_at=utc_now_iso(),
            generated_by=actor,
        )

    def _select(
        self,
        request: VisionFinanceNarrativeRequest,
        documents: list[VisionDocumentRecord],
    ) -> list[VisionDocumentRecord]:
        selected, _ = self._select_with_duplicate_exclusions(request, documents)
        return selected

    def _select_with_duplicate_exclusions(
        self,
        request: VisionFinanceNarrativeRequest,
        documents: list[VisionDocumentRecord],
    ) -> tuple[list[VisionDocumentRecord], list[_DuplicateInvoiceExclusion]]:
        requested_ids = set(request.document_ids)
        eligible = []
        for document in documents:
            if document.status not in VERIFIED_DOCUMENT_STATES:
                continue
            if requested_ids and document.document_id not in requested_ids:
                continue
            fields = self._verified_fields(document)
            vendor = self._value(fields, "vendor_name", "party_name", "supplier_name")
            if request.vendor_name and self._normalise(vendor) != self._normalise(
                request.vendor_name
            ):
                continue
            document_date = self._document_date(fields)
            if request.date_from and (
                not document_date or document_date < self._date(request.date_from)
            ):
                continue
            if request.date_to and (
                not document_date or document_date > self._date(request.date_to)
            ):
                continue
            if fields:
                eligible.append(document)

        invoice_groups: dict[tuple[str, str], list[VisionDocumentRecord]] = defaultdict(list)
        if request.narrative_type in DEDUPE_NARRATIVE_TYPES:
            for document in eligible:
                if document.document_type != VisionDocumentType.invoice:
                    continue
                fields = self._verified_fields(document)
                invoice_number = self._value(fields, "invoice_number")
                vendor = self._value(fields, "vendor_name", "party_name", "supplier_name")
                normalized_invoice_number = self._normalise_invoice_number(invoice_number)
                normalized_vendor = self._normalise(vendor)
                if normalized_invoice_number and normalized_vendor:
                    invoice_groups[(normalized_vendor, normalized_invoice_number)].append(document)

        exclusions = []
        excluded_document_ids: set[str] = set()
        for (normalized_vendor, normalized_invoice_number), group in sorted(invoice_groups.items()):
            if len(group) < 2:
                continue
            conflicts = self._duplicate_invoice_conflicts(group)
            if conflicts:
                raise ValueError(
                    "Conflicting approved invoices share verified vendor and invoice number "
                    f"({normalized_vendor}, {normalized_invoice_number}); review fields: "
                    f"{', '.join(conflicts)}"
                )
            retained = min(group, key=self._duplicate_invoice_preference)
            excluded = tuple(
                sorted(
                    (item for item in group if item.document_id != retained.document_id),
                    key=lambda item: item.document_id,
                )
            )
            if not excluded:
                continue
            excluded_document_ids.update(item.document_id for item in excluded)
            exclusions.append(
                _DuplicateInvoiceExclusion(
                    normalized_invoice_number=normalized_invoice_number,
                    normalized_vendor=normalized_vendor,
                    retained=retained,
                    excluded=excluded,
                )
            )

        selected = sorted(
            [
                document
                for document in eligible
                if document.document_id not in excluded_document_ids
            ],
            key=lambda document: document.document_id,
        )
        return selected, exclusions

    def _duplicate_invoice_preference(self, document: VisionDocumentRecord) -> tuple[int, int, str]:
        verified_completeness = len(self._verified_fields(document))
        approved_rank = 0 if document.status == VisionProcessingStatus.approved else 1
        return approved_rank, -verified_completeness, document.document_id

    def _duplicate_invoice_conflicts(
        self,
        documents: list[VisionDocumentRecord],
    ) -> list[str]:
        conflicts = []
        for field_name in (
            "total_amount",
            "subtotal",
            "tax",
            "outstanding_amount",
            "currency",
            "invoice_date",
            "purchase_order_number",
            "payment_status",
            "status",
        ):
            values = {
                self._normalise_comparison_value(
                    field_name,
                    self._value(self._verified_fields(document), field_name),
                )
                for document in documents
            } - {""}
            if len(values) > 1:
                conflicts.append(field_name)
        return conflicts

    def _normalise_comparison_value(self, field_name: str, value: str) -> str:
        if not value:
            return ""
        if field_name in {"total_amount", "subtotal", "tax", "outstanding_amount"}:
            amount = self._decimal(value)
            return format(amount, "f") if amount is not None else self._normalise(value)
        if field_name == "currency":
            return (self._currency(value) or value).upper().strip()
        if field_name == "invoice_date":
            try:
                return self._date(value).isoformat()
            except ValueError:
                return self._normalise(value)
        return self._normalise(value)

    def _trace_duplicate_invoice_exclusions(
        self,
        calculations: list[VisionCalculationTrace],
        exclusions: list[_DuplicateInvoiceExclusion],
        date_range: dict[str, str | None],
    ) -> None:
        for exclusion in exclusions:
            excluded_ids = ",".join(item.document_id for item in exclusion.excluded)
            self._trace(
                calculations,
                "Duplicate invoice documents excluded",
                str(len(exclusion.excluded)),
                [exclusion.retained, *exclusion.excluded],
                (
                    "deduplicate(exact normalised verified invoice_number="
                    f"{exclusion.normalized_invoice_number}, verified_vendor="
                    f"{exclusion.normalized_vendor}); "
                    f"retained_document_id={exclusion.retained.document_id}; "
                    f"excluded_document_ids={excluded_ids}"
                ),
                date_range,
                field_names=("invoice_number",),
            )

    def _duplicate_exclusion_citations(
        self,
        documents: list[VisionDocumentRecord],
    ) -> list[VisionAskCitation]:
        citations = []
        for document in documents:
            invoice_number = self._verified_fields(document).get("invoice_number")
            if invoice_number is None:
                continue
            evidence = invoice_number.evidence
            citations.append(
                VisionAskCitation(
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=evidence.page_number,
                    source_text=evidence.source_text,
                    bounding_boxes=[evidence.bounding_box] if evidence.bounding_box else [],
                    evidence_id=evidence.evidence_id,
                    model_confidence=evidence.model_confidence,
                    model_name=evidence.model_name,
                    model_version=evidence.model_version,
                    processing_timestamp=evidence.processing_timestamp,
                    reviewer_status=evidence.reviewer_status,
                    extraction_status="excluded_duplicate_source",
                    evidence_type="duplicate_exclusion_evidence",
                )
            )
        return citations

    def _compose(
        self,
        narrative_type: VisionFinanceNarrativeType,
        documents: list[VisionDocumentRecord],
        calculations: list[VisionCalculationTrace],
        date_range: dict[str, str | None],
    ) -> str:
        if narrative_type == VisionFinanceNarrativeType.monthly_invoice_summary:
            invoices = [
                item for item in documents if item.document_type == VisionDocumentType.invoice
            ]
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
                self._value(
                    self._verified_fields(item), "vendor_name", "party_name", "supplier_name"
                )
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
            if (
                status in {"outstanding", "unpaid", "overdue", "pending"}
                or "outstanding_amount" in fields
            ):
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
        count = self._count_trace(
            "Documents with reviewed indicators", affected, calculations, date_range
        )
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
        contracts = [
            item for item in documents if item.document_type == VisionDocumentType.contract
        ]
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
        raw = self._value(
            fields, "invoice_date", "approval_date", "effective_date", "agreement_date"
        )
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
            self._format_currency(currency, value) for currency, value in sorted(totals.items())
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

    def _normalise_invoice_number(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return "".join(character for character in normalized if character.isalnum())


verified_finance_narrative_service = VerifiedFinanceNarrativeService()
