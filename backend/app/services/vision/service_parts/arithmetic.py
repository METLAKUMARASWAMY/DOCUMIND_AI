"""Deterministic calculation answers with auditable inputs.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionAskCalculationInput,
    VisionAskCalculationOperation,
    VisionAskCalculationTrace,
    VisionAskCitation,
    VisionAskResponse,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionProcessingStatus,
)


class ArithmeticMixin:
    def _answer_arithmetic(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del top_k, tenant_id
        lower = question.lower()
        tokens = set(re.findall(r"[a-z0-9]+", lower))
        selected_documents = self._documents_for_question(question, documents)
        if any(term in lower for term in ("date interval", "days between")):
            return self._answer_date_interval(question, selected_documents)
        if any(term in lower for term in ("count", "how many", "number of")) and "invoice" in lower:
            return self._answer_verified_invoice_count(selected_documents)
        invoice_po_question = bool(tokens & {"invoice", "invoices"}) and (
            "po" in tokens or "purchase order" in lower
        )
        if invoice_po_question and (
            "percentage" in tokens or "percent" in tokens or "%" in question
        ):
            return self._answer_invoice_po_variance(selected_documents)
        if invoice_po_question and bool(
            tokens & {"difference", "delta", "differ", "different", "minus", "variance", "versus"}
        ):
            return self._answer_invoice_po_difference(selected_documents)
        return self._answer_decimal_sum(question, selected_documents)

    def _answer_decimal_sum(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        requested = self._requested_field_names(question)
        numeric_names = requested & {"subtotal", "tax", "discount", "total_amount"}
        if not numeric_names:
            numeric_names = {"total_amount"}
        if "invoice" in question.lower():
            documents = [
                document
                for document in documents
                if document.document_type == VisionDocumentType.invoice
            ]
        parsed: list[tuple[VisionDocumentRecord, ExtractedVisionField, Decimal, str | None]] = []
        for document in documents:
            for field_name in sorted(numeric_names):
                field = self._verified_calculation_field(document, field_name)
                if field is None:
                    continue
                value = self._decimal_value(self._calculation_field_value(field))
                if value is not None:
                    parsed.append(
                        (document, field, value, self._calculation_currency(document, field))
                    )
        if not parsed:
            return self._not_found_answer(requires_review=False)
        inputs = [
            self._calculation_input(
                document,
                field,
                role="addend",
                normalized_value=format(value, "f"),
                value_type="decimal",
                currency=currency,
            )
            for document, field, value, currency in parsed
        ]
        citations = [self._field_citation(field) for _, field, _, _ in parsed]
        currency, currency_error = self._consistent_calculation_currency(
            [item[3] for item in parsed]
        )
        if currency_error:
            return self._calculation_guardrail_response(
                operation=VisionAskCalculationOperation.sum,
                inputs=inputs,
                citations=citations,
                formula="sum(Decimal(verified input))",
                message=currency_error,
            )
        total = sum((value for _, _, value, _ in parsed), Decimal(0)).quantize(Decimal("0.01"))
        return VisionAskResponse(
            answer=f"{currency + ' ' if currency else ''}{total}",
            answer_status="supported",
            citations=citations,
            confidence=(
                self._mean_available_field_confidence(field for _, field, _, _ in parsed) or 0.0
            ),
            reasoning_summary=(
                f"Summed {len(parsed)} reviewed value(s) using deterministic Decimal arithmetic."
            ),
            requires_review=False,
            retriever="deterministic-decimal-aggregation",
            retrieval_path="deterministic_calculation",
            calculation_trace=VisionAskCalculationTrace(
                operation=VisionAskCalculationOperation.sum,
                inputs=inputs,
                result=format(total, ".2f"),
                result_type="decimal",
                currency=currency,
                formula="sum(Decimal(verified input))",
            ),
        )

    def _answer_verified_invoice_count(
        self,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        counted: list[tuple[VisionDocumentRecord, ExtractedVisionField]] = []
        verified_document_states = {
            VisionProcessingStatus.approved,
            VisionProcessingStatus.completed,
        }
        for document in documents:
            if (
                document.document_type != VisionDocumentType.invoice
                or document.status not in verified_document_states
            ):
                continue
            identity = self._verified_calculation_field(document, "invoice_number")
            if identity is not None:
                counted.append((document, identity))
        inputs = [
            self._calculation_input(
                document,
                field,
                role="counted_invoice",
                normalized_value=self._calculation_field_value(field),
                value_type="document",
            )
            for document, field in counted
        ]
        count = len({document.document_id for document, _ in counted})
        return VisionAskResponse(
            answer=f"{count} verified invoice(s).",
            answer_status="supported",
            citations=[self._field_citation(field) for _, field in counted],
            confidence=(
                self._mean_available_field_confidence(field for _, field in counted) or 1.0
            ),
            reasoning_summary=(
                "Counted distinct approved or completed invoice documents with a reviewed "
                "invoice-number field."
            ),
            requires_review=False,
            retriever="deterministic-document-count",
            retrieval_path="deterministic_calculation",
            calculation_trace=VisionAskCalculationTrace(
                operation=VisionAskCalculationOperation.count_verified_invoices,
                inputs=inputs,
                result=str(count),
                result_type="integer",
                unit="invoices",
                formula=(
                    "count(distinct approved/completed invoice document_id with reviewed "
                    "invoice_number evidence)"
                ),
            ),
        )

    def _answer_invoice_po_difference(
        self,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        return self._answer_invoice_po_calculation(
            documents,
            VisionAskCalculationOperation.invoice_po_difference,
        )

    def _answer_invoice_po_variance(
        self,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        return self._answer_invoice_po_calculation(
            documents,
            VisionAskCalculationOperation.percentage_variance,
        )

    def _answer_invoice_po_calculation(
        self,
        documents: list[VisionDocumentRecord],
        operation: VisionAskCalculationOperation,
    ) -> VisionAskResponse:
        pair = self._linked_invoice_po_pair(documents)
        if pair is None:
            return VisionAskResponse(
                answer=(
                    "Select one invoice and its linked purchase order so the calculation has "
                    "an unambiguous evidence pair."
                ),
                answer_status="clarification_needed",
                confidence=0.0,
                reasoning_summary="No unique reviewed invoice-to-purchase-order pair was found.",
                requires_review=False,
                retriever="deterministic-calculation-engine-v1",
                retrieval_path="calculation_pair_resolution",
            )
        invoice, purchase_order = pair
        invoice_total = self._verified_calculation_field(invoice, "total_amount")
        po_total = self._verified_calculation_field(purchase_order, "total_amount")
        if invoice_total is None or po_total is None:
            return self._not_found_answer(requires_review=False)
        invoice_value = self._decimal_value(self._calculation_field_value(invoice_total))
        po_value = self._decimal_value(self._calculation_field_value(po_total))
        if invoice_value is None or po_value is None:
            return self._not_found_answer(requires_review=False)
        invoice_currency = self._calculation_currency(invoice, invoice_total)
        po_currency = self._calculation_currency(purchase_order, po_total)
        inputs = [
            self._calculation_input(
                invoice,
                invoice_total,
                role="invoice_amount",
                normalized_value=format(invoice_value, "f"),
                value_type="decimal",
                currency=invoice_currency,
            ),
            self._calculation_input(
                purchase_order,
                po_total,
                role="purchase_order_amount",
                normalized_value=format(po_value, "f"),
                value_type="decimal",
                currency=po_currency,
            ),
        ]
        citations = [
            self._field_citation(invoice_total),
            self._field_citation(po_total),
        ]
        currency, currency_error = self._consistent_calculation_currency(
            [invoice_currency, po_currency]
        )
        formula = (
            "((Decimal(invoice_total) - Decimal(purchase_order_total)) / "
            "Decimal(purchase_order_total)) * 100"
            if operation == VisionAskCalculationOperation.percentage_variance
            else "Decimal(invoice_total) - Decimal(purchase_order_total)"
        )
        if currency_error:
            return self._calculation_guardrail_response(
                operation=operation,
                inputs=inputs,
                citations=citations,
                formula=formula,
                message=currency_error,
            )
        difference = invoice_value - po_value
        if operation == VisionAskCalculationOperation.percentage_variance:
            if po_value == 0:
                return self._calculation_guardrail_response(
                    operation=operation,
                    inputs=inputs,
                    citations=citations,
                    formula=formula,
                    message=(
                        "Percentage variance cannot be calculated because the reviewed "
                        "purchase-order total is zero."
                    ),
                )
            result = ((difference / po_value) * Decimal(100)).quantize(Decimal("0.01"))
            answer = f"The invoice-to-purchase-order percentage variance is {result}%."
            result_type = "percentage"
            unit = "percent"
            result_currency = None
        else:
            result = difference.quantize(Decimal("0.01"))
            answer = (
                "Invoice total minus purchase-order total is "
                f"{currency + ' ' if currency else ''}{result}."
            )
            result_type = "decimal"
            unit = None
            result_currency = currency
        return VisionAskResponse(
            answer=answer,
            answer_status="supported",
            citations=citations,
            confidence=self._mean_available_field_confidence([invoice_total, po_total]) or 0.0,
            reasoning_summary=(
                "Used one uniquely linked invoice/PO pair and reviewed totals with deterministic "
                "Decimal arithmetic."
            ),
            requires_review=False,
            retriever="deterministic-calculation-engine-v1",
            retrieval_path="deterministic_calculation",
            calculation_trace=VisionAskCalculationTrace(
                operation=operation,
                inputs=inputs,
                result=format(result, ".2f"),
                result_type=result_type,
                currency=result_currency,
                unit=unit,
                formula=formula,
            ),
        )

    def _answer_date_interval(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        lower = question.lower()
        if "order date" in lower or "delivery date" in lower:
            start_name, end_name = "order_date", "delivery_date"
        elif "effective date" in lower or "expiry date" in lower:
            start_name, end_name = "effective_date", "expiry_date"
        else:
            start_name, end_name = "invoice_date", "due_date"
        candidates: list[
            tuple[
                VisionDocumentRecord,
                ExtractedVisionField,
                ExtractedVisionField,
                datetime,
                datetime,
            ]
        ] = []
        for document in documents:
            start_field = self._verified_calculation_field(document, start_name)
            end_field = self._verified_calculation_field(document, end_name)
            if start_field is None or end_field is None:
                continue
            start_date = self._date(self._calculation_field_value(start_field))
            end_date = self._date(self._calculation_field_value(end_field))
            if start_date is not None and end_date is not None:
                candidates.append((document, start_field, end_field, start_date, end_date))
        if len(candidates) > 1:
            return VisionAskResponse(
                answer="Select one document so the requested date interval is unambiguous.",
                answer_status="clarification_needed",
                confidence=0.0,
                reasoning_summary="More than one document contains the reviewed date pair.",
                requires_review=False,
                retriever="deterministic-calculation-engine-v1",
                retrieval_path="calculation_pair_resolution",
            )
        if not candidates:
            return self._not_found_answer(requires_review=False)
        document, start_field, end_field, start_date, end_date = candidates[0]
        interval = (end_date.date() - start_date.date()).days
        inputs = [
            self._calculation_input(
                document,
                start_field,
                role="start_date",
                normalized_value=start_date.date().isoformat(),
                value_type="date",
            ),
            self._calculation_input(
                document,
                end_field,
                role="end_date",
                normalized_value=end_date.date().isoformat(),
                value_type="date",
            ),
        ]
        return VisionAskResponse(
            answer=(
                f"The interval from {start_name.replace('_', ' ')} to "
                f"{end_name.replace('_', ' ')} is {interval} day(s)."
            ),
            answer_status="supported",
            citations=[self._field_citation(start_field), self._field_citation(end_field)],
            confidence=self._mean_available_field_confidence([start_field, end_field]) or 0.0,
            reasoning_summary="Parsed reviewed dates and subtracted calendar dates deterministically.",
            requires_review=False,
            retriever="deterministic-date-calculation",
            retrieval_path="deterministic_calculation",
            calculation_trace=VisionAskCalculationTrace(
                operation=VisionAskCalculationOperation.date_interval,
                inputs=inputs,
                result=str(interval),
                result_type="integer",
                unit="days",
                formula="date(end_date) - date(start_date)",
            ),
        )

    def _linked_invoice_po_pair(
        self,
        documents: list[VisionDocumentRecord],
    ) -> tuple[VisionDocumentRecord, VisionDocumentRecord] | None:
        invoices = [item for item in documents if item.document_type == VisionDocumentType.invoice]
        purchase_orders = [
            item for item in documents if item.document_type == VisionDocumentType.purchase_order
        ]
        linked = []
        for invoice in invoices:
            invoice_reference = self._verified_calculation_field(invoice, "purchase_order_number")
            if invoice_reference is None:
                continue
            normalized_reference = self._normalize(self._calculation_field_value(invoice_reference))
            for purchase_order in purchase_orders:
                po_reference = self._verified_calculation_field(
                    purchase_order, "purchase_order_number"
                )
                if (
                    po_reference is not None
                    and normalized_reference
                    and normalized_reference
                    == self._normalize(self._calculation_field_value(po_reference))
                ):
                    linked.append((invoice, purchase_order))
        if len(linked) == 1:
            return linked[0]
        if not linked and len(invoices) == 1 and len(purchase_orders) == 1:
            return invoices[0], purchase_orders[0]
        return None

    def _verified_calculation_field(
        self,
        document: VisionDocumentRecord,
        field_name: str,
    ) -> ExtractedVisionField | None:
        return next(
            (
                field
                for field in document.extracted_fields
                if field.name == field_name
                and field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
            ),
            None,
        )

    def _calculation_field_value(self, field: ExtractedVisionField) -> str:
        if field.reviewed_value is not None:
            return field.reviewed_value
        return field.normalized_value or field.value

    def _calculation_input(
        self,
        document: VisionDocumentRecord,
        field: ExtractedVisionField,
        *,
        role: str,
        normalized_value: str,
        value_type: str,
        currency: str | None = None,
    ) -> VisionAskCalculationInput:
        return VisionAskCalculationInput(
            document_id=document.document_id,
            document_type=document.document_type,
            role=role,
            field_name=field.name,
            raw_value=self._calculation_field_value(field),
            normalized_value=normalized_value,
            value_type=value_type,
            currency=currency,
            evidence_id=field.evidence.evidence_id,
            page_number=field.evidence.page_number,
            reviewer_status=field.reviewer_status,
        )

    def _calculation_currency(
        self,
        document: VisionDocumentRecord,
        amount_field: ExtractedVisionField,
    ) -> str | None:
        currency_field = self._verified_calculation_field(document, "currency")
        candidates = []
        if currency_field is not None:
            candidates.append(self._calculation_field_value(currency_field))
        candidates.append(self._calculation_field_value(amount_field))
        known_codes = {
            "AED",
            "AUD",
            "CAD",
            "CHF",
            "CNY",
            "EUR",
            "GBP",
            "INR",
            "JPY",
            "SGD",
            "USD",
        }
        symbol_codes = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
        for candidate in candidates:
            upper = candidate.upper()
            code = next(
                (item for item in re.findall(r"\b[A-Z]{3}\b", upper) if item in known_codes),
                None,
            )
            if code:
                return code
            for symbol, symbol_code in symbol_codes.items():
                if symbol in candidate:
                    return symbol_code
        return None

    def _consistent_calculation_currency(
        self,
        currencies: list[str | None],
    ) -> tuple[str | None, str | None]:
        known = {currency for currency in currencies if currency}
        if len(known) > 1:
            return None, (
                "The calculation spans multiple currencies and cannot be combined without "
                "verified conversion data."
            )
        if known and any(currency is None for currency in currencies):
            return None, (
                "The calculation mixes currency-labelled and unlabelled values, so it was "
                "stopped until every input currency is verified."
            )
        return next(iter(known), None), None

    def _calculation_guardrail_response(
        self,
        *,
        operation: VisionAskCalculationOperation,
        inputs: list[VisionAskCalculationInput],
        citations: list[VisionAskCitation],
        formula: str,
        message: str,
    ) -> VisionAskResponse:
        return VisionAskResponse(
            answer=message,
            answer_status="partially_supported",
            citations=citations,
            confidence=0.0,
            reasoning_summary="Deterministic calculation stopped at a calculation guardrail.",
            requires_review=True,
            retriever="deterministic-calculation-engine-v1",
            retrieval_path="calculation_guardrail",
            calculation_trace=VisionAskCalculationTrace(
                operation=operation,
                inputs=inputs,
                result=None,
                result_type="not_calculated",
                formula=formula,
            ),
        )

    def _decimal_value(self, value: str) -> Decimal | None:
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
        if not match:
            return None
        try:
            return Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return None
