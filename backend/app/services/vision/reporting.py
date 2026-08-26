from __future__ import annotations

import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation

from app.core.security import utc_now_iso
from app.models.vision import (
    ReviewerStatus,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionFinanceNarrative,
    VisionReportRecord,
    VisionReportRequest,
    VisionReportType,
)


class VisionReportService:
    def generate(
        self,
        request: VisionReportRequest,
        documents: list[VisionDocumentRecord],
        narratives: list[VisionFinanceNarrative],
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionReportRecord:
        selected = self._select_documents(request, documents)
        if request.report_type == VisionReportType.finance_narrative and request.narrative_id:
            narrative = next(
                (item for item in narratives if item.narrative_id == request.narrative_id),
                None,
            )
            if narrative:
                narrative_documents = set(narrative.source_document_ids or narrative.document_ids)
                selected = [item for item in documents if item.document_id in narrative_documents]
        content = self._content(request, selected, narratives)
        return VisionReportRecord(
            tenant_id=tenant_id,
            report_type=request.report_type,
            title=self._title(request.report_type),
            document_ids=[item.document_id for item in selected],
            content=content,
            generated_by=actor,
            created_at=utc_now_iso(),
        )

    def markdown(self, report: VisionReportRecord) -> str:
        return "\n".join(
            [
                f"# {report.title}",
                "",
                f"Report ID: {report.report_id}",
                f"Generated: {report.created_at}",
                "",
                "```json",
                json.dumps(report.content, indent=2, ensure_ascii=True),
                "```",
                "",
                "Risk terminology policy: screening indicators require human review. Only an authorised reviewer may assign CONFIRMED_FRAUD.",
            ]
        )

    def pdf(self, report: VisionReportRecord) -> bytes:
        lines = [line[:105] for line in self.markdown(report).splitlines() if line.strip()][:58]
        stream_lines = []
        for index, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            size = 15 if index == 0 else 8
            stream_lines.append(f"BT /F1 {size} Tf 36 {770 - index * 12} Td ({escaped}) Tj ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream",
        ]
        output = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for number, body in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{number} 0 obj\n".encode("ascii"))
            output.extend(body)
            output.extend(b"\nendobj\n")
        xref_offset = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        return bytes(output)

    def _select_documents(
        self,
        request: VisionReportRequest,
        documents: list[VisionDocumentRecord],
    ) -> list[VisionDocumentRecord]:
        ids = set(request.document_ids)
        if request.document_id:
            ids.add(request.document_id)
        selected = [item for item in documents if not ids or item.document_id in ids]
        if request.vendor_name:
            vendor = self._normalise(request.vendor_name)
            selected = [item for item in selected if self._normalise(self._vendor(item)) == vendor]
        if (
            request.report_type
            in {
                VisionReportType.document_intelligence,
                VisionReportType.risk_screening,
                VisionReportType.contract_review,
            }
            and not selected
        ):
            raise ValueError("At least one accessible document is required for this report type")
        if request.report_type == VisionReportType.contract_review:
            selected = [
                item for item in selected if item.document_type == VisionDocumentType.contract
            ]
            if not selected:
                raise ValueError("A contract document is required for a contract review report")
        return selected

    def _content(
        self,
        request: VisionReportRequest,
        documents: list[VisionDocumentRecord],
        narratives: list[VisionFinanceNarrative],
    ) -> dict:
        if request.report_type == VisionReportType.document_intelligence:
            return {
                "documents": [self._document_payload(item) for item in documents],
                "risk_policy": self._risk_policy(),
            }
        if request.report_type == VisionReportType.risk_screening:
            return {
                "documents": [
                    {
                        "document_id": item.document_id,
                        "status": item.status.value,
                        "risk_indicators": [
                            risk.model_dump(mode="json") for risk in item.risk_indicators
                        ],
                        "duplicate_matches": [
                            match.model_dump(mode="json") for match in item.duplicate_candidates
                        ],
                        "tamper_screening": [
                            screen.model_dump(mode="json") for screen in item.tamper_analysis
                        ],
                        "anomaly_screening": [
                            screen.model_dump(mode="json") for screen in item.anomaly_analysis
                        ],
                        "audit_history": [
                            event.model_dump(mode="json") for event in item.audit_events
                        ],
                    }
                    for item in documents
                ],
                "risk_policy": self._risk_policy(),
            }
        if request.report_type == VisionReportType.contract_review:
            return {
                "contracts": [
                    {
                        "document_id": item.document_id,
                        "clauses": [
                            field.model_dump(mode="json")
                            for field in item.extracted_fields
                            if field.name in self._contract_fields()
                        ],
                        "review_indicators": [
                            risk.model_dump(mode="json") for risk in item.risk_indicators
                        ],
                        "reviewer_decision": item.status.value,
                    }
                    for item in documents
                ],
                "legal_policy": "Clause indicators require qualified human legal review and are not legal judgements.",
            }
        if request.report_type == VisionReportType.vendor_summary:
            return self._vendor_summary(documents, request.vendor_name)
        if request.report_type == VisionReportType.finance_narrative:
            narrative = next(
                (item for item in narratives if item.narrative_id == request.narrative_id),
                None,
            )
            if not narrative:
                raise ValueError("narrative_id must identify an accessible generated narrative")
            return narrative.model_dump(mode="json")
        if not request.evaluation:
            raise ValueError(
                "Actual labelled evaluation output is required for a model evaluation report"
            )
        if not request.evaluation.get("dataset") or not request.evaluation.get("metrics"):
            raise ValueError(
                "Evaluation output must include dataset provenance and calculated metrics"
            )
        return {
            "evaluation": request.evaluation,
            "metric_policy": "Only metrics calculated from the identified labelled dataset are included.",
        }

    def _document_payload(self, document: VisionDocumentRecord) -> dict:
        return {
            "document_id": document.document_id,
            "filename": document.filename,
            "document_type": document.document_type.value,
            "status": document.status.value,
            "summary": self._summary(document),
            "ocr_layout": {
                "ocr_word_count": len(document.ocr_words),
                "ocr_line_count": len(document.ocr_lines),
                "layout_region_count": len(document.layout_regions),
                "table_count": len(document.tables),
                "page_artifacts": [
                    artifact.model_dump(mode="json") for artifact in document.page_artifacts
                ],
                "quality": [item.model_dump(mode="json") for item in document.quality],
            },
            "field_completeness": document.field_completeness.model_dump(mode="json"),
            "fields": [self._field_payload(field) for field in document.extracted_fields],
            "validation_results": [
                risk.model_dump(mode="json")
                for risk in document.risk_indicators
                if risk.category in {"validation", "cross_document"}
            ],
            "risk_indicators": [risk.model_dump(mode="json") for risk in document.risk_indicators],
            "duplicate_matches": [
                item.model_dump(mode="json") for item in document.duplicate_candidates
            ],
            "tables": [table.model_dump(mode="json") for table in document.tables],
            "reviewer_decision": document.status.value,
            "model_runs": [run.model_dump(mode="json") for run in document.model_runs],
            "audit_history": [event.model_dump(mode="json") for event in document.audit_events],
            "graph": {
                "nodes": [
                    node.model_dump(mode="json") if hasattr(node, "model_dump") else dict(node)
                    for node in document.graph.nodes
                ],
                "edges": [
                    edge.model_dump(mode="json") if hasattr(edge, "model_dump") else dict(edge)
                    for edge in document.graph.edges
                ],
            },
            "limitations": {
                "confidence_calibrated": False,
                "requires_human_review": document.status.value
                in {"REVIEW_REQUIRED", "UNDER_REVIEW"},
                "risk_language": self._risk_policy(),
                "structured_extraction_status": (document.metadata.get("extraction") or {}).get(
                    "status"
                ),
            },
        }

    def _field_payload(self, field) -> dict:
        payload = field.model_dump(mode="json")
        payload["display_value"] = self._mask(field.name, field.reviewed_value or field.value)
        payload.pop("value", None)
        payload.pop("reviewed_value", None)
        return payload

    def _vendor_summary(
        self,
        documents: list[VisionDocumentRecord],
        requested_vendor: str | None,
    ) -> dict:
        groups: dict[str, list[VisionDocumentRecord]] = defaultdict(list)
        for document in documents:
            vendor = self._vendor(document)
            if vendor:
                groups[vendor].append(document)
        if requested_vendor and not groups:
            raise ValueError("No accessible documents match vendor_name")
        vendors = []
        for vendor, records in sorted(groups.items()):
            totals: dict[str, Decimal] = defaultdict(Decimal)
            evidence_ids = []
            for document in records:
                fields = {field.name: field for field in document.extracted_fields}
                amount_field = fields.get("total_amount")
                currency_field = fields.get("currency")
                if not amount_field or amount_field.reviewer_status not in {
                    ReviewerStatus.accepted,
                    ReviewerStatus.edited,
                }:
                    continue
                amount = self._decimal(amount_field.reviewed_value or amount_field.value)
                if amount is None:
                    continue
                currency = (
                    currency_field.reviewed_value or currency_field.value
                    if currency_field
                    and currency_field.reviewer_status
                    in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    else "UNSPECIFIED"
                )
                totals[currency] += amount
                evidence_ids.append(amount_field.evidence.evidence_id)
            vendors.append(
                {
                    "vendor_name": vendor,
                    "document_count": len(records),
                    "document_ids": [item.document_id for item in records],
                    "verified_totals": {key: format(value, ".2f") for key, value in totals.items()},
                    "source_evidence_ids": evidence_ids,
                    "calculation_function": "sum(Decimal(verified total_amount)) grouped by verified currency",
                }
            )
        return {"vendors": vendors, "risk_policy": self._risk_policy()}

    def _vendor(self, document: VisionDocumentRecord) -> str:
        for name in ("vendor_name", "party_name", "supplier_name"):
            field = next((item for item in document.extracted_fields if item.name == name), None)
            if field:
                return field.reviewed_value or field.value
        return ""

    def _summary(self, document: VisionDocumentRecord) -> str:
        completeness = document.field_completeness
        required_expected = (
            str(completeness.required_expected_count)
            if completeness.required_expected_count is not None
            else "unavailable"
        )
        required_matched = (
            str(completeness.required_matched_count)
            if completeness.required_matched_count is not None
            else "unavailable"
        )
        required_missing = (
            str(completeness.required_missing_count)
            if completeness.required_missing_count is not None
            else "unavailable"
        )
        return (
            f"{document.filename} is classified as {document.document_type.value}, has "
            f"{document.page_count} page(s), {len(document.extracted_fields)} extracted field(s), "
            f"and {len(document.risk_indicators)} review indicator(s). Field completeness: "
            f"required expected {required_expected}, required matched {required_matched}, "
            f"required missing {required_missing}, optional extracted "
            f"{completeness.optional_extracted_count}, total extracted "
            f"{completeness.total_extracted_count}, duplicates rejected "
            f"{completeness.duplicates_rejected_count}, disputed "
            f"{completeness.disputed_count}."
        )

    def _mask(self, name: str, value: str) -> str:
        if "bank" not in name.lower() and "account" not in name.lower():
            return value
        compact = re.sub(r"\s+", "", value)
        return "*" * max(4, len(compact) - 4) + compact[-4:]

    def _decimal(self, value: str) -> Decimal | None:
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value or "")
        if not match:
            return None
        try:
            return Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return None

    def _normalise(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _title(self, report_type: VisionReportType) -> str:
        return report_type.value.replace("_", " ").title()

    def _risk_policy(self) -> str:
        return "Screening indicators require human review; this report does not automatically confirm fraud."

    def _contract_fields(self) -> set[str]:
        return {
            "parties",
            "agreement_date",
            "effective_date",
            "expiry_date",
            "governing_law",
            "renewal_term",
            "termination_for_convenience",
            "notice_period",
            "anti_assignment",
            "exclusivity",
            "non_compete",
            "audit_rights",
            "insurance",
            "cap_on_liability",
            "uncapped_liability",
            "perpetual_licence",
            "revenue_sharing",
            "change_of_control",
            "confidentiality",
            "indemnification",
            "payment_terms",
        }


vision_report_service = VisionReportService()
